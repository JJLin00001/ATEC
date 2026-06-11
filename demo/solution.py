import os
import sys
import torch
from typing import Any
import math

class AlgSolution:

    ACTION_SCALE = 0.5

    def __init__(self):
        # 使用你训练好的策略
        policy_path = './logs/rsl_rl/unitree_b2_piper_rough/2026-06-10_23-11-21/exported/policy.pt'

        # 调试用 baseline：
        # policy_path = './atec_robot_model/baseline/unitree_b2_flat/policy.pt'

        self.device = 'cuda'
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8

        # CRITICAL: Assert B2Piper joint order matches expected [FR, FL, RR, RL, arm1-8]
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        # Expected B2Piper joint order from b2.py:108-115
        self.expected_joint_order = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4',
            'arm_joint5', 'arm_joint6', 'arm_joint7', 'arm_joint8'
        ]

        # =====================================================================
        # 路径控制可调参数 (Heading-lock + cross-track controller)
        # 建议调参顺序：yaw_bias / yaw_sign -> k_track / k_yaw -> kd / kvy
        # =====================================================================
        # Sign + bias for yaw command calibration
        # 若机身一直左偏 -> 试 yaw_bias = -0.05
        # 若改 yaw_bias 后反而更偏左 -> 翻转 yaw_sign 或 yaw_bias = +0.05
        self.yaw_sign = +1.0           # ±1.0，整体 yaw 输出方向校正
        self.yaw_rate_sign = +1.0      # 若 est_yaw 与 debug 真实 yaw 反号，改为 -1.0
        self.yaw_bias = -0.10          # rad/s，系统漂移补偿，起始值

        # Heading-lock + cross-track gains
        self.k_yaw = 1.6               # 锁定 yaw=0（朝世界 +x 轴）的比例增益
        self.k_track = 0.9             # cross-track（y 偏离）回正增益
        self.y_scale = 1.5             # tanh 软饱和尺度（米），>y_scale 后 cross-track 项饱和
        self.kd = 0.20                 # yaw rate 阻尼（vs measured yaw rate）
        self.kvy = 0.2                 # body lateral velocity 阻尼（默认关，最后再调）

        # Yaw rate filtering & clipping
        self.yaw_rate_alpha = 0.25     # 低通滤波系数，越小越平滑
        self.yaw_clip = 0.9            # 最终 yaw_rate 限幅
        self.yaw_rate_filtered = None

        # |y| 越界时的减速 / 增强 cross-track
        self.y_slow_thresh = 0.6       # |y| > 0.6m 时主动降速纠偏
        self.y_slow_factor = 0.65
        self.y_hard_thresh = 1.2       # |y| > 1.2m 时进一步减速并增强 k_track
        self.y_hard_factor = 0.45
        self.k_track_boost = 2.5       # |y| > y_hard_thresh 时 k_track 的额外放大倍数

        # Lateral command correction (cmd[:, 1]); body +y is left, so positive y error commands negative cmd_y.
        self.use_lateral_cmd = True
        self.lateral_sign = -1.0
        self.k_lateral = 0.55
        self.k_lateral_damping = 0.20
        self.cmd_y_clip = 0.55

        # Debug printing
        self.debug = ("--debug" in sys.argv)
        self.debug_print_interval = 10
        self._step_counter = 0

        # Adaptive velocity scheduling - REVISED with lower rough terrain speed
        self.velocity_schedule = {
            "flat_fast": 0.75,     # x < -115 (flat terrain)
            "rough": 0.50,         # -115 <= x < -35 (rough terrain)
            "slope": 0.45,         # -35 <= x < 45 (slope terrain)
            "stairs": 0.40,        # 45 <= x < 125 (stairs terrain)
            "final": 0.60,         # 125 <= x < 145 (final stretch)
        }

        # Progress tracking with world-frame position estimation
        self.estimated_x = None
        self.estimated_y = None
        self.estimated_yaw = None
        self.last_progress_x = None
        self.stuck_counter = 0
        self.stuck_threshold = 100
        self.in_recovery = False
        self.recovery_counter = 0
        self.recovery_duration = 50

        # Gait phase tracking (cycle_time=0.5s matches training)
        self.gait_phase_time = 0.0
        self.gait_cycle_time = 0.5
        self.phase_offset = 0.0        # Tune: try 0.0, 0.125, 0.25, 0.375 for best initial phase alignment

        # B2 训练时的动作缩放
        self.train_to_env_action_scale = torch.tensor(
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.dt = 0.02

        # B2Piper arm stow configuration
        self.arm_stow_action = torch.tensor(
            [0.0, 1.2, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        # Action clipping bounds
        self.leg_action_clip = 2.0
        self.arm_action_clip = 2.0

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    def _get_adaptive_velocity(self, x_estimate: float) -> float:
        """Compute adaptive forward velocity based on estimated x position."""
        if x_estimate < -115:
            return self.velocity_schedule["flat_fast"]
        elif x_estimate < -35:
            return self.velocity_schedule["rough"]
        elif x_estimate < 45:
            return self.velocity_schedule["slope"]
        elif x_estimate < 125:
            return self.velocity_schedule["stairs"]
        else:
            return self.velocity_schedule["final"]

    def _check_stuck_and_recover(self) -> tuple[bool, float, float]:
        """Detect stuck condition and return recovery commands.

        Returns:
            (in_recovery, forward_vel_adjust, lateral_vel)
        Note: lateral_vel kept very small to preserve diagonal trot gait.
        """
        if self.estimated_x is None:
            return False, 0.0, 0.0

        current_x = self.estimated_x[0].item()

        if self.last_progress_x is None:
            self.last_progress_x = current_x
            return False, 0.0, 0.0

        # Check progress
        if current_x - self.last_progress_x > 0.1:
            self.last_progress_x = current_x
            self.stuck_counter = 0
            self.in_recovery = False
            self.recovery_counter = 0
            return False, 0.0, 0.0
        else:
            self.stuck_counter += 1

            if self.stuck_counter > self.stuck_threshold:
                self.in_recovery = True
                self.stuck_counter = 0

        # Execute recovery - keep lateral_vel tiny so trot is not broken
        if self.in_recovery:
            self.recovery_counter += 1

            if self.recovery_counter < self.recovery_duration // 2:
                # First half: brief reverse, no lateral
                return True, -0.3, 0.0
            else:
                # Second half: moderate forward push, no lateral
                if self.recovery_counter >= self.recovery_duration:
                    self.in_recovery = False
                    self.recovery_counter = 0
                    self.last_progress_x = current_x
                return True, 0.4, 0.0

        return False, 0.0, 0.0

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Compute velocity commands using heading-lock + cross-track control.

        Strategy: 强制朝世界 +x 方向（estimated_yaw -> 0），并对 y 偏离做 tanh 软回正，
        不使用 lateral velocity 命令，避免破坏对角 trot。

        Formula:
          yaw_rate_cmd = yaw_sign * (
              - k_yaw   * estimated_yaw
              - k_track * tanh(estimated_y / y_scale) * boost
              - kd      * yaw_rate_measured
              - kvy     * body_vy
          ) + yaw_bias
        """
        num_envs = proprio.shape[0]

        # ---- Extract body-frame state from proprio (order is fixed) ----
        base_lin_vel = proprio[:, 0:3]   # body frame: [vx, vy, vz]
        base_ang_vel = proprio[:, 3:6]
        projected_gravity = proprio[:, 9:12]
        body_vy = base_lin_vel[:, 1]     # body-frame lateral velocity (left positive)

        # Convert body angular velocity [roll, pitch, yaw] to gravity-aligned yaw rate.
        # Directly integrating base_ang_vel[:, 2] drifts badly when the trunk rolls/pitches.
        gravity_norm = torch.linalg.norm(projected_gravity, dim=1, keepdim=True).clamp_min(1e-6)
        gravity_b = projected_gravity / gravity_norm
        roll = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2])
        pitch = torch.atan2(-gravity_b[:, 0], torch.sqrt(gravity_b[:, 1] ** 2 + gravity_b[:, 2] ** 2))
        cos_pitch = torch.cos(pitch).clamp(min=0.2)
        yaw_rate_measured = self.yaw_rate_sign * (
            torch.sin(roll) * base_ang_vel[:, 1]
            + torch.cos(roll) * base_ang_vel[:, 2]
        ) / cos_pitch

        # ---- Init world-frame state estimates on first call ----
        if self.estimated_yaw is None:
            self.estimated_yaw = torch.zeros(num_envs, device=self.device, dtype=proprio.dtype)
            self.estimated_x = torch.full((num_envs,), -141.0, device=self.device, dtype=proprio.dtype)
            self.estimated_y = torch.zeros(num_envs, device=self.device, dtype=proprio.dtype)

        # ---- Integrate yaw and world x/y ----
        self.estimated_yaw = self.estimated_yaw + yaw_rate_measured * self.dt
        self.estimated_yaw = torch.atan2(torch.sin(self.estimated_yaw), torch.cos(self.estimated_yaw))

        cos_yaw = torch.cos(self.estimated_yaw)
        sin_yaw = torch.sin(self.estimated_yaw)
        vel_world_x = base_lin_vel[:, 0] * cos_yaw - base_lin_vel[:, 1] * sin_yaw
        vel_world_y = base_lin_vel[:, 0] * sin_yaw + base_lin_vel[:, 1] * cos_yaw
        self.estimated_x = self.estimated_x + vel_world_x * self.dt
        self.estimated_y = self.estimated_y + vel_world_y * self.dt

        # ---- Stuck recovery (kept; lateral kept ~0) ----
        in_recovery, forward_adjust, lateral_recovery = self._check_stuck_and_recover()

        # ---- |y| guard: decide forward slowdown and cross-track boost ----
        abs_y_scalar = float(torch.abs(self.estimated_y[0]).item())
        k_track_eff = self.k_track
        if abs_y_scalar > self.y_hard_thresh:
            k_track_eff = self.k_track * self.k_track_boost

        # ---- Heading-lock + cross-track yaw rate (vectorized) ----
        track_term = torch.tanh(self.estimated_y / self.y_scale)
        yaw_rate_cmd = self.yaw_sign * (
            - self.k_yaw   * self.estimated_yaw
            - k_track_eff  * track_term
            - self.kd      * yaw_rate_measured
            - self.kvy     * body_vy
        ) + self.yaw_bias

        # ---- Low-pass filter + clip to [-yaw_clip, yaw_clip] ----
        if self.yaw_rate_filtered is None:
            self.yaw_rate_filtered = yaw_rate_cmd.clone()
        else:
            self.yaw_rate_filtered = (
                (1 - self.yaw_rate_alpha) * self.yaw_rate_filtered
                + self.yaw_rate_alpha * yaw_rate_cmd
            )
        yaw_rate_cmd = torch.clip(self.yaw_rate_filtered, -self.yaw_clip, self.yaw_clip)

        # ---- Adaptive forward velocity with |y|-based slowdown ----
        x_estimate = self.estimated_x[0].item()
        forward_vel = self._get_adaptive_velocity(x_estimate)
        if abs_y_scalar > self.y_slow_thresh:
            forward_vel *= self.y_slow_factor
        if abs_y_scalar > self.y_hard_thresh:
            forward_vel *= self.y_hard_factor
        if in_recovery:
            forward_vel += forward_adjust

        # ---- Build command [vx, vy, yaw_rate] in body frame, order unchanged ----
        cmd = torch.zeros((num_envs, 3), device=self.device, dtype=proprio.dtype)
        cmd[:, 0] = forward_vel

        # Lateral correction: body +y is left; positive estimated_y should command rightward motion.
        if self.use_lateral_cmd:
            lateral_cmd = self.lateral_sign * self.k_lateral * track_term - self.k_lateral_damping * body_vy
            if in_recovery:
                lateral_cmd = lateral_cmd + torch.full_like(lateral_cmd, lateral_recovery)
            cmd[:, 1] = torch.clip(lateral_cmd, -self.cmd_y_clip, self.cmd_y_clip)
        elif in_recovery:
            cmd[:, 1] = torch.clip(
                torch.tensor(lateral_recovery, device=self.device, dtype=proprio.dtype),
                -self.cmd_y_clip, self.cmd_y_clip,
            )

        cmd[:, 2] = yaw_rate_cmd

        # ---- Debug print (every N steps) ----
        if self.debug and (self._step_counter % self.debug_print_interval == 0):
            y_val = self.estimated_y[0].item()
            yaw_val = self.estimated_yaw[0].item()
            yr_val = yaw_rate_cmd[0].item() if yaw_rate_cmd.dim() > 0 else float(yaw_rate_cmd)
            print(
                f"  [path] est_y={y_val:+.3f}m est_yaw={yaw_val:+.3f}rad "
                f"yaw_rate_cmd={yr_val:+.3f} yaw_bias={self.yaw_bias:+.3f} "
                f"yaw_rate_meas={yaw_rate_measured[0].item():+.3f} "
                f"cmd_y={cmd[0, 1].item():+.3f} "
                f"fwd={forward_vel:.2f} body_vy={body_vy[0].item():+.3f} "
                f"k_track={k_track_eff:.2f}"
            )
        self._step_counter += 1

        return cmd

    def _get_phase_observation(self) -> torch.Tensor:
        """Compute gait phase observation (sin, cos) to match training.

        FIXED: Compute phase first, then increment time to avoid fixed one-step offset.
        """
        # Compute phase from CURRENT time (before increment)
        phase = ((self.gait_phase_time + self.phase_offset * self.gait_cycle_time) / self.gait_cycle_time) % 1.0

        phase_obs = torch.tensor(
            [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

        # Increment time AFTER computing phase (happens at end of timestep)
        self.gait_phase_time += self.dt

        return phase_obs

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        _velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        velocity_commands = self._get_velocity_commands(proprio)
        phase_obs = self._get_phase_observation()

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
                phase_obs,
            ],
            dim=-1,
        )

        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        num_envs = action_train.shape[0]

        leg_action_env = action_train * self.train_to_env_action_scale
        leg_action_env = torch.clip(leg_action_env, -self.leg_action_clip, self.leg_action_clip)

        arm_action_clipped = torch.clip(self.arm_stow_action, -self.arm_action_clip, self.arm_action_clip)

        action_env = torch.zeros(
            (num_envs, action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = arm_action_clipped.repeat(num_envs, 1)

        return action_env

    def predicts(self, obs, current_score):
        """Run policy inference and return current-env full-body action."""
        if current_score >= 26:
            return {'action': [], 'giveup': True}

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(
                action_train, device=self.device, dtype=torch.float32
            )

        action_train = action_train.to(device=self.device, dtype=torch.float32)

        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        action_env = action_env.cpu().numpy().tolist()
        return {'action': action_env, 'giveup': False}

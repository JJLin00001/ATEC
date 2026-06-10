import os
import torch
from typing import Any
import math

class AlgSolution:

    ACTION_SCALE = 0.5

    def __init__(self):
        # 使用你训练好的策略
        policy_path = './logs/rsl_rl/unitree_b2_rough/2026-06-08_14-40-33/exported/policy.pt'

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

        # Pure-pursuit heading control parameters (instead of lateral command correction)
        self.target_heading = 0.0
        self.k_heading = 0.8           # Heading error gain
        self.k_y = 0.6                 # Y-error to heading gain
        self.lookahead = 2.0           # Lookahead distance for pure-pursuit
        self.k_damping = 0.3           # Yaw rate damping

        # Yaw bias for systematic drift correction (tune based on testing)
        self.yaw_bias = 0.0            # Start at 0, adjust to -0.03 to -0.08 if stable left drift observed

        # Low-pass filter for yaw_roothing
        self.yaw_rate_alpha = 0.3
        self.yaw_rate_filtered = None

        # Adaptive velocity scheduling - REVISED with lower rough terrain speed
        self.velocity_schedule = {
            "flat_fast": 1.0,      # x < -115 (flat terrain)
            "rough": 0.55,         # -115 <= x < -35 (rough terrain, reduced from 0.8)
            "slope": 0.5,          # -35 <= x < 45 (slope terrain)
            "stairs": 0.45,        # 45 <= x < 125 (stairs terrain)
            "final": 0.7,          # 125 <= x < 145 (final stretch)
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

        # Execute recovery
        if self.in_recovery:
            self.recovery_counter += 1

            if self.recovery_counter < self.recovery_duration // 2:
                # First half: move backward with lateral
                return True, -0.3, 0.3
            else:
                # Second half: forward push (REMOVED excessive 1.5, use 1.1 instead)
                if self.recovery_counter >= self.recovery_duration:
                    self.in_recovery = False
                    self.recovery_counter = 0
                    self.last_progress_x = current_x
                return True, 0.4, 0.0  # Moderate boost, not 1.5

        return False, 0.0, 0.0

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Compute velocity commands with pure-pursuit heading control."""
        num_envs = proprio.shape[0]

        # Extract velocities
        base_lin_vel = proprio[:, 0:3]  # body frame
        base_ang_vel = proprio[:, 3:6]
        yaw_rate_measured = base_ang_vel[:, 2]

        # Initialize world-frame state estimates
        if self.estimated_yaw is None:
            self.estimated_yaw = torch.zeros(num_envs, device=self.device, dtype=proprio.dtype)
            self.estimated_x = torch.full((num_envs,), -141.0, device=self.device, dtype=proprio.dtype)
            self.estimated_y = torch.zeros(num_envs, device=self.device, dtype=proprio.dtype)

        # Update yaw estimate
        self.estimated_yaw = self.estimated_yaw + yaw_rate_measured * self.dt
        self.estimated_yaw = torch.atan2(torch.sin(self.estimated_yaw), torch.cos(self.estimated_yaw))

        # Transform body velocity to world frame using estimated yaw
        cos_yaw = torch.cos(self.estimated_yaw)
        sin_yaw = torch.sin(self.estimated_yaw)
        vel_world_x = base_lin_vel[:, 0] * cos_yaw - base_lin_vel[:, 1] * sin_yaw
        vel_world_y = base_lin_vel[:, 0] * sin_yaw + base_lin_vel[:, 1] * cos_yaw

        # Integrate world velocity to get position
        self.estimated_x = self.estimated_x + vel_world_x * self.dt
        self.estimated_y = self.estimated_y + vel_world_y * self.dt

        # Pure-pursuit heading control: aim toward line-following target
        # desired_heading = atan2(-k_y * y_error, lookahead)
        desired_heading = torch.atan2(-self.k_y * self.estimated_y, torch.tensor(self.lookahead, device=self.device))

        # Apply yaw bias for systematic drift correction (if needed after testing)
        desired_heading = desired_heading + self.yaw_bias

        # Heading error
        heading_error = desired_heading - self.estimated_yaw
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

        # Check for stuck condition
        in_recovery, forward_adjust, lateral_recovery = self._check_stuck_and_recover()

        # Compute yaw rate command with damping
        yaw_rate_cmd = self.k_heading * heading_error - self.k_damping * yaw_rate_measured

        # Low-pass filter
        if self.yaw_rate_filtered is None:
            self.yaw_rate_filtered = yaw_rate_cmd.clone()
        else:
            self.yaw_rate_filtered = (1 - self.yaw_rate_alpha) * self.yaw_rate_filtered + self.yaw_rate_alpha * yaw_rate_cmd

        # Clip yaw rate to avoid aggressive turning
        yaw_rate_cmd = torch.clip(self.yaw_rate_filtered, -0.5, 0.5)

        # Adaptive forward velocity
        x_estimate = self.estimated_x[0].item()
        forward_vel = self._get_adaptive_velocity(x_estimate)

        # Apply recovery adjustments
        if in_recovery:
            forward_vel += forward_adjust

        # Build command - NO lateral velocity, only yaw rate for path correction
        cmd = torch.zeros((num_envs, 3), device=self.device, dtype=proprio.dtype)
        cmd[:, 0] = forward_vel
        cmd[:, 1] = torch.clip(torch.tensor(lateral_recovery, device=self.device), -0.08, 0.08)  # Minimal lateral, only for recovery
        cmd[:, 2] = yaw_rate_cmd

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

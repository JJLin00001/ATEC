import os
import torch
from typing import Any

class AlgSolution:

    ACTION_SCALE = 0.5

    def __init__(self):
        # 使用你训练好的策略
        policy_path = './atec_robot_model/baseline/unitree_b2_flat/policy.pt'

        # 调试用 baseline：
        # policy_path = './atec_robot_model/baseline/unitree_b2_flat/policy.pt'

        self.device = 'cuda'
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        # Heading control parameters
        # Note: stiffness increased from training value (0.5) to 1.5 for stronger
        # drift correction since proprio integration accumulates error.
        self.target_heading = 0.0  # Task A: go straight forward (0 radians in world frame)
        self.heading_control_stiffness = 1.5
        # Lateral velocity feedback to counter sideways drift in body frame
        self.lateral_drift_gain = 0.5

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

        # Forward velocity command. Task A goes from x=-141 to x=145 (286m).
        # Higher forward velocity = faster traversal. 1.0 m/s is a balanced choice.
        # NOTE: yaw_rate (3rd element) is computed dynamically by heading PD controller.
        self.fixed_velocity_commands = torch.tensor(
            [1.0, 0.0, 0.0],  # [forward_vel, lateral_vel, yaw_rate]
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        # Yaw estimate (integrated from base_ang_vel.z), since proprio doesn't
        # provide world-frame heading directly. dt = 1/50 (control freq 50Hz).
        self.estimated_yaw = None  # initialized on first call
        self.dt = 0.02

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        """Optional action customization.

        Return None to use official default action config.

        Allowed groups:
            - leg
            - arm
            - wheel

        Allowed fields:
            - mode: "position", "velocity", or "effort"
            - scale: positive float
            - clip: None or [min, max]
        """
        return {}

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Compute velocity commands with closed-loop heading and lateral drift control.

        Training used heading_command=True with heading_control_stiffness=0.5.
        We replicate that PD law and additionally compensate for any residual
        lateral (body-frame y) velocity to keep the robot tracking a straight line.
        """
        num_envs = proprio.shape[0]

        # Extract base_lin_vel (indices 0:3) and base_ang_vel (indices 3:6)
        base_lin_vel = proprio[:, 0:3]
        base_ang_vel = proprio[:, 3:6]
        yaw_rate_measured = base_ang_vel[:, 2]
        lateral_vel_measured = base_lin_vel[:, 1]  # body-frame y velocity

        # Integrate yaw_rate to estimate world-frame heading
        if self.estimated_yaw is None:
            self.estimated_yaw = torch.zeros(num_envs, device=self.device, dtype=proprio.dtype)

        self.estimated_yaw = self.estimated_yaw + yaw_rate_measured * self.dt
        self.estimated_yaw = torch.atan2(torch.sin(self.estimated_yaw), torch.cos(self.estimated_yaw))

        # Heading PD: stronger stiffness than training (1.5 vs 0.5) to actively
        # correct integration drift over the 30+ second episode.
        heading_error = self.target_heading - self.estimated_yaw
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

        # Add lateral-drift compensation: if drifting +y, command negative yaw
        # rate to turn back toward the +x heading.
        yaw_rate_cmd = (
            self.heading_control_stiffness * heading_error
            - self.lateral_drift_gain * lateral_vel_measured
        )
        yaw_rate_cmd = torch.clip(yaw_rate_cmd, -1.0, 1.0)

        cmd = self.fixed_velocity_commands.to(dtype=proprio.dtype, device=self.device)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        cmd[:, 2] = yaw_rate_cmd

        return cmd

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

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Map training-time 12D leg action to current env 20D full-body action."""
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale

        action_env = torch.zeros(
            (num_envs, action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)

        return action_env

    def predicts(self, obs, current_score):
        """Run policy inference and return current-env full-body action."""
        # Task A total score is 26 (2+4+8+8+4). Don't give up until we've completed it.
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

import os
import torch
from typing import Any

class AlgSolution:

    ACTION_SCALE = 0.5

    def __init__(self):
        # Policy path: prioritize environment variable, then default to B2Piper rough policy
        # After training the B2Piper rough policy, it should be exported and placed here.
        # To override, set environment variable:
        # export ATEC_POLICY_PATH=/path/to/your/policy.pt
        policy_path = os.environ.get(
            'ATEC_POLICY_PATH',
            './logs/rsl_rl/unitree_b2_piper_rough/2026-06-09_19-12-47/exported/policy.pt'
        )

        # Fallback chain: try B2Piper, then old B2 rough, then baseline
        if not os.path.exists(policy_path):
            policy_path_b2_rough = './logs/rsl_rl/unitree_b2_rough/2026-06-08_20-47-08/exported/policy.pt'
            if os.path.exists(policy_path_b2_rough):
                policy_path = policy_path_b2_rough
                print(f"[INFO] B2Piper policy not found, using B2 rough policy: {policy_path}")
            else:
                policy_path = './atec_robot_model/baseline/unitree_b2_flat/policy.pt'
                print(f"[WARNING] No trained rough policy found, using baseline: {policy_path}")

        self.device = 'cuda'
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        self.policy_obs_dim = int(self.policy.actor._modules["0"].in_features)

        self.leg_action_dim = 12
        self.arm_action_dim = 8

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        # Heading control parameters - tuned for B2Piper deployment
        # The training uses heading_command with stiffness 0.5, but at deployment we
        # need stronger correction since yaw integration drifts over 30+ sec episodes.
        self.target_heading = 0.0  # Task A: go straight forward (0 radians in world frame)
        self.heading_control_stiffness = 0.5
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

        # Forward velocity command with gradual ramp-up for smooth start
        # Task A: straight line from x=-141 to x=145 (286m total distance)
        # B2Piper training uses 0.25-0.7 m/s, deploy at mid-range for stability
        self.target_forward_velocity = 0.5  # m/s
        self.fixed_velocity_commands = torch.tensor(
            [self.target_forward_velocity, 0.0, 0.0],  # [forward_vel, lateral_vel, yaw_rate]
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        # Velocity ramp parameters: gradually increase from 0 to target over first N steps
        # Gentle ramp avoids jerky startup on rough terrain
        self.ramp_steps = 50  # Ramp up over 1 second (50Hz control)
        self.current_step = 0

        # Yaw estimate (integrated from base_ang_vel.z), since proprio doesn't
        # provide world-frame heading directly. dt = 1/50 (control freq 50Hz).
        self.estimated_yaw = None  # initialized on first call
        self.dt = 0.02

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        # Gentle action rate limiting: B2Piper training already learned stable gait,
        # so deployment-side limiting should be minimal to avoid breaking the learned policy.
        self.prev_leg_action = None
        self.action_rate_limit = 0.3  # Max change per step in training-scale units (gentle)
        self.zero_action_steps = 50  # Hold zero action for first 50 steps (1 sec warmup)
        self.predict_step = 0

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

        Includes ramp-up: gradually increase forward velocity from 0 to target.
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

        # Velocity ramp: gradually increase forward velocity
        ramp_factor = min(1.0, self.current_step / self.ramp_steps)
        forward_vel_cmd = self.target_forward_velocity * ramp_factor

        cmd = torch.zeros((num_envs, 3), device=self.device, dtype=proprio.dtype)
        cmd[:, 0] = forward_vel_cmd
        cmd[:, 1] = 0.0
        cmd[:, 2] = yaw_rate_cmd

        self.current_step += 1

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

        policy_obs_terms = [
            base_ang_vel * 0.25,
            projected_gravity,
            velocity_commands,
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_train_leg,
        ]

        proprio_policy_dim = sum(term.shape[-1] for term in policy_obs_terms)
        extero_dim = self.policy_obs_dim - proprio_policy_dim
        if extero_dim > 0:
            if "extero" not in obs:
                policy_obs_terms.append(
                    torch.zeros(
                        (proprio.shape[0], extero_dim),
                        device=self.device,
                        dtype=proprio.dtype,
                    )
                )
            else:
                extero = obs["extero"].to(device=self.device, dtype=proprio.dtype)
                if extero.shape[-1] < extero_dim:
                    raise ValueError(
                        f"Policy expects {extero_dim} extero observations, but got {extero.shape[-1]}."
                    )
                if extero.shape[-1] == extero_dim:
                    policy_obs_terms.append(extero)
                else:
                    sample_ids = torch.linspace(
                        0,
                        extero.shape[-1] - 1,
                        extero_dim,
                        device=self.device,
                    ).long()
                    policy_obs_terms.append(extero.index_select(-1, sample_ids))

        policy_obs = torch.cat(policy_obs_terms, dim=-1)
        if policy_obs.shape[-1] != self.policy_obs_dim:
            raise ValueError(
                f"Policy observation dim mismatch: got {policy_obs.shape[-1]}, expected {self.policy_obs_dim}"
            )

        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Map training-time 12D leg action to current env 20D full-body action.

        Applies gentle action rate limiting for warmup smoothness. The trained B2Piper
        policy already learned stable diagonal trot; deployment-side constraints should
        be minimal to preserve the learned gait.
        """
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        num_envs = action_train.shape[0]

        # Gentle action rate limiting: only prevents startup jerk, doesn't constrain learned gait
        if self.prev_leg_action is not None:
            action_delta = action_train - self.prev_leg_action
            action_delta = torch.clamp(action_delta, -self.action_rate_limit, self.action_rate_limit)
            action_train = self.prev_leg_action + action_delta

        self.prev_leg_action = action_train.clone()

        # Gentle clamp: trust the policy's learned range, only clip extreme outliers
        action_train = torch.clamp(action_train, -5.0, 5.0)

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

        if self.predict_step < self.zero_action_steps:
            num_envs = proprio.shape[0]
            self.prev_leg_action = torch.zeros(
                (num_envs, self.leg_action_dim),
                device=self.device,
                dtype=torch.float32,
            )
            self.predict_step += 1
            action_env = torch.zeros(
                (num_envs, action_dim),
                device=self.device,
                dtype=torch.float32,
            )
            return {'action': action_env.cpu().numpy().tolist(), 'giveup': False}

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
        self.predict_step += 1
        action_env = action_env.cpu().numpy().tolist()
        return {'action': action_env, 'giveup': False}

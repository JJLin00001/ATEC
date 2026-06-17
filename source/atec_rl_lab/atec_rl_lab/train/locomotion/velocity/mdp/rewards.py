# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame.

    Uses exponential kernel for reward computation.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    defined in :attr:`synced_feet_pair_names` to bias the policy towards a desired gait,
    i.e trotting, bounding, or pacing. Note that this reward is only for quadrupedal gaits
    with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(
#         env.action_manager.action - 2 * env.action_manager.prev_action
#         + env.action_manager.prev_prev_action
#     )
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# ==============================================================================
# NEW TROT GAIT REWARDS - Added for solving RL_foot long air time issue
# ==============================================================================


def current_contact_count_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    expect_contact_num: int,
    force_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize when current number of feet in contact != expect_contact_num.

    Unlike feet_contact which uses compute_first_contact (only triggers on contact events),
    this reward evaluates the CURRENT instantaneous contact state based on force magnitude.
    This provides continuous feedback to maintain the desired contact pattern at every timestep.

    Args:
        expect_contact_num: Expected number of feet in contact (2 for trot gait)
        force_threshold: Minimum force (N) to consider a foot in contact (default 1.0)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Compute current contact based on force magnitude (NOT first_contact events)
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    force_magnitude = torch.linalg.norm(net_forces, dim=2)  # (num_envs, num_feet)
    is_contact = force_magnitude > force_threshold  # (num_envs, num_feet)

    # Count current number of feet in contact
    current_contact_count = torch.sum(is_contact.float(), dim=1)  # (num_envs,)

    # Penalize deviation from expected contact count
    penalty = torch.abs(current_contact_count - expect_contact_num)

    # Only apply when moving
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return penalty


def long_air_time_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    air_time_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize any foot that has current_air_time > threshold.

    This directly targets the RL_foot long air time problem by penalizing EACH foot
    individually when it stays airborne too long. The penalty scales with how much
    the air time exceeds the threshold.

    Args:
        air_time_threshold: Maximum allowed continuous air time (e.g., 0.35s)
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get current air time for each foot
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # (num_envs, num_feet)

    # Penalize when air time exceeds threshold, with linear scaling
    excess_air_time = torch.clamp(current_air_time - air_time_threshold, min=0.0)
    penalty = torch.sum(excess_air_time, dim=1)  # Sum across all feet

    # Only apply when moving
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return penalty


def long_contact_time_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_time_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet that stay in contact for too long while moving.

    This targets the common rough-terrain failure mode where one foot remains
    pinned on an obstacle while the other feet keep cycling or stay airborne.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    excess_contact_time = torch.clamp(current_contact_time - contact_time_threshold, min=0.0)
    penalty = torch.sum(excess_contact_time, dim=1)

    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return penalty


def swing_foot_clearance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_height: float,
    force_threshold: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize swing feet that remain too low in the base frame.

    Feet closer to the body have less chance of catching rough terrain edges.
    The term is applied only to feet currently considered airborne.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_magnitude = torch.linalg.norm(net_forces, dim=2)
    in_air = force_magnitude <= force_threshold

    foot_pos_rel_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    num_feet = len(asset_cfg.body_ids)
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    foot_pos_b = math_utils.quat_apply_inverse(root_quat, foot_pos_rel_w.reshape(-1, 3)).reshape(
        env.num_envs, num_feet, 3
    )

    low_clearance = torch.clamp(min_height - foot_pos_b[:, :, 2], min=0.0)
    penalty = torch.sum(low_clearance * in_air.float(), dim=1)

    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return penalty


def diagonal_trot_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    force_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward diagonal trot contact pattern (FR+RL or FL+RR), penalize others.

    Trot gait contact patterns:
    - GOOD: FR+RL (diagonal), FL+RR (diagonal), or 1 foot (transition)
    - BAD: FR+FL (same side), RR+RL (same side), 3 feet, 4 feet

    This reward directly shapes the desired gait pattern at the contact level.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Compute current contact state
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_magnitude = torch.linalg.norm(net_forces, dim=2)
    is_contact = force_magnitude > force_threshold  # (num_envs, 4) for [FR, FL, RR, RL]

    # Extract individual foot contact states (assuming order FR, FL, RR, RL)
    FR_contact = is_contact[:, 0]
    FL_contact = is_contact[:, 1]
    RR_contact = is_contact[:, 2]
    RL_contact = is_contact[:, 3]

    # Diagonal pairs
    diagonal_1 = FR_contact & RL_contact  # FR + RL
    diagonal_2 = FL_contact & RR_contact  # FL + RR

    # Same-side pairs (BAD)
    same_side_front = FR_contact & FL_contact  # Front pair
    same_side_rear = RR_contact & RL_contact   # Rear pair

    # Count total contacts
    contact_count = torch.sum(is_contact.float(), dim=1)

    # Reward structure:
    # +1.0 for diagonal pairs (good trot)
    # -0.5 for same-side pairs (bad pattern)
    # -0.3 for 3 or 4 feet contact (bad pattern)
    # 0.0 for single foot (neutral, transition phase)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward += diagonal_1.float() * 1.0
    reward += diagonal_2.float() * 1.0
    reward -= same_side_front.float() * 0.5
    reward -= same_side_rear.float() * 0.5
    reward -= ((contact_count == 3) | (contact_count == 4)).float() * 0.3

    # Only apply when moving
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return reward


def phase_trot_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    cycle_time: float,
    force_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward phase-synchronized trot: first half expects FR+RL, second half expects FL+RR.

    FIXED: Symmetric logic - each phase rewards its expected pair and penalizes wrong pairs.
    - Phase [0, 0.5): expect FR+RL diagonal, penalize FR/RL missing, penalize FL/RR over-contact
    - Phase [0.5, 1.0): expect FL+RR diagonal, penalize FL/RR missing, penalize FR/RL over-contact

    No single foot gets special treatment across both half-cycles.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Compute current gait phase
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = (env.episode_length_buf * env.step_dt / cycle_time) % 1.0  # Range [0, 1)

    # Compute current contact state
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_magnitude = torch.linalg.norm(net_forces, dim=2)
    is_contact = force_magnitude > force_threshold  # (num_envs, 4) body_ids order specified in cfg

    # Extract foot contact (order matches sensor_cfg.body_names)
    FR_contact = is_contact[:, 0]
    FL_contact = is_contact[:, 1]
    RR_contact = is_contact[:, 2]
    RL_contact = is_contact[:, 3]

    # Expected contact patterns by phase
    first_half = phase < 0.5  # Expect FR+RL
    second_half = phase >= 0.5  # Expect FL+RR

    # Reward for matching expected pattern
    reward = torch.zeros(env.num_envs, device=env.device)

    # First half: reward FR+RL diagonal match
    diagonal_1_match = (FR_contact & RL_contact).float()
    reward += torch.where(first_half, diagonal_1_match * 1.0, torch.zeros_like(reward))

    # Penalize missing FR or RL in first half
    reward -= torch.where(first_half & ~FR_contact, torch.ones_like(reward) * 0.8, torch.zeros_like(reward))
    reward -= torch.where(first_half & ~RL_contact, torch.ones_like(reward) * 0.8, torch.zeros_like(reward))

    # Penalize FL/RR over-contact in first half (they should be in air)
    reward -= torch.where(first_half & FL_contact & RR_contact, torch.ones_like(reward) * 0.5, torch.zeros_like(reward))

    # Second half: reward FL+RR diagonal match
    diagonal_2_match = (FL_contact & RR_contact).float()
    reward += torch.where(second_half, diagonal_2_match * 1.0, torch.zeros_like(reward))

    # Penalize missing FL or RR in second half
    reward -= torch.where(second_half & ~FL_contact, torch.ones_like(reward) * 0.8, torch.zeros_like(reward))
    reward -= torch.where(second_half & ~RR_contact, torch.ones_like(reward) * 0.8, torch.zeros_like(reward))

    # Penalize FR/RL over-contact in second half (they should be in air)
    reward -= torch.where(second_half & FR_contact & RL_contact, torch.ones_like(reward) * 0.5, torch.zeros_like(reward))

    # Only apply when moving
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return reward


def signed_trot_joint_mirror(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint position asymmetry for diagonal trot pairs with signed mirror.

    For quadruped trot, diagonal legs (FR-RL, FL-RR) should mirror each other:
    - Hip joints: OPPOSITE signs (left hip > 0, right hip < 0)
    - Thigh/Calf joints: SAME signs (both flex/extend together)

    Joint order assumed: FR[hip,thigh,calf], FL[hip,thigh,calf], RR[hip,thigh,calf], RL[hip,thigh,calf]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (num_envs, 12)

    # Extract diagonal pairs
    FR = joint_pos[:, 0:3]  # [hip, thigh, calf]
    FL = joint_pos[:, 3:6]
    RR = joint_pos[:, 6:9]
    RL = joint_pos[:, 9:12]

    # Diagonal pair 1: FR - RL
    # Hip: opposite signs (negate FR hip for comparison)
    hip_diff_1 = torch.abs(-FR[:, 0] - RL[:, 0])
    # Thigh/Calf: same signs
    thigh_diff_1 = torch.abs(FR[:, 1] - RL[:, 1])
    calf_diff_1 = torch.abs(FR[:, 2] - RL[:, 2])

    # Diagonal pair 2: FL - RR
    hip_diff_2 = torch.abs(-FL[:, 0] - RR[:, 0])
    thigh_diff_2 = torch.abs(FL[:, 1] - RR[:, 1])
    calf_diff_2 = torch.abs(FL[:, 2] - RR[:, 2])

    # Total penalty
    penalty = hip_diff_1 + thigh_diff_1 + calf_diff_1 + hip_diff_2 + thigh_diff_2 + calf_diff_2

    # Only apply when moving
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return penalty


def signed_trot_action_mirror(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize action asymmetry for diagonal trot pairs with signed mirror.

    Same as signed_trot_joint_mirror but applied to actions instead of joint positions.
    This encourages symmetric control commands to diagonal leg pairs.
    """
    actions_all = env.action_manager.action
    joint_ids = asset_cfg.joint_ids

    # The action tensor is ordered by the action term, not by articulation joint ids.
    # On B2 these ids happen to match. On B2Piper the robot has passive arm joints,
    # so leg articulation ids can exceed the 12-D leg action tensor.
    if isinstance(joint_ids, slice):
        actions = actions_all[:, joint_ids]
    elif actions_all.shape[1] == len(joint_ids):
        actions = actions_all
    else:
        actions = actions_all[:, joint_ids]

    # Extract diagonal pairs
    FR_act = actions[:, 0:3]
    FL_act = actions[:, 3:6]
    RR_act = actions[:, 6:9]
    RL_act = actions[:, 9:12]

    # Diagonal pair 1: FR - RL (signed mirror)
    hip_diff_1 = torch.abs(-FR_act[:, 0] - RL_act[:, 0])
    thigh_diff_1 = torch.abs(FR_act[:, 1] - RL_act[:, 1])
    calf_diff_1 = torch.abs(FR_act[:, 2] - RL_act[:, 2])

    # Diagonal pair 2: FL - RR (signed mirror)
    hip_diff_2 = torch.abs(-FL_act[:, 0] - RR_act[:, 0])
    thigh_diff_2 = torch.abs(FL_act[:, 1] - RR_act[:, 1])
    calf_diff_2 = torch.abs(FL_act[:, 2] - RR_act[:, 2])

    # Total penalty
    penalty = hip_diff_1 + thigh_diff_1 + calf_diff_1 + hip_diff_2 + thigh_diff_2 + calf_diff_2

    # Only apply when moving
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return penalty


def diagonal_pair_duty_balance_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize duty cycle imbalance between diagonal pairs (FR+RL vs FL+RR).

    Uses current_air_time and current_contact_time as stateless approximations
    to measure whether the two diagonal pairs have balanced duty cycles.

    This prevents one diagonal pair from dominating (high duty) while the other
    barely touches the ground (low duty), which would cause asymmetric gait.

    Expected behavior: Both pairs should have similar average air/contact times
    for a balanced trot gait (~50% duty each).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get current air and contact times (order matches sensor_cfg.body_names)
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]  # (num_envs, 4)
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    # Extract times for each foot (order: FR, FL, RR, RL)
    FR_air = current_air_time[:, 0]
    FL_air = current_air_time[:, 1]
    RR_air = current_air_time[:, 2]
    RL_air = current_air_time[:, 3]

    FR_contact = current_contact_time[:, 0]
    FL_contact = current_contact_time[:, 1]
    RR_contact = current_contact_time[:, 2]
    RL_contact = current_contact_time[:, 3]

    # Diagonal pair averages
    pair1_air = (FR_air + RL_air) / 2.0    # FR+RL diagonal
    pair2_air = (FL_air + RR_air) / 2.0    # FL+RR diagonal

    pair1_contact = (FR_contact + RL_contact) / 2.0
    pair2_contact = (FL_contact + RR_contact) / 2.0

    # Penalize imbalance in both air and contact times
    air_imbalance = torch.abs(pair1_air - pair2_air)
    contact_imbalance = torch.abs(pair1_contact - pair2_contact)

    penalty = air_imbalance + contact_imbalance

    # Only apply when moving
    penalty *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return penalty

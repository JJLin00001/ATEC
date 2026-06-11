# Configuration for Unitree B2 Piper (with arms) on rough terrain

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG


@configclass
class UnitreeB2PiperRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for B2 Piper robot on rough terrain.

    This uses the full B2Piper model (with arms) but only controls the 12 leg joints.
    Arms are fixed in stow configuration via init_state.

    This ensures training dynamics match deployment on TaskA-B2Piper.
    """
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    # fmt: off
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    arm_stow_joint_pos = {
        "arm_joint1": 0.0,
        "arm_joint2": 1.2,
        "arm_joint3": 0.0,
        "arm_joint4": -1.5,
        "arm_joint5": 0.0,
        "arm_joint6": 0.0,
        "arm_joint7": 0.0,
        "arm_joint8": 0.0,
    }
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # Train with the real B2Piper body while keeping the arm stowed.
        self.scene.robot.init_state.joint_pos.update(self.arm_stow_joint_pos)

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None  # Blind policy (proprioceptive only)
        # Only observe leg joints, not arms
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.leg_joint_names

        # ------------------------------Actions------------------------------
        # Only control leg joints (12 DOF)
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)

        # ------------------------------Rewards------------------------------
        # Copy all reward weights from UnitreeB2RoughEnvCfg
        # General
        self.rewards.is_terminated.weight = -2.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.base_height_l2.weight = -0.5
        self.rewards.base_height_l2.params["target_height"] = 0.53
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -1e-5
        self.rewards.stand_still.weight = -1.0
        self.rewards.joint_pos_penalty.weight = -0.5
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        for reward_term in (
            self.rewards.joint_torques_l2,
            self.rewards.joint_vel_l2,
            self.rewards.joint_acc_l2,
            self.rewards.joint_pos_limits,
            self.rewards.joint_vel_limits,
            self.rewards.joint_power,
            self.rewards.stand_still,
            self.rewards.joint_pos_penalty,
        ):
            reward_term.params["asset_cfg"].joint_names = self.leg_joint_names

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards - CRITICAL for forward motion
        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # Gait rewards - balanced weights
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.rewards.feet_air_time_variance.params["sensor_cfg"].preserve_order = True

        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.params["expect_contact_num"] = 2

        # NEW TROT GAIT REWARDS
        self.rewards.current_contact_count_penalty.weight = -1.0
        self.rewards.current_contact_count_penalty.params["sensor_cfg"].body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.rewards.current_contact_count_penalty.params["sensor_cfg"].preserve_order = True

        self.rewards.long_air_time_penalty.weight = -1.5
        self.rewards.long_air_time_penalty.params["sensor_cfg"].body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.rewards.long_air_time_penalty.params["sensor_cfg"].preserve_order = True

        self.rewards.diagonal_trot_contact_reward.weight = 2.5
        self.rewards.diagonal_trot_contact_reward.params["sensor_cfg"].body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.rewards.diagonal_trot_contact_reward.params["sensor_cfg"].preserve_order = True

        self.rewards.phase_trot_contact_reward.weight = 1.5
        self.rewards.phase_trot_contact_reward.params["sensor_cfg"].body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.rewards.phase_trot_contact_reward.params["sensor_cfg"].preserve_order = True

        self.rewards.signed_trot_joint_mirror.weight = -0.2
        self.rewards.signed_trot_joint_mirror.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.signed_trot_action_mirror.weight = -0.1
        self.rewards.signed_trot_action_mirror.params["asset_cfg"].joint_names = self.leg_joint_names

        self.rewards.diagonal_pair_duty_balance_penalty.weight = -2.0
        self.rewards.diagonal_pair_duty_balance_penalty.params["sensor_cfg"].body_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.rewards.diagonal_pair_duty_balance_penalty.params["sensor_cfg"].preserve_order = True

        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.1
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]

        self.rewards.feet_gait.weight = 3.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 4.0

        # Disable zero-weight rewards
        if self.__class__.__name__ == "UnitreeB2PiperRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip", ".*_thigh"]

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

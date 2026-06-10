# Reference: https://github.com/fan-ziqi/robot_lab

from isaaclab.utils import configclass


from atec_rl_lab.train.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from atec_rl_lab.assets.robots import UNITREE_B2_CFG, UNITREE_B2_PIPER_CFG


@configclass
class UnitreeB2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_B2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        # Policy observation: 45-dim proprio-only (no height_scan)
        # Actor sees: base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
        #             + joint_pos(12) + joint_vel(12) + actions(12) = 45 dim
        # Critic can optionally keep height_scan for asymmetric training
        self.observations.policy.base_lin_vel = None  # Already disabled
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.height_scan = None  # CRITICAL: disable for 45-dim actor
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # Critic keeps height_scan for asymmetric advantage (can be disabled if causing issues)
        # If RayCaster hangs, set self.observations.critic.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.05),  # Reduced z variation to start closer to ground
                "roll": (-0.1, 0.1),  # Much smaller roll range for stable initialization
                "pitch": (-0.1, 0.1),  # Much smaller pitch range for stable initialization
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),  # Reduced initial velocities
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
        # Tuned for stable diagonal trot on rough terrain, suppressing bounding gait

        # General
        self.rewards.is_terminated.weight = -2.0  # Penalize early termination

        # Root penalties - increased for stability
        self.rewards.lin_vel_z_l2.weight = -3.0  # Higher penalty for vertical oscillation
        self.rewards.ang_vel_xy_l2.weight = -0.3  # Doubled to reduce roll/pitch oscillation
        self.rewards.flat_orientation_l2.weight = -3.0  # Stronger flat body orientation
        self.rewards.base_height_l2.weight = -1.5  # Encourage maintaining target height
        self.rewards.base_height_l2.params["target_height"] = 0.53
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = -0.5e-3  # Penalize sudden acceleration
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2e-5  # Doubled to encourage efficiency
        self.rewards.joint_vel_l2.weight = -1e-4  # Small penalty on joint velocity
        self.rewards.joint_acc_l2.weight = -2.5e-7  # Increased to smooth motion
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5  # Doubled to penalize excessive power
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties - CRITICAL for suppressing jerky/bounding motion
        self.rewards.action_rate_l2.weight = -0.03  # Tripled: strongly penalize rapid action changes

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -2e-4  # Slightly higher to reduce impact forces
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Gait rewards - CRITICAL for diagonal trot
        self.rewards.feet_air_time.weight = 0.0  # DISABLED: was encouraging bounding
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.2  # Doubled to avoid stumbling on rough terrain
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -1.0  # Doubled: critical for traction on rough terrain
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 3.5  # INCREASED: strongly reward diagonal trot pattern
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 3.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # Terminate when robot's body (non-foot parts) hit the ground
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip", ".*_thigh"]

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        # Conservative velocity commands for initial training on rough terrain
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.8)  # Conservative forward range
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)  # Limited lateral
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)  # Limited turning


@configclass
class UnitreeB2PiperRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """B2 + Piper arm rough terrain locomotion env.

    Trains the same 12-DoF leg policy as UnitreeB2RoughEnvCfg, but on the B2Piper body
    so that base inertia, contact distribution, and CoM match the TaskA evaluation robot.
    The 8 arm joints are NOT controlled by the policy; they are held at a fixed pose by
    the implicit actuator PD targets (set via ``init_state.joint_pos`` defaults).

    Actor observation stays 45-dim proprio-only (12 leg joints), so a checkpoint trained
    on plain B2 rough can be directly used to warm-start training here.
    """

    # Default arm joint targets (radians). The Piper arm has 8 joints
    # (arm_joint1..arm_joint8). 0 rad keeps the arm in its USD-defined neutral pose;
    # if a more compact tucked pose is preferred, tune these here without changing the
    # locomotion policy interface.
    arm_joint_default_pos = {
        "arm_joint1": 0.0,
        "arm_joint2": 0.0,
        "arm_joint3": 0.0,
        "arm_joint4": 0.0,
        "arm_joint5": 0.0,
        "arm_joint6": 0.0,
        "arm_joint7": 0.0,
        "arm_joint8": 0.0,
    }

    def __post_init__(self):
        # post init of parent (sets up B2 rough rewards/observations/etc.)
        super().__post_init__()

        # ------------------------------Robot------------------------------
        # Replace plain B2 with B2 + Piper arm. Keep init pose stable on the ground.
        b2_piper_cfg = UNITREE_B2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Merge leg default joint angles (inherited from B2) with arm hold pose.
        # The arm joints will be held at these targets via ImplicitActuator PD because
        # the policy never writes to arm_joint.* DOFs (action only covers leg joints).
        merged_joint_pos = dict(b2_piper_cfg.init_state.joint_pos)
        merged_joint_pos.update(self.arm_joint_default_pos)
        b2_piper_cfg = b2_piper_cfg.replace(
            init_state=b2_piper_cfg.init_state.replace(joint_pos=merged_joint_pos),
        )
        self.scene.robot = b2_piper_cfg

        # ------------------------------Actions------------------------------
        # Policy still controls only the 12 leg joints; arm DOFs are held by PD.
        # Re-apply the leg-only joint_names (parent set this from B2 config but the
        # B2Piper articulation has 20 DOFs total — make sure the action manager
        # only targets the 12 legs).
        self.actions.joint_pos.joint_names = list(UNITREE_B2_PIPER_CFG.leg_joint_names)

        # ------------------------------Observations------------------------------
        # Actor observation must stay 45-dim proprio-only so we can warm-start from
        # the existing unitree_b2_rough checkpoint. Restrict joint_pos / joint_vel
        # observations to the 12 leg joints (the parent sets this to B2 leg names,
        # which happen to match B2Piper leg joints, but be explicit here).
        leg_joint_names = list(UNITREE_B2_PIPER_CFG.leg_joint_names)
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = leg_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = leg_joint_names
        # CRITICAL: Also restrict critic observations to leg joints only for checkpoint compatibility
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = leg_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = leg_joint_names
        # height_scan stays None (already disabled in parent for 45-dim actor)

        # ------------------------------Events------------------------------
        # Some randomization terms in the parent reference body name patterns from
        # plain B2; rebind them to the B2Piper body names where needed. Most regex
        # patterns (e.g. ".*_hip", ".*_thigh") still match correctly on B2Piper.

        # ------------------------------Rewards: enable straight-line + diagonal balance--
        # B2Piper-specific tweaks. All defaults stay weight=0 in the base RewardsCfg,
        # so we only enable here.

        # Suppress lateral drift and uncommanded yaw to walk straight.
        self.rewards.lin_vel_y_l2.weight = -0.5
        self.rewards.yaw_rate_l2.weight = -0.2

        # Penalize asymmetry between FL+RR and FR+RL diagonal pairs (air time / contact time).
        self.rewards.diagonal_pair_air_time_balance.weight = -0.5
        self.rewards.diagonal_pair_air_time_balance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.diagonal_pair_air_time_balance.params["synced_feet_pair_names"] = (
            ("FL_foot", "RR_foot"),
            ("FR_foot", "RL_foot"),
        )

        # Keep diagonal trot strongly enforced (slightly softer than parent to leave
        # room for the new balance reward to take effect).
        self.rewards.feet_gait.weight = 3.0

        # ------------------------------Commands------------------------------
        # TaskA-style: forward-only, no lateral/yaw command. Heading is corrected at
        # deployment via solution.py PD on yaw integration.
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Re-apply zero-weight reward pruning for this leaf class
        if self.__class__.__name__ == "UnitreeB2PiperRoughEnvCfg":
            self.disable_zero_weight_rewards()

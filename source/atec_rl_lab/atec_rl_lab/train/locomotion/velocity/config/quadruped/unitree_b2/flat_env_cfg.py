# Reference: https://github.com/fan-ziqi/robot_lab

from isaaclab.utils import configclass

from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.rough_env_cfg import UnitreeB2RoughEnvCfg


@configclass
class UnitreeB2FlatEnvCfg(UnitreeB2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = f"{ATEC_ASSETS_MODEL_DIR}/scene/plane/default_environment.usd"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.commands.base_velocity.debug_vis = False
        self.terminations.terrain_out_of_bounds = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2FlatEnvCfg":
            self.disable_zero_weight_rewards()

# PPO runner configuration for Unitree B2 Piper rough-terrain locomotion.

from isaaclab.utils import configclass

from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg import (
    UnitreeB2RoughPPORunnerCfg,
)


@configclass
class UnitreeB2PiperRoughPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    """Use the same PPO hyperparameters as B2, but keep B2Piper logs separate."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_piper_rough"


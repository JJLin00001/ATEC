# Configuration for Unitree B2 Piper robot

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="ATEC-Isaac-Velocity-Rough-Unitree-B2Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2PiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2.agents.rsl_rl_ppo_cfg:UnitreeB2RoughPPORunnerCfg"
    },
)

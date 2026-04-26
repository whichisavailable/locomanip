# Copyright (c) 2024-2025 Ziqi Fan
"""
Module for registering Gym environments for the Unitree Go2 Arm robot locomotion tasks.

This module registers two Gymnasium environments for velocity-based locomotion control
of the Unitree Go2 Arm robot with different terrain configurations:

1. RobotLab-Isaac-Velocity-Flat-Go2Arm-v0: Environment for flat terrain locomotion
2. RobotLab-Isaac-Velocity-Rough-Go2Arm-v0: Environment for rough terrain locomotion

Each environment is configured with:
- entry_point: The class path to the environment implementation (ManagerBasedRLEnv)
- env_cfg_entry_point: The configuration class for environment parameters
- rsl_rl_cfg_entry_point: The configuration class for RSL-RL PPO training algorithm

Note:
    entry_point is a string that specifies the full path to the environment class
    that Gymnasium will instantiate when creating the environment. It follows the
    format "module.path:ClassName" and is used by gym.register() to dynamically
    import and initialize the environment.
"""
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 注册文件
# Gym 内部创建环境时相当于：
# env = ManagerBasedRLEnv(
#     env_cfg_entry_point="your_module.flat_env_cfg:UnitreeGo2ArmFlatEnvCfg",
#     rsl_rl_cfg_entry_point="agents.rsl_rl_ppo_cfg:UnitreeGo2ArmFlatPPORunnerCfg"
# )

gym.register(
    id="RobotLab-Isaac-Flat-Go2Arm-v0",
    entry_point=f"{__name__}.env:Go2ArmManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # UnitreeGo2ArmFlatEnvCfg这是一个环境配置类
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2ArmFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArmFlatPPORunnerCfg",
        # "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2ArmFlatTrainerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Rough-Go2Arm-v0",
    entry_point=f"{__name__}.env:Go2ArmManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2ArmRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ArmTeacherRoughPPORunnerCfg",
        # "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2ArmRoughTrainerCfg",
    },
)

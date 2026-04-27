# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.cus_velocity_env_cfg import (
    GO2ARM_BASE_BODY_NAME,
    GO2ARM_FOOT_BODY_NAMES,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import observations as mdp_obs

from .rough_env_cfg import UnitreeGo2ArmRoughEnvCfg


@configclass
class UnitreeGo2ArmFlatEnvCfg(UnitreeGo2ArmRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        terrain_contact_filter_prim_paths = self._terrain_contact_filter_prim_paths()
        self.scene.FL_foot_contact.filter_prim_paths_expr = terrain_contact_filter_prim_paths
        self.scene.FR_foot_contact.filter_prim_paths_expr = terrain_contact_filter_prim_paths
        self.scene.RL_foot_contact.filter_prim_paths_expr = terrain_contact_filter_prim_paths
        self.scene.RR_foot_contact.filter_prim_paths_expr = terrain_contact_filter_prim_paths
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # On a flat plane these quantities have exact analytic forms, so the ray scanners are unnecessary.
        self.observations.privileged.base_height.func = mdp_obs.base_height_on_plane
        self.observations.privileged.base_height.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME])
        }
        self.observations.privileged.foot_heights.func = mdp_obs.foot_heights_on_plane
        self.observations.privileged.foot_heights.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES)
        }
        self.rewards.total_reward.params["mani_regularization_min_base_height_sensor_cfg"] = None
        self.rewards.total_reward.params["loco_regularization_base_height_sensor_cfg"] = None
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_ground_sensor_names"] = None
        self.terminations.base_height_termination.params["sensor_cfg"] = None

        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.FL_foot_scanner = None
        self.scene.FR_foot_scanner = None
        self.scene.RL_foot_scanner = None
        self.scene.RR_foot_scanner = None

    # ''' # If the weight of rewards is 0, set rewards to None
    # if self.__class__.__name__ == "UnitreeGo2ArmFlatEnvCfg":
    #     self.disable_zero_weight_rewards()'''

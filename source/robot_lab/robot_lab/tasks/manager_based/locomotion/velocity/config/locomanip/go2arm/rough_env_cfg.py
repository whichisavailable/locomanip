# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.assets.unitree import UNITREE_Go2Arm_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.cus_velocity_env_cfg import (
    GO2ARM_BASE_BODY_NAME,
    GO2ARM_NON_FOOT_BODY_REGEX,
    GO2ARM_TROT_PHASE_OFFSETS,
    Go2ArmTeacherActionsCfg,
    Go2ArmTeacherCommandsCfg,
    Go2ArmTeacherCoreObsCfg,
    Go2ArmTeacherCriticExtraObsCfg,
    Go2ArmTeacherPrivilegedObsCfg,
    LocomotionVelocityRoughEnvCfg,
)


@configclass
class UnitreeGo2ArmRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = GO2ARM_BASE_BODY_NAME
    foot_link_name: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    rsl_rl_init_noise_std: float | tuple[float, ...] = (
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    )
    foot_scanner_prim_path: list[str] = [
        "FL_foot",
        "FR_foot",
        "RL_foot",
        "RR_foot",
    ]
    reward_log_interval_iterations: int = 1
    reward_log_steps_per_iteration: int = 24
    enable_debug_reward_logging: bool = True
    enable_collision_group_logging: bool = False
    enable_contact_verification_logging: bool = False
    enable_termination_debug_logging: bool = False
    episode_log_key_prefixes: tuple[str, ...] = (
        "R/",
        "Len/",
        "Term/",
    )
    # 是否启用精确足端接触实现；关闭后退回calf近似，便于做对照实验。
    # 零动作站立验证开关。
    # 当前默认关闭，恢复正常策略动作输入；后续如需再次验证 default pose，可临时改成 True。
    debug_zero_action: bool = False

    def _terrain_contact_filter_prim_paths(self) -> list[str]:
        """根据当前地形类型返回足端 filtered contact sensor 需要过滤到的地形 prim。"""
        if self.scene.terrain.terrain_type == "plane":
            return ["/World/ground/terrain/GroundPlane/CollisionPlane"]
        return ["/World/ground/terrain/mesh"]

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16384

        # Teacher observation layout.
        self.observations.policy = Go2ArmTeacherCoreObsCfg()
        self.observations.critic = None
        self.observations.privileged = Go2ArmTeacherPrivilegedObsCfg()
        self.observations.critic_extra = Go2ArmTeacherCriticExtraObsCfg()

        # Confirmed 18D absolute joint-position action target.
        self.actions = Go2ArmTeacherActionsCfg()
        self.actions.joint_pos.delta_clip = {
            "^(FL|FR|RL|RR)_hip_joint$": (-0.62832, 0.62832),
            "^(FL|FR)_thigh_joint$": (-1.42248, 1.61442),
            "^(RL|RR)_thigh_joint$": (-0.79416, 2.24274),
            "^(FL|FR|RL|RR)_calf_joint$": (-0.73362, 0.39734),
            # 机械臂 6 个关节按 URDF 原始 position limits，
            # 先乘 asset 里的 soft_joint_pos_limit_factor=0.9 得到软限位，
            # 再从默认姿态出发只开放到软限位剩余空间的 60%，
            # 这样既保留足够的 reaching 工作空间，也避免联训初期大动作把底盘拉翻。
            "^joint1$": (-2.0944, 2.0944),
            "^joint2$": (0.0, 2.512),
            "^joint3$": (-2.3736, 0.0),
            "^joint4$": (-1.396, 1.396),
            "^joint5$": (-0.976, 0.976),
            "^joint6$": (-1.67552, 1.67552),
        }

        # Only keep the ee_pose command.
        self.commands = Go2ArmTeacherCommandsCfg()
        # 固定点 reaching 阶段进一步减弱累计误差的累积速度，让它更多作为后期细化驱动而不是前期主惩罚。
        self.commands.ee_pose.cumulative_error_weight = 0.01
        self.commands.ee_pose.reference_error_velocity = 0.5
        self.commands.ee_pose.orientation_error_weight = 0.1

        # go2arm 三阶段课程：
        # 1) 近端安全操纵：围绕默认末端工作点学习站稳与摆臂；
        # 2) z 轴高难命令：先用较低比例引入前下方 / 前上方极值任务，
        #    再同步扩 z 上下极限，最后在极限固定后继续增大高难样本比例；
        # 3) 全局扩展：xy 范围逐步放大，同时 z 在最大范围内均匀采样。
        #
        # 默认 ee pose 由 go2arm URDF + 默认 joint_pos 推得：
        #   ee_pos_b ≈ (0.1695, 0.0, 0.4911)
        #   ee_euler_b ≈ (0.0, 1.4836, 0.0)
        # stage1 围绕默认工作点给一个近端安全盒。
        # 这里改成混合采样：
        #   - xy 在 base frame 下采样，保证“前/后/左/右”的语义始终跟随机身；
        #   - z 在 world frame 下采样，便于直接围绕地面高度组织课程。
        # 高 z 世界系上界按 go2arm URDF 软限位(0.9)下的随机前向可达样本估计：
        #   在 x∈[0.12, 0.30], |y|<=0.12 的前向窗口内，局部 z 的 99.5% 分位约为 0.8076，
        #   再加 nominal base height≈0.4m，取 world z 上界 1.20。
        # Start from forward targets so the error-gated reward naturally emphasizes locomotion first.
        stage1_position_range_b = (0.70, 1.60, 0.0, 0.0, 0.0, 0.0)
        stage2_position_range_b = (0.45, 2.20, 0.0, 0.0, 0.0, 0.0)
        stage3_position_range_b = (0.25, 2.50, 0.0, 0.0, 0.0, 0.0)
        # Fixed orientation range: keep targets inside the wrist workspace that remains practical with
        # the arm action range clipped to about 80% of the joint limits.
        stage3_euler_xyz_range_b = (-0.35, 0.35, -0.35, 0.35, -1.20, 1.20)
        stage1_euler_xyz_range_b = stage3_euler_xyz_range_b
        stage2_euler_xyz_range_b = stage3_euler_xyz_range_b
        self.commands.ee_pose.position_range_b = stage1_position_range_b
        self.commands.ee_pose.sample_z_in_world_frame = True
        self.commands.ee_pose.euler_xyz_range_b = stage1_euler_xyz_range_b

        # Scene.
        self.scene.robot = UNITREE_Go2Arm_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.merge_fixed_joints = False
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.FL_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_scanner_prim_path[0]
        self.scene.FR_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_scanner_prim_path[1]
        self.scene.RL_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_scanner_prim_path[2]
        self.scene.RR_foot_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_scanner_prim_path[3]
        self.scene.FL_foot_contact.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_scanner_prim_path[0]
        self.scene.FR_foot_contact.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_scanner_prim_path[1]
        self.scene.RL_foot_contact.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_scanner_prim_path[2]
        self.scene.RR_foot_contact.prim_path = "{ENV_REGEX_NS}/Robot/" + self.foot_scanner_prim_path[3]
        # 精确足端 filtered contact 需要过滤到真实地形碰撞 prim。
        terrain_contact_filter_prim_paths = self._terrain_contact_filter_prim_paths()
        self.scene.FL_foot_contact.filter_prim_paths_expr = terrain_contact_filter_prim_paths
        self.scene.FR_foot_contact.filter_prim_paths_expr = terrain_contact_filter_prim_paths
        self.scene.RL_foot_contact.filter_prim_paths_expr = terrain_contact_filter_prim_paths
        self.scene.RR_foot_contact.filter_prim_paths_expr = terrain_contact_filter_prim_paths

        # Treat every non-foot body as illegal contact.
        # This prevents the policy from using base/body ground contact as a stabilizing strategy.
        non_foot_contact_body_regex = GO2ARM_NON_FOOT_BODY_REGEX
        self.rewards.total_reward.params[
            "mani_regularization_support_non_foot_contact_sensor_cfg"
        ].body_names = non_foot_contact_body_regex
        self.rewards.total_reward.params["basic_collision_sensor_cfg"].body_names = non_foot_contact_body_regex

        # Validation override: make the arm and shoulder-side mount light so the base can drive reaching
        # with a small stabilization cost. Delete this event to restore physical masses.
        self.events.scale_arm_mass_validation = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["arm_mount", "link1", "link2", "link3", "link4", "link5", "link6"],
                    preserve_order=True,
                ),
                "mass_distribution_params": (1.0, 1.0),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )

        # Events.
        target_pos_range = self.commands.ee_pose.position_range_b or self.commands.ee_pose.position_range_w
        target_yaw_range = (
            self.commands.ee_pose.euler_xyz_range_b[4:6]
            if self.commands.ee_pose.euler_xyz_range_b is not None
            else self.commands.ee_pose.euler_xyz_range[4:6]
        )
        del target_pos_range, target_yaw_range
        root_reset_x_half_range = 0.02
        root_reset_y_half_range = 0.02
        root_reset_yaw_half_range = 0.08
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-root_reset_x_half_range, root_reset_x_half_range),
            "y": (-root_reset_y_half_range, root_reset_y_half_range),
            "yaw": (-root_reset_yaw_half_range, root_reset_yaw_half_range),
        }
        self.events.randomize_reset_joints.params["position_range"] = (-0.01, 0.01)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.02, 0.02)

        # Curriculum.
        self.curriculum.terrain_levels = None
        self.curriculum.go2arm_reaching_stages = CurrTerm(
            func=mdp.go2arm_reaching_stages,
            params={
                "command_name": "ee_pose",
                "steps_per_iteration": 24,
                "stage1_end_iteration": 100,
                "stage2_hold_end_iteration": 200,
                "stage2_expand_end_iteration": 300,
                "stage2_ratio_end_iteration": 425,
                "stage3_xy_end_iteration": 625,
                "stage2_expand_reach_fraction": 0.5,
                "stage2_ratio_reach_fraction": 0.5,
                "position_range_b_stage1_start": (0.70, 1.20, 0.0, 0.0, 0.0, 0.0),
                "position_range_b_stage1": stage1_position_range_b,
                "position_range_b_stage2_allowed_start": stage2_position_range_b,
                "position_range_b_stage3": stage3_position_range_b,
                "world_z_range_stage1": (0.45, 0.6),
                "world_z_range_stage2_allowed_start": (0.30, 0.7),
                "world_z_range_stage3": (0.1, 1.0),
                "euler_xyz_range_b_stage1": stage1_euler_xyz_range_b,
                "euler_xyz_range_b_stage2_allowed": stage2_euler_xyz_range_b,
                "euler_xyz_range_b_stage3": stage3_euler_xyz_range_b,
                "position_range_b_hard_low_start": (0.45, 2.20, 0.0, 0.0, 0.0, 0.0),
                "position_range_b_hard_low_final": (0.35, 2.50, 0.0, 0.0, 0.0, 0.0),
                "world_z_range_hard_low_start": (0.20, 0.35),
                "world_z_range_hard_low_final": (0.1, 0.35),
                "position_range_b_hard_high_start": (0.45, 2.20, 0.0, 0.0, 0.0, 0.0),
                "position_range_b_hard_high_final": (0.35, 2.50, 0.0, 0.0, 0.0, 0.0),
                "world_z_range_hard_high_start": (0.8, 0.9),
                "world_z_range_hard_high_final": (0.8, 1),
                "euler_xyz_range_b_hard_low": stage3_euler_xyz_range_b,
                "euler_xyz_range_b_hard_high": stage3_euler_xyz_range_b,
                "hard_low_sample_prob_stage2_base": 0.08,
                "hard_high_sample_prob_stage2_base": 0.08,
                "hard_low_sample_prob_stage2_final": 0.20,
                "hard_high_sample_prob_stage2_final": 0.15,
                "reset_joint_position_range_stage1": (-0.01, 0.01),
                "reset_joint_position_range_stage2": (-0.02, 0.02),
                "reset_joint_position_range_stage3": (-0.04, 0.04),
                "reset_joint_velocity_range_stage1": (-0.02, 0.02),
                "reset_joint_velocity_range_stage2": (-0.03, 0.03),
                "reset_joint_velocity_range_stage3": (-0.05, 0.05),
                "reset_root_x_range_stage1": (-root_reset_x_half_range, root_reset_x_half_range),
                "reset_root_x_range_stage2": (-0.03, 0.03),
                "reset_root_x_range_stage3": (-0.06, 0.06),
                "reset_root_y_range_stage1": (-root_reset_y_half_range, root_reset_y_half_range),
                "reset_root_y_range_stage2": (-0.03, 0.03),
                "reset_root_y_range_stage3": (-0.06, 0.06),
                "reset_root_yaw_range_stage1": (-root_reset_yaw_half_range, root_reset_yaw_half_range),
                "reset_root_yaw_range_stage2": (-0.10, 0.10),
                "reset_root_yaw_range_stage3": (-0.18, 0.18),
            },
        )

        # Rewards.
        # Training only uses total_reward.
        # Atomic reward logs are written separately by the go2arm local env.
        # Keep the stateful potential term alive for total_reward internal reuse.
        self.rewards.total_reward.weight = 1.0
        # 不能把势奖励项权重设成 0，否则 RewardManager 不会执行这个有状态 term，
        # total_reward 内部读取到的 last_reward / potential 缓存就会一直停留在 0。
        # 这里保留极小权重，仅用于激活实例与更新内部状态，对最终训练总奖励影响可忽略。
        self.rewards.ee_tracking_potential.weight = 1.0e-6

        self.rewards.total_reward.params["gating_mu"] = 0.65
        self.rewards.total_reward.params["gating_l"] = 0.9
        self.rewards.total_reward.params["gating_fixed_d"] = None
        # 放大势奖励，让“这一步比上一步更接近目标”在误差较大时成为主导正反馈。
        # 这次直接把该项改成 -tracking_error，因此外层直接给较大的固定权重来验证驱动效果。
        self.rewards.total_reward.params["mani_potential_weight"] = 50.0
        # 当前主要问题已经不是生存，而是 reaching 不够积极，因此把跟踪奖励做得更稠密一些。
        self.rewards.total_reward.params["mani_position_std"] = 0.35
        self.rewards.total_reward.params["mani_orientation_std"] = 0.30
        # 倒数型势函数：phi = clip(0.3 / error, 0, 5)。
        self.rewards.ee_tracking_potential.params["gain"] = 0.3
        self.rewards.ee_tracking_potential.params["clip_min"] = 0.0
        self.rewards.ee_tracking_potential.params["clip_max"] = 5.0
        self.rewards.total_reward.params["mani_regularization_support_roll_weight"] = 0.1
        self.rewards.total_reward.params["mani_regularization_support_roll_std"] = 0.15**2
        # 当前已经不存在明显非法碰撞，下一步重点抑制足端在地面上的横向蹭滑。
        self.rewards.total_reward.params["mani_regularization_support_feet_slide_weight"] = 0.10
        self.rewards.total_reward.params["mani_regularization_support_feet_slide_std"] = 0.1
        self.rewards.total_reward.params["mani_regularization_support_foot_air_weight"] = 0.16
        self.rewards.total_reward.params["mani_regularization_support_foot_air_clip_max"] = 2.0
        self.rewards.total_reward.params["mani_regularization_support_non_foot_contact_weight"] = 0.2
        self.rewards.total_reward.params["mani_regularization_support_non_foot_contact_threshold"] = 1.0
        self.rewards.total_reward.params["mani_regularization_support_non_foot_contact_count_weight"] = 1.0
        self.rewards.total_reward.params["mani_regularization_support_non_foot_contact_force_weight"] = 0.5
        self.rewards.total_reward.params["mani_regularization_support_non_foot_contact_force_scale"] = 30.0
        self.rewards.total_reward.params["mani_regularization_support_non_foot_contact_clip_max"] = 4.0
        self.rewards.total_reward.params["mani_regularization_target_height_pitch_weight"] = 0.08
        self.rewards.total_reward.params["mani_regularization_target_height_pitch_low_height"] = 0.45
        self.rewards.total_reward.params["mani_regularization_target_height_pitch_high_height"] = 0.95
        self.rewards.total_reward.params["mani_regularization_target_height_pitch_std"] = 0.15
        self.rewards.total_reward.params["mani_regularization_min_base_height_weight"] = 0.10
        self.rewards.total_reward.params["mani_regularization_min_base_height_sensor_cfg"] = (
            self.rewards.total_reward.params["loco_regularization_base_height_sensor_cfg"]
        )
        self.rewards.total_reward.params["mani_regularization_min_base_height_minimum_height"] = 0.20
        self.rewards.total_reward.params["mani_regularization_min_base_height_std"] = 0.05
        self.rewards.total_reward.params["mani_regularization_posture_deviation_weight"] = 0.05
        self.rewards.total_reward.params["mani_regularization_posture_deviation_std"] = math.sqrt(6.0) * 0.4
        self.rewards.total_reward.params["mani_regularization_posture_deviation_joint_weights"] = (
            1.0,
            1.25,
            0.75,
            1.0,
            1.0,
            1.0,
        )
        self.rewards.total_reward.params["mani_regularization_joint_limit_safety_weight"] = 0.1
        self.rewards.total_reward.params["mani_regularization_joint_limit_safety_std"] = 1.0
        self.rewards.total_reward.params["mani_regularization_support_left_right_x_symmetry_weight"] = 0.06
        self.rewards.total_reward.params["mani_regularization_support_left_right_x_symmetry_std"] = 0.04
        self.rewards.total_reward.params["mani_regularization_support_left_right_y_symmetry_weight"] = 0.08
        self.rewards.total_reward.params["mani_regularization_support_left_right_y_symmetry_std"] = 0.03
        self.rewards.total_reward.params["mani_regularization_support_symmetry_max_base_lin_speed"] = 0.08
        self.rewards.total_reward.params["mani_regularization_support_symmetry_max_base_ang_speed"] = 0.25
        self.rewards.total_reward.params["mani_cumulative_error_clip_max"] = 20.0
        self.rewards.total_reward.params["workspace_position_weight"] = 1.0
        self.rewards.total_reward.params["workspace_position_x_min"] = 0.25
        self.rewards.total_reward.params["workspace_position_x_max"] = 0.35
        self.rewards.total_reward.params["workspace_position_y_weight"] = 0.2
        self.rewards.total_reward.params["workspace_position_std"] = 0.5
        self.rewards.total_reward.params["workspace_position_clip_max"] = 1.5
        # 继续保留轻度高度约束，但当前主要问题更偏向前倾和竖直速度，因此高度项不额外拉太高。
        self.rewards.total_reward.params["loco_regularization_base_height_weight"] = 0.03
        self.rewards.total_reward.params["loco_regularization_base_height_std"] = 0.05
        self.rewards.total_reward.params["loco_regularization_base_height_target_height"] = 0.4
        self.rewards.total_reward.params["loco_regularization_base_roll_weight"] = 0.12
        self.rewards.total_reward.params["loco_regularization_base_roll_std"] = 0.1
        # 日志里 base_pitch 持续偏大，适当加重前后俯仰约束。
        self.rewards.total_reward.params["loco_regularization_base_pitch_weight"] = 0.10
        self.rewards.total_reward.params["loco_regularization_base_pitch_std"] = 0.1
        self.rewards.total_reward.params["loco_regularization_base_roll_ang_vel_weight"] = 0.05
        self.rewards.total_reward.params["loco_regularization_base_roll_ang_vel_std"] = 0.5
        # 俯仰角速度同样偏大，增加动态稳定性惩罚，抑制点头式下砸。
        self.rewards.total_reward.params["loco_regularization_base_pitch_ang_vel_weight"] = 0.05
        self.rewards.total_reward.params["loco_regularization_base_pitch_ang_vel_std"] = 0.5
        # 继续压制 base 竖直方向速度，减少反复下砸和弹跳。
        self.rewards.total_reward.params["loco_regularization_base_z_vel_weight"] = 0.04
        self.rewards.total_reward.params["loco_regularization_base_z_vel_std"] = 0.1
        # 把 base 在 body frame 下的侧向漂移也压进去，专门抑制“先偏一个 yaw 再斜着向前走”。
        self.rewards.total_reward.params["loco_regularization_base_lateral_vel_weight"] = 0.12
        self.rewards.total_reward.params["loco_regularization_base_lateral_vel_std"] = 0.15
        # 给腿部一个轻度的默认构型先验，优先约束 hip，其次 thigh，最后 calf，避免步态过分别扭。
        self.rewards.total_reward.params["loco_regularization_leg_posture_deviation_weight"] = 0.04
        self.rewards.total_reward.params["loco_regularization_leg_posture_deviation_std"] = (
            math.sqrt(4.0 * (1.0 + 0.7 + 0.4)) * 0.25
        )
        self.rewards.total_reward.params["loco_regularization_leg_posture_deviation_joint_weights"] = (
            1.2,
            0.7,
            0.4,
            1.2,
            0.7,
            0.4,
            1.2,
            0.7,
            0.4,
            1.2,
            0.7,
            0.4,
        )
        # locomotion 阶段改为约束左右脚最近一次触地落脚点的对称性，分别约束 x/y 两个轴向。
        self.rewards.total_reward.params["loco_regularization_touchdown_left_right_x_symmetry_weight"] = 0.10
        self.rewards.total_reward.params["loco_regularization_touchdown_left_right_x_symmetry_std"] = 0.04
        self.rewards.total_reward.params["loco_regularization_touchdown_left_right_y_symmetry_weight"] = 0.12
        self.rewards.total_reward.params["loco_regularization_touchdown_left_right_y_symmetry_std"] = 0.03
        self.rewards.total_reward.params["loco_regularization_touchdown_foot_y_distance_weight"] = 0.1
        self.rewards.total_reward.params["loco_regularization_touchdown_foot_y_distance_std"] = 0.03
        self.rewards.total_reward.params["loco_regularization_touchdown_foot_y_distance_min_distance"] = 0.15
        self.rewards.total_reward.params["loco_regularization_diagonal_foot_symmetry_weight"] = 0.2
        self.rewards.total_reward.params["loco_regularization_diagonal_foot_symmetry_std"] = 0.05
        self.rewards.total_reward.params["loco_regularization_diagonal_foot_symmetry_sensor_cfg"] = (
            self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_sensor_cfg"]
        )
        # 适度提高步态接触规律权重，让四个足端的接触分布更稳定。
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_weight"] = 0.8
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_force_std"] = 1.0
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_height_std"] = 0.0025
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_vel_std"] = 0.01
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_cycle_time"] = 0.40
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_phase_offsets"] = (
            GO2ARM_TROT_PHASE_OFFSETS
        )
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_swing_height"] = 0.10
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_soft_contact_k"] = 6.0
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_contact_force_threshold"] = 2.0
        self.rewards.total_reward.params["loco_regularization_feet_contact_soft_trot_support_factor_low"] = 0.5
        self.rewards.total_reward.params["loco_tracking_std"] = 0.5
        self.rewards.total_reward.params["loco_tracking_threshold"] = 0.05
        self.rewards.total_reward.params["loco_tracking_weight"] = 5.0
        self.rewards.total_reward.params["loco_arm_swing_weight"] = 0.15
        # 静态姿态约束之外，再补一个较轻的 arm 动态抑制，优先压制 locomotion 阶段肘关节快速上下摆动。
        self.rewards.total_reward.params["loco_arm_dynamic_weight"] = 0.01
        self.rewards.total_reward.params["basic_is_alive_weight"] = 0.2
        self.rewards.total_reward.params["basic_collision_weight"] = -5.0
        self.rewards.total_reward.params["basic_collision_threshold"] = 1.0
        self.rewards.total_reward.params["basic_collision_count_weight"] = 1.0
        self.rewards.total_reward.params["basic_collision_force_weight"] = 0.5
        self.rewards.total_reward.params["basic_collision_force_scale"] = 20.0
        self.rewards.total_reward.params["basic_action_smoothness_first_weight"] = -0.005
        self.rewards.total_reward.params["basic_action_smoothness_second_weight"] = -0.0015
        self.rewards.total_reward.params["basic_joint_torque_sq_weight"] = -1.6e-3
        self.rewards.total_reward.params["basic_joint_power_weight"] = -1.32e-2

        # Do not drop zero-weight rewards here.
        # total_reward still needs ee_tracking_potential as an internal stateful term.

        # Terminations.
        # Any non-foot contact contributes to illegal-contact termination.
        self.terminations.non_foot_contact_termination.params["sensor_cfg"].body_names = non_foot_contact_body_regex
        self.terminations.base_orientation_termination.params["asset_cfg"].body_names = [self.base_link_name]
        self.terminations.base_orientation_termination.params["soft_roll_pitch_limit"] = 0.20
        self.terminations.base_orientation_termination.params["hard_roll_pitch_limit"] = 0.35
        self.terminations.base_orientation_termination.params["consecutive_steps"] = 5
        self.terminations.base_height_termination.params["asset_cfg"].body_names = [self.base_link_name]
        # 当前主要 termination 已经从力矩切换为 base_height，先略微放宽高度边界，避免过早截断稳定化学习。
        self.terminations.base_height_termination.params["soft_minimum_height"] = 0.20
        self.terminations.base_height_termination.params["hard_minimum_height"] = 0.16
        self.terminations.base_height_termination.params["consecutive_steps"] = 5
        self.terminations.task_success.params["asset_cfg"].body_names = [self.base_link_name]

        # Curriculums.
        self.scene.terrain.max_init_terrain_level = 0

        # Commands.

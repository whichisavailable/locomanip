# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import re
from collections.abc import Callable
from dataclasses import MISSING
from typing import cast

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import observations as mdp_obs
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg, CommandsCfg, ObservationsCfg

# 添加实验室的包
# import isaaclab_nhb.tasks.mdp_nhb as mdp_nhb
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##

GO2ARM_LEG_JOINT_NAMES = [
    # 左前腿
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    # 右前腿
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    # 左后腿
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    # 右后腿
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# 机械臂 6 个关节。
GO2ARM_ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# 机器人总关节顺序：腿在前，机械臂在后。
GO2ARM_ALL_JOINT_NAMES = GO2ARM_LEG_JOINT_NAMES + GO2ARM_ARM_JOINT_NAMES
GO2ARM_BASE_BODY_NAME = "base_link"

# 四个足端 body 名称。
GO2ARM_FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
GO2ARM_FOOT_NAMES = list(GO2ARM_FOOT_BODY_NAMES)
GO2ARM_CONTACT_SENSOR_PRIM_PATH = (
    "{ENV_REGEX_NS}/Robot/.*(?:FL_foot|FR_foot|RL_foot|RR_foot|"
    "(?:FL|FR|RL|RR)_calf(?:_link)?|base_link|link[1-6])$"
)
# Global contact covers feet plus selected illegal-contact bodies; dedicated foot sensors still define legal support.
GO2ARM_NON_FOOT_BODY_REGEX = [r"^(?!.*(?:FL_foot|FR_foot|RL_foot|RR_foot)$).+"]
# 预设 trot 步态偏置。
GO2ARM_TROT_PHASE_OFFSETS = (0.0, 0.5, 0.5, 0.0)
GO2ARM_LEGAL_SUPPORT_BODY_NAMES = list(GO2ARM_FOOT_BODY_NAMES)

# 四个足端高度扫描器的名字，顺序与 GO2ARM_FOOT_NAMES 保持一致。
GO2ARM_FOOT_SCANNER_NAMES = (
    "FL_foot_scanner",
    "FR_foot_scanner",
    "RL_foot_scanner",
    "RR_foot_scanner",
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """go2arm 任务的场景配置。"""

    # 地形实体。
    terrain = TerrainImporterCfg(
        # 地形在 USD 场景树中的挂载路径。
        prim_path="/World/ground",
        # 默认仍保留 generator 形式，flat 环境会在子类里改成 plane。
        terrain_type="generator",
        # rough 版本使用的地形生成器。
        terrain_generator=ROUGH_TERRAINS_CFG,
        # 初始 terrain level；flat 任务里这项不会真正用到。
        max_init_terrain_level=5,
        # 让地形参与默认碰撞组。
        collision_group=-1,
        # 地形物理材质。
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # 摩擦系数组合方式使用 multiply。
            friction_combine_mode="multiply",
            # 恢复系数组合方式也使用 multiply。
            restitution_combine_mode="multiply",
            # 地形静摩擦默认值。
            static_friction=1.0,
            # 地形动摩擦默认值。
            dynamic_friction=1.0,
            # 地形恢复系数固定为 0，通过 multiply 与足端材质共同决定接触行为。
            restitution=0.0,
        ),
        # 地形渲染材质，仅影响显示，不影响物理。
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        # 默认关闭地形调试可视化。
        debug_vis=False,
    )
    # 机器人本体，由具体任务子类赋值。
    robot: ArticulationCfg = MISSING
    # 机器人主体高度扫描器。
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + GO2ARM_BASE_BODY_NAME,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 基座附近的小范围高度扫描器，用于估计基座局部地形高度。
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + GO2ARM_BASE_BODY_NAME,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 四个足端各自的地形扫描器，用于 privileged 观测和支撑相关奖励。
    FL_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.2)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    FR_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.2)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    RL_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.2)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    RR_foot_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.2, 0.2)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # go2arm 的全局 contact sensor 只保留任务语义真正需要的 body：
    # 1. foot/calf/calflower 用于精确足端接触与重复接触去重；
    # 2. arm links 用于非法接触与碰撞惩罚。
    # 足端 air/contact time 由四个独立 foot sensor 维护，因此这里不再跟踪 air time。
    contact_forces = ContactSensorCfg(
        prim_path=GO2ARM_CONTACT_SENSOR_PRIM_PATH,
        history_length=3,
        track_air_time=False,
    )
    # go2arm 精确足端接触专用 filtered sensor。
    # 每个 sensor 只绑定一个 foot body，避免把整机多 body contact sensor 直接拿去做 filtered contact。
    FL_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        history_length=3,
        track_air_time=True,
        # 绮剧‘瓒崇璇箟渚濊禆 PhysX patch point / friction patch buffer锛岃繖閲屽繀椤绘樉寮忓惎鐢ㄣ€?
        # 涓嶅啀璧板惎鐢ㄦ爣蹇楀叧闂絾鐩存帴璇诲簳灞俿iew 鐨勭伆鑹茶矾寰勶紝閬垮厤 patch 鏁版嵁鍦ㄤ笉鍚屽湴褰?/
        # Isaac Sim 鐗堟湰涓嬪嚭鐜伴儴鍒嗘湭瀹氫箟鎴栧潗鏍囪В閲婁笉绋冲畾鐨勬儏鍐点€?
        track_contact_points=False,
        track_friction_forces=False,
        # 精确足端 contact 需要读取 PhysX patch 级数据；8 在 rough terrain 或小腿擦地时偏小，
        # 容易把同一刚体上的多 patch 接触截断掉，导致合法足端力和非法小腿接触都被低估。
        # 这里先把上限抬高到更稳妥的 32，只扩大缓冲，不改变现有接触语义。
        max_contact_data_count_per_prim=32,
    )
    FR_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        history_length=3,
        track_air_time=True,
        track_contact_points=False,
        track_friction_forces=False,
        # 同 FL_foot_contact：增大 patch 缓冲上限，避免多 patch 接触被截断。
        max_contact_data_count_per_prim=32,
    )
    RL_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        history_length=3,
        track_air_time=True,
        track_contact_points=False,
        track_friction_forces=False,
        # 同 FL_foot_contact：增大 patch 缓冲上限，避免多 patch 接触被截断。
        max_contact_data_count_per_prim=32,
    )
    RR_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        history_length=3,
        track_air_time=True,
        track_contact_points=False,
        track_friction_forces=False,
        # 同 FL_foot_contact：增大 patch 缓冲上限，避免多 patch 接触被截断。
        max_contact_data_count_per_prim=32,
    )
    # 场景光照。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class Go2ArmTeacherCommandsCfg:
    """go2arm teacher 配置里可复用的命令片段。"""

    # 只保留末端位姿命令。
    ee_pose = mdp.EndEffectorPoseCommandCfg(
        # 命令作用的资产名。
        asset_name="robot",
        # 末端执行器 body 名。
        ee_body_name="link6",
        # 基座 body 名。
        base_body_name=GO2ARM_BASE_BODY_NAME,
        # episode 内保持不变，只在 reset 时重采样。
        resampling_time_range=(1.0e9, 1.0e9),
        # 世界系位置范围仅用于兼容旧配置；z 下界取地面。
        position_range_w=(0.15, 0.55, -0.35, 0.35, 0.0, 0.45),
        # 世界系欧拉角范围仅用于兼容旧配置；go2arm 实际使用 base frame 采样。
        euler_xyz_range=(-0.5, 0.5, -0.5, 0.5, -3.14159, 3.14159),
        # 先在 base frame 内采样，再映射到世界系；z 允许低于 base 高度，对应地面附近操作。
        position_range_b=(0.05, 0.55, -0.35, 0.35, -0.30, 0.15),
        euler_xyz_range_b=(-0.5, 0.5, -0.5, 0.5, -3.14159, 3.14159),
        # 按 go2arm URDF 中 base(0.3762x0.0935x0.114) 与 arm_mount 盒体并集，加少量裕量过滤机身核心附近坏命令。
        reject_position_cuboid=(-0.20, 0.20, -0.06, 0.06, -0.08, 0.10),
        # 单次采样最大尝试次数。
        max_sampling_tries=128,
        # 跟踪误差中位置项权重。
        position_error_weight=1.0,
        # 跟踪误差中姿态项权重。
        orientation_error_weight=0.1,
        # 参考误差下降速度，后面可按需要调整。
        reference_error_velocity=0.0,
        # 累积误差固定权重，后续可替换成动态权重。
        cumulative_error_weight=1.0,
    )


class Go2ArmDefaultDeltaJointPositionAction(joint_actions.JointPositionAction):
    """以默认关节姿态为零点、直接输出关节偏移量的动作项。"""

    cfg: "Go2ArmDefaultDeltaJointPositionActionCfg"

    def __init__(self, cfg: "Go2ArmDefaultDeltaJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._env = env
        # 这里显式解析“偏移量 clip”，而不是沿用 JointAction 里对最终绝对目标的 clip。
        # 目标是让策略直接输出“弧度偏移”，再把偏移限制在一个合理小范围内。
        self._delta_clip = None
        if cfg.delta_clip is not None:
            self._delta_clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self.action_dim, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.delta_clip, self._joint_names)
            self._delta_clip[:, index_list] = torch.tensor(value_list, device=self.device)
        self._fixed_delta_action_joint_ids = None
        if cfg.fixed_delta_action_joint_names is not None:
            fixed_ids = [
                joint_id
                for joint_id, joint_name in enumerate(self._joint_names)
                if any(re.fullmatch(pattern, joint_name) for pattern in cfg.fixed_delta_action_joint_names)
            ]
            if fixed_ids:
                self._fixed_delta_action_joint_ids = torch.tensor(fixed_ids, dtype=torch.long, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        # Store the effective action after any curriculum mask, so observations/rewards see executed deltas.
        effective_actions = actions
        if self._fixed_delta_action_joint_ids is not None and self.cfg.fixed_delta_action_until_iteration is not None:
            step = int(getattr(self._env, "common_step_counter", 0))
            current_iteration = float(step) / float(max(self.cfg.fixed_delta_action_steps_per_iteration, 1))
            if current_iteration < float(self.cfg.fixed_delta_action_until_iteration):
                effective_actions = actions.clone()
                effective_actions[:, self._fixed_delta_action_joint_ids] = float(self.cfg.fixed_delta_action_value)
        prev_effective_action = getattr(self._env, "_go2arm_effective_action", None)
        prev_prev_effective_action = getattr(self._env, "_go2arm_prev_effective_action", None)
        if prev_effective_action is None:
            prev_effective_action = torch.zeros_like(effective_actions)
        if prev_prev_effective_action is None:
            prev_prev_effective_action = torch.zeros_like(effective_actions)
        self._env._go2arm_prev_prev_effective_action = prev_prev_effective_action.clone()
        self._env._go2arm_prev_effective_action = prev_effective_action.clone()
        self._env._go2arm_effective_action = effective_actions.detach().clone()
        self._raw_actions[:] = effective_actions
        delta_actions = self._raw_actions
        if self._delta_clip is not None:
            # 这里 clip 的是“相对默认姿态的关节偏移量”，不是最终绝对关节目标。
            delta_actions = torch.clamp(delta_actions, min=self._delta_clip[:, :, 0], max=self._delta_clip[:, :, 1])
        # 最终目标保持为：default_joint_pos + delta_action，不再额外乘一个 scale。
        self._processed_actions = delta_actions + self._offset

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        for attr_name in (
            "_go2arm_effective_action",
            "_go2arm_prev_effective_action",
            "_go2arm_prev_prev_effective_action",
        ):
            value = getattr(self._env, attr_name, None)
            if value is not None:
                value[env_ids] = 0.0


@configclass
class Go2ArmDefaultDeltaJointPositionActionCfg(mdp.JointPositionActionCfg):
    """go2arm 默认姿态中心关节偏移动作配置。"""

    class_type: type[ActionTerm] = Go2ArmDefaultDeltaJointPositionAction
    # 对“关节偏移量”做 clip，而不是对最终绝对关节目标做 clip。
    delta_clip: dict[str, tuple[float, float]] | None = None
    fixed_delta_action_joint_names: list[str] | None = None
    fixed_delta_action_until_iteration: int | None = None
    fixed_delta_action_steps_per_iteration: int = 24
    fixed_delta_action_value: float = 0.0


@configclass
class Go2ArmTeacherActionsCfg:
    """go2arm teacher 配置里可复用的默认姿态中心关节位置偏移动作。"""

    # 18 维动作，覆盖 12 个腿关节和 6 个机械臂关节。
    # 这里的动作语义不是“网络直接输出最终绝对关节角”，
    # 而是以机器人默认关节姿态为中心，在其附近输出一个缩放后的关节位置偏移。
    # 这里进一步去掉了隐藏的 scale 乘法，改成更直接的：
    # target_joint_pos = default_joint_pos + clipped_delta_action。
    joint_pos = Go2ArmDefaultDeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=GO2ARM_ALL_JOINT_NAMES,
        # 不再用 scale 改写策略输出的语义；策略输出本身就表示关节偏移量（单位弧度）。
        scale=1.0,
        # 以默认关节姿态作为动作零点，而不是把 0 动作解释成 0rad 绝对关节角。
        use_default_offset=True,
        # 直接限制每个关节相对默认姿态的最大偏移量，保持与之前约 0.1rad 的有效动作幅度接近。
        delta_clip={
            "^(FL|FR|RL|RR)_.*$": (-0.1, 0.1),
            "^joint[1-6]$": (-0.2, 0.2),
        },
        # 最终绝对目标不再额外做 clip，避免把 offset 后的目标再次按绝对值截断。
        clip=None,
        preserve_order=True,
    )


@configclass
class Go2ArmTeacherCoreObsCfg(ObsGroup):
    """go2arm teacher 中 actor 和 critic 共用的核心观测。"""

    # 18 个关节的相对位置。
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ALL_JOINT_NAMES, preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 18 个关节的相对速度。
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ALL_JOINT_NAMES, preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 当前末端执行器在 base frame 下的位置和姿态。
    ee_current_pose = ObsTerm(
        func=mdp_obs.end_effector_pose_b,
        params={
            "ee_body_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            "base_body_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 上一步动作。
    actions = ObsTerm(
        func=mdp.last_action,
        params={"action_name": "joint_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 当前末端位姿命令，3 维位置 + 9 维旋转矩阵。
    ee_pose_command = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "ee_pose"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 基座线速度。
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 基座角速度。
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 重力在 base frame 下的投影。
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 参考跟踪误差 e_ref(t)。
    reference_tracking_error = ObsTerm(
        func=mdp_obs.command_term_reference_tracking_error,
        params={"command_name": "ee_pose"},
        clip=(0.0, 100.0),
        scale=1.0,
    )
    # 4 维 trot 步态相位信号。
    gait_phase = ObsTerm(
        func=mdp_obs.trot_phase_sin,
        params={"cycle_time": 0.5, "phase_offsets": GO2ARM_TROT_PHASE_OFFSETS},
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        # 这里先不加噪声，后面可按训练阶段再决定。
        self.enable_corruption = False
        # 所有观测项按顺序拼接成一个张量。
        self.concatenate_terms = True


@configclass
class Go2ArmTeacherPrivilegedObsCfg(ObsGroup):
    """go2arm teacher 中原始特权信息的可复用观测块。"""

    # 基座相对脚下地形的高度。
    base_height = ObsTerm(
        func=mdp_obs.base_height_from_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner_base")},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 四个足端各自的离地高度。
    foot_heights = ObsTerm(
        func=mdp_obs.foot_heights_from_scanners,
        params={
            "sensor_names": GO2ARM_FOOT_SCANNER_NAMES,
            "asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES),
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 四个足端的接触力。
    feet_contact_forces = ObsTerm(
        func=mdp_obs.feet_contact_forces,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2ARM_FOOT_BODY_NAMES)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 当前足端材质的静摩擦系数。
    static_friction = ObsTerm(
        func=mdp_obs.static_friction,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES)},
        clip=(0.0, 10.0),
        scale=1.0,
    )
    # 基座受到的外力和外力矩。
    base_external_wrench = ObsTerm(
        func=mdp_obs.base_external_wrench,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME])},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 基座被 interval push 注入的速度扰动。
    base_external_push_velocity = ObsTerm(
        func=mdp_obs.base_external_push_velocity,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME])},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 基座质量随机化带来的扰动量。
    base_mass_disturbance = ObsTerm(
        func=mdp_obs.base_mass_disturbance,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME])},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 末端 link6 质量随机化带来的扰动量。
    ee_mass_disturbance = ObsTerm(
        func=mdp_obs.ee_mass_disturbance,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"])},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 末端执行器受到的外力和外力矩。
    ee_external_wrench = ObsTerm(
        func=mdp_obs.ee_external_wrench,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["link6"])},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 末端执行器相对基座的线速度和角速度。
    ee_velocity_b = ObsTerm(
        func=mdp_obs.body_velocity_b,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            "base_body_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 四个足端在世界系中的平面速度。
    feet_planar_velocities_w = ObsTerm(
        func=mdp_obs.feet_planar_velocities_w,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_FOOT_BODY_NAMES),
            "base_body_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # 观测延迟的 privileged 标量。
    observation_delay = ObsTerm(
        func=mdp_obs.observation_delay,
        clip=(0.0, 100.0),
        scale=1.0,
    )

    def __post_init__(self):
        # 特权观测默认不加噪声，因为后面会单独经过特权编码器。
        self.enable_corruption = False
        # 原始特权信息按固定顺序拼接，便于后续编码器读取。
        self.concatenate_terms = True


@configclass
class Go2ArmTeacherCriticExtraObsCfg(ObsGroup):
    """go2arm teacher 中 critic 专属的额外观测。"""

    # critic 额外看的累积误差。
    cumulative_tracking_error = ObsTerm(
        func=mdp_obs.command_term_cumulative_tracking_error,
        params={"command_name": "ee_pose"},
        clip=(0.0, 1e6),
        scale=1.0,
    )

    def __post_init__(self):
        # critic 额外观测默认不加噪声。
        self.enable_corruption = False
        # 这些额外项同样按顺序拼接。
        self.concatenate_terms = True


@configclass
class EventCfg:
    """go2arm 任务的事件配置。"""

    # 先验证奖励机制时，仅保留 reset 随机化；其余动力学随机化后续再开。
    randomize_rigid_body_material = None
    randomize_rigid_body_mass_base = None
    randomize_rigid_body_mass_ee = None
    # 基座外力域随机化：在 reset 时对 base_link 采样常值外力/外力矩，并在整个 episode 内持续生效。
    randomize_apply_external_force_torque_base = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
            "force_range": (-20.0, 20.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    # 末端执行器持续外 wrench：reset 时对 link6 采样常值外力；未显式给出力矩时保持为 0。
    randomize_apply_external_force_torque_ee = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            "force_range": (-3.0, 3.0),
            "torque_range": (0.0, 0.0),
        },
    )
    # 偶发 push：每隔 6-8s 通过直接改 root 的 6 维速度施加一次瞬时扰动。
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.15, 0.15),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.15, 0.15),
            }
        },
    )
    randomize_com_positions = None
    # 关节初始状态随机化
    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # 同时随机腿部和机械臂关节的初始状态。
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FL_hip_joint",
                    "FL_thigh_joint",
                    "FL_calf_joint",
                    "FR_hip_joint",
                    "FR_thigh_joint",
                    "FR_calf_joint",
                    "RL_hip_joint",
                    "RL_thigh_joint",
                    "RL_calf_joint",
                    "RR_hip_joint",
                    "RR_thigh_joint",
                    "RR_calf_joint",
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                ],
            ),
            # 关节初始位置扰动范围。
            "position_range": (-0.1, 0.1),
            # 关节初始速度扰动范围。
            "velocity_range": (-0.2, 0.2),
        },
    )
    # base初始状态随机化
    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
            # 只随机平移和偏航；roll/pitch 不在 reset 时额外打扰。
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.35, 0.35)},
            "velocity_range": {
                # 初始线速度保持为 0。
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                # 初始角速度保持为 0。
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """go2arm 任务的奖励配置。"""

    # go2arm teacher 奖励主线只保留总奖励入口。
    total_reward = RewTerm(
        # 统一奖励入口，内部再拆分 manipulation / locomotion / basic 三大块。
        func=mdp.total_reward,
        # 基类里先置 0，具体任务在 rough/flat 子类里再覆盖成真正使用的权重。
        weight=0.0,
        params={
            # 门控函数 D(reference_tracking_error) 的参数。
            # 总奖励内部固定读取 ee_pose 这条命令项。
            "gating_command_name": "ee_pose",
            # 门控函数的均值参数，控制从 manipulation 过渡到 locomotion 的中心位置。
            "gating_mu": 1.5,
            # 门控函数的斜率参数，决定切换有多陡。
            "gating_l": 1.0,
            # manipulation 主项。
            # 位置跟踪项使用 ee_pose 当前命令。
            "mani_position_command_name": "ee_pose",
            # 位置误差的指数奖励标准差。
            "mani_position_std": math.sqrt(0.005),
            # 位置误差奖励的幂次。
            "mani_position_power": 5.0,
            # 姿态跟踪项同样使用 ee_pose 当前命令。
            "mani_orientation_command_name": "ee_pose",
            # 姿态误差的指数奖励标准差。
            "mani_orientation_std": math.sqrt(0.01),
            # 姿态误差奖励的幂次。
            "mani_orientation_power": 5.0,
            # manipulation 正则项。
            # 基座横滚稳定性正则的权重。
            "mani_regularization_support_roll_weight": 0.0,
            # 读取基座姿态的 body 配置。
            "mani_regularization_support_roll_asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_BASE_BODY_NAME),
            "mani_regularization_support_roll_std": 0.15**2,
            # 支撑足滑动惩罚权重。
            "mani_regularization_support_feet_slide_weight": 0.0,
            # 足端接触力传感器配置。
            "mani_regularization_support_feet_slide_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GO2ARM_FOOT_BODY_NAMES
            ),
            # 足端刚体配置。
            "mani_regularization_support_feet_slide_asset_cfg": SceneEntityCfg(
                "robot", body_names=GO2ARM_FOOT_BODY_NAMES
            ),
            "mani_regularization_support_feet_slide_std": 0.1,
            # 足端不触地惩罚权重。
            "mani_regularization_support_foot_air_weight": 0.0,
            # 判断足端是否离地的接触力阈值。
            "mani_regularization_support_foot_air_threshold": 1.0,
            # 足端离地判断所用的接触传感器。
            "mani_regularization_support_foot_air_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GO2ARM_FOOT_BODY_NAMES
            ),
            "mani_regularization_support_foot_air_clip_max": 2.0,
            # 非足端接触惩罚总权重。
            "mani_regularization_support_non_foot_contact_weight": 0.0,
            # 非足端接触噪声过滤阈值。
            "mani_regularization_support_non_foot_contact_threshold": 1.0,
            # 非足端接触检测所用传感器；这里要和当前合法支撑端定义保持一致，排除四个 foot。
            "mani_regularization_support_non_foot_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GO2ARM_NON_FOOT_BODY_REGEX
            ),
            # 非足端接触里“发生了几个 body 接触”的计数项权重。
            "mani_regularization_support_non_foot_contact_count_weight": 1.0,
            # 非足端接触里“超阈值力大小”的幅值项权重。
            "mani_regularization_support_non_foot_contact_force_weight": 1.0,
            # 非足端接触幅值项的归一化尺度。
            "mani_regularization_support_non_foot_contact_force_scale": 10.0,
            "mani_regularization_support_non_foot_contact_clip_max": 4.0,
            # Target-height-conditioned base pitch direction regularization.
            "mani_regularization_target_height_pitch_weight": 0.0,
            "mani_regularization_target_height_pitch_command_name": "ee_pose",
            "mani_regularization_target_height_pitch_asset_cfg": SceneEntityCfg(
                "robot", body_names=GO2ARM_BASE_BODY_NAME
            ),
            "mani_regularization_target_height_pitch_low_height": 0.45,
            "mani_regularization_target_height_pitch_high_height": 0.95,
            "mani_regularization_target_height_pitch_std": 0.15,
            "mani_regularization_min_base_height_weight": 0.0,
            "mani_regularization_min_base_height_asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_BASE_BODY_NAME),
            "mani_regularization_min_base_height_sensor_cfg": None,
            "mani_regularization_min_base_height_minimum_height": 0.20,
            "mani_regularization_min_base_height_std": 0.05,
            # 机械臂姿态偏离中性位形的正则权重。
            "mani_regularization_posture_deviation_weight": 0.0,
            # 机械臂姿态偏离正则读取的关节集合。
            "mani_regularization_posture_deviation_asset_cfg": SceneEntityCfg(
                "robot", joint_names=GO2ARM_ARM_JOINT_NAMES, preserve_order=True
            ),
            "mani_regularization_posture_deviation_std": math.sqrt(6.0) * 0.4,
            "mani_regularization_posture_deviation_joint_weights": None,
            # 机械臂靠近关节限位时的安全正则权重。
            "mani_regularization_joint_limit_safety_weight": 0.0,
            # 机械臂关节限位安全项读取的关节集合。
            "mani_regularization_joint_limit_safety_asset_cfg": SceneEntityCfg(
                "robot", joint_names=GO2ARM_ARM_JOINT_NAMES
            ),
            "mani_regularization_joint_limit_safety_std": 1.0,
            "mani_regularization_support_left_right_x_symmetry_weight": 0.0,
            "mani_regularization_support_left_right_x_symmetry_std": 0.05,
            "mani_regularization_support_left_right_y_symmetry_weight": 0.0,
            "mani_regularization_support_left_right_y_symmetry_std": 0.05,
            "mani_regularization_support_foot_xy_range_weight": 0.0,
            "mani_regularization_support_foot_xy_range_std": 0.02,
            "mani_regularization_support_foot_xy_range_x_abs_min": 0.0,
            "mani_regularization_support_foot_xy_range_x_abs_max": None,
            "mani_regularization_support_foot_xy_range_y_abs_min": 0.0,
            "mani_regularization_support_foot_xy_range_y_abs_max": None,
            # manipulation 势奖励与累计误差。
            # 势奖励内部使用的命令名。
            "mani_potential_command_name": "ee_pose",
            # 势奖励的标准差。
            "mani_potential_std": math.sqrt(0.005),
            # 势奖励放大权重；默认给 1.0，具体任务可在子类里继续放大。
            "mani_potential_weight": 1.0,
            # 累计误差项使用的命令名。
            "mani_cumulative_error_command_name": "ee_pose",
            # 累计误差上限，防止数值无限增大。
            "mani_cumulative_error_clip_max": 20.0,
            "workspace_position_weight": 0.0,
            "workspace_position_command_name": "ee_pose",
            "workspace_position_x_min": 0.25,
            "workspace_position_x_max": 0.35,
            "workspace_position_y_weight": 0.2,
            "workspace_position_std": 0.25,
            "workspace_position_clip_max": 1.0,
            # locomotion 主项。
            # locomotion 子项同样读取 ee_pose 命令，用于和操纵误差共同门控。
            "loco_tracking_command_name": "ee_pose",
            # locomotion 跟踪项标准差。
            "loco_tracking_std": math.sqrt(0.01),
            # locomotion 跟踪项阈值。
            "loco_tracking_threshold": 0.0,
            "loco_tracking_weight": 1.0,
            # 基座高度正则权重。
            "loco_regularization_base_height_weight": 0.0,
            # 基座高度误差标准差。
            "loco_regularization_base_height_std": 0.1,
            # 基座高度目标值。
            "loco_regularization_base_height_target_height": 0.33,
            # 读取基座高度的 body 配置。
            "loco_regularization_base_height_asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_BASE_BODY_NAME),
            # 用于估计基座脚下地形高度的扫描器。
            "loco_regularization_base_height_sensor_cfg": SceneEntityCfg("height_scanner_base"),
            # 基座 roll 正则权重。
            "loco_regularization_base_roll_weight": 0.0,
            # 基座 roll 正则标准差。
            "loco_regularization_base_roll_std": 0.1,
            # 基座 roll 正则对应的 body。
            "loco_regularization_base_roll_asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_BASE_BODY_NAME),
            # 基座 pitch 正则权重。
            "loco_regularization_base_pitch_weight": 0.0,
            # 基座 pitch 正则标准差。
            "loco_regularization_base_pitch_std": 0.1,
            # 基座 pitch 正则对应的 body。
            "loco_regularization_base_pitch_asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_BASE_BODY_NAME),
            # 基座 roll 角速度正则权重。
            "loco_regularization_base_roll_ang_vel_weight": 0.0,
            # 基座 roll 角速度正则标准差。
            "loco_regularization_base_roll_ang_vel_std": 0.2,
            # 基座 roll 角速度正则读取的 body。
            "loco_regularization_base_roll_ang_vel_asset_cfg": SceneEntityCfg(
                "robot", body_names=GO2ARM_BASE_BODY_NAME
            ),
            # 基座 pitch 角速度正则权重。
            "loco_regularization_base_pitch_ang_vel_weight": 0.0,
            # 基座 pitch 角速度正则标准差。
            "loco_regularization_base_pitch_ang_vel_std": 0.2,
            # 基座 pitch 角速度正则读取的 body。
            "loco_regularization_base_pitch_ang_vel_asset_cfg": SceneEntityCfg(
                "robot", body_names=GO2ARM_BASE_BODY_NAME
            ),
            # 基座 z 方向速度正则权重。
            "loco_regularization_base_z_vel_weight": 0.0,
            # 基座 z 方向速度正则标准差。
            "loco_regularization_base_z_vel_std": 0.2,
            # 基座 z 方向速度正则读取的 body。
            "loco_regularization_base_z_vel_asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_BASE_BODY_NAME),
            # 基座 body frame 侧向速度正则权重，用于抑制“身体不朝前、而是斜着走/横着漂”。
            "loco_regularization_base_lateral_vel_weight": 0.0,
            # 基座 body frame 侧向速度正则标准差。
            "loco_regularization_base_lateral_vel_std": 0.2,
            # 基座 body frame 侧向速度正则读取的 body。
            "loco_regularization_base_lateral_vel_asset_cfg": SceneEntityCfg("robot", body_names=GO2ARM_BASE_BODY_NAME),
            "loco_regularization_leg_posture_deviation_weight": 0.0,
            "loco_regularization_leg_posture_deviation_std": math.sqrt(4.0 * (1.0 + 0.7 + 0.4)) * 0.25,
            "loco_regularization_leg_posture_deviation_asset_cfg": SceneEntityCfg(
                "robot", joint_names=GO2ARM_LEG_JOINT_NAMES, preserve_order=True
            ),
            "loco_regularization_leg_posture_deviation_joint_weights": None,
            # 相对 reset 默认站姿的前后摆动对称正则，用 phase-aware 的方式约束左右腿 x 向位移关系。
            "loco_regularization_touchdown_left_right_x_symmetry_weight": 0.0,
            "loco_regularization_touchdown_left_right_x_symmetry_std": 0.05,
            "loco_regularization_touchdown_left_right_y_symmetry_weight": 0.0,
            "loco_regularization_touchdown_left_right_y_symmetry_std": 0.05,
            "loco_regularization_touchdown_foot_y_distance_weight": 0.0,
            "loco_regularization_touchdown_foot_y_distance_std": 0.03,
            "loco_regularization_touchdown_foot_y_distance_min_distance": 0.12,
            "loco_regularization_touchdown_foot_y_distance_max_distance": None,
            # 软 trot 足端接触规律正则权重。
            "loco_regularization_feet_contact_soft_trot_weight": 0.0,
            # soft trot 正则读取的足端接触传感器。
            "loco_regularization_feet_contact_soft_trot_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=GO2ARM_FOOT_BODY_NAMES
            ),
            # soft trot 正则读取的足端刚体。
            "loco_regularization_feet_contact_soft_trot_asset_cfg": SceneEntityCfg(
                "robot", body_names=GO2ARM_FOOT_BODY_NAMES
            ),
            # soft trot 中接触力误差的标准差。
            "loco_regularization_feet_contact_soft_trot_force_std": 1.0,
            # soft trot 中足端高度误差的标准差。
            "loco_regularization_feet_contact_soft_trot_height_std": 0.05,
            # soft trot 中足端速度误差的标准差。
            "loco_regularization_feet_contact_soft_trot_vel_std": 0.01,
            # soft trot 参考步态周期。
            "loco_regularization_feet_contact_soft_trot_cycle_time": 0.5,
            # soft trot 四足相位偏置。
            "loco_regularization_feet_contact_soft_trot_phase_offsets": GO2ARM_TROT_PHASE_OFFSETS,
            # soft trot 摆腿阶段参考抬脚高度。
            "loco_regularization_feet_contact_soft_trot_swing_height": 0.08,
            # soft contact 函数的平滑系数。
            "loco_regularization_feet_contact_soft_trot_soft_contact_k": 10.0,
            # 认为“接触成立”的接触力阈值。
            "loco_regularization_feet_contact_soft_trot_contact_force_threshold": 1.0,
            # soft trot 判断地面高度所用的四个足端扫描器。
            "loco_regularization_feet_contact_soft_trot_ground_sensor_names": GO2ARM_FOOT_SCANNER_NAMES,
            # 机械臂摆动正则权重，用于约束 locomotion 阶段的 arm 动作幅度。
            "loco_arm_swing_weight": 0.0,
            # arm swing 正则读取的机械臂关节集合。
            "loco_arm_swing_asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ARM_JOINT_NAMES),
            # 机械臂动态摆动正则权重，用于抑制 locomotion 阶段通过肘关节等快速摆动来补偿机身。
            "loco_arm_dynamic_weight": 0.0,
            # arm dynamic 正则读取的机械臂关节集合。
            "loco_arm_dynamic_asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ARM_JOINT_NAMES),
            # basic 奖励按加权求和。
            # 存活奖励权重。
            "basic_is_alive_weight": 0.0,
            "basic_termination_penalty_weight": 0.0,
            "basic_termination_penalty_excluded_terms": ("task_success",),
            # 碰撞惩罚总权重。
            "basic_collision_weight": 0.0,
            # 碰撞噪声过滤阈值。
            "basic_collision_threshold": 1.0,
            # 碰撞检测传感器；这里同样排除当前被视为合法支撑端的四个 calf。
            "basic_collision_sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2ARM_NON_FOOT_BODY_REGEX),
            # 碰撞计数项权重。
            "basic_collision_count_weight": 1.0,
            # 碰撞超阈值作用力项权重。
            "basic_collision_force_weight": 1.0,
            # 碰撞超阈值作用力项的归一化尺度。
            "basic_collision_force_scale": 10.0,
            # 一阶动作平滑惩罚权重。
            "basic_action_smoothness_first_weight": 0.0,
            # 二阶动作平滑惩罚权重。
            "basic_action_smoothness_second_weight": 0.0,
            # 关节力矩平方和惩罚权重。
            "basic_joint_torque_sq_weight": 0.0,
            # 关节力矩平方和惩罚读取的关节集合。
            "basic_joint_torque_sq_asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ALL_JOINT_NAMES),
            "basic_joint_torque_sq_normalize_by_effort_limit": False,
            # 关节功率惩罚权重。
            "basic_joint_power_weight": 0.0,
            # 关节功率惩罚读取的关节集合。
            "basic_joint_power_asset_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ALL_JOINT_NAMES),
            "basic_joint_power_normalize_by_effort_limit": False,
        },
    )
    # 保留有状态势奖励项，供 total_reward 复用上一时刻缓存。
    ee_tracking_potential = RewTerm(
        # 有状态势奖励类本体。
        func=cast(Callable[..., torch.Tensor], mdp.EETrackingPotentialReward),
        # 这项主要作为 total_reward 内部的势奖励缓存使用。
        # 这里不能设成 0.0，否则 RewardManager 不会真正执行该 term，内部 current/prev/last_reward 都不会更新。
        # 使用极小权重仅用于“激活实例并更新缓存”，对最终总奖励数值影响可以忽略。
        weight=1.0e-6,
        params={
            # 势奖励读取的命令项。
            "command_name": "ee_pose",
            # 倒数型势函数系数：phi = clip(gain / error, clip_min, clip_max)。
            "gain": 0.3,
            # clip 下界。
            "clip_min": 0.0,
            # clip 上界，避免误差过小时势函数过大。
            "clip_max": 5.0,
            # 防止除零。
            "eps": 1.0e-6,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the go2arm MDP."""

    # episode 超时终止。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 机器人跑出地形边界时终止。
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # 非足端碰撞终止：soft 连续计数，hard 立即结束。
    non_foot_contact_termination = DoneTerm(
        func=mdp.contact_termination,
        params={
            # 终止条件里的非法接触过滤也必须和 reward 的合法支撑端定义保持一致。
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=GO2ARM_NON_FOOT_BODY_REGEX),
            "soft_force_threshold": 1.0,
            "hard_force_threshold": 5.0,
            "consecutive_steps": 3,
        },
    )
    # 基座姿态终止：roll/pitch 过大时触发。
    base_orientation_termination = DoneTerm(
        func=mdp.base_orientation_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
            "soft_roll_pitch_limit": 0.4,
            "hard_roll_pitch_limit": 0.5,
            "consecutive_steps": 5,
        },
    )
    # 基座高度终止：过低则认为跌倒或趴地。
    base_height_termination = DoneTerm(
        func=mdp.base_height_termination,
        params={
            "soft_minimum_height": 0.2,
            "hard_minimum_height": 0.16,
            "asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "consecutive_steps": 5,
        },
    )
    # 关节位置严重越界终止。
    joint_position_termination = DoneTerm(
        func=mdp.joint_position_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "soft_max_violation": 0.10,
            "hard_max_violation": 0.30,
            "consecutive_steps": 3,
        },
    )
    # 关节速度严重越界终止。
    joint_velocity_termination = DoneTerm(
        func=mdp.joint_velocity_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "soft_ratio": 1.0,
            "soft_max_violation": 0.50,
            "hard_max_violation": 2.0,
            "consecutive_steps": 3,
        },
    )
    # 关节力矩严重越界终止。
    joint_torque_termination = DoneTerm(
        func=mdp.joint_torque_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # Legacy absolute clipping sum: sum(abs(computed_torque - applied_torque)).
            "soft_max_violation": 3.0,
            "hard_max_violation": 10.0,
            "consecutive_steps": 8,
        },
    )
    # 成功终止：基座和机械臂速度都很小，且末端跟踪误差足够小。
    task_success = DoneTerm(
        func=mdp.task_success_termination,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=[GO2ARM_BASE_BODY_NAME]),
            "arm_joint_cfg": SceneEntityCfg("robot", joint_names=GO2ARM_ARM_JOINT_NAMES, preserve_order=True),
            "base_lin_vel_threshold": 0.05,
            "base_ang_vel_threshold": 0.10,
            "arm_joint_vel_threshold": 0.05,
            "ee_tracking_error_threshold": 0.05,
            "consecutive_steps": 3,
        },
    )


@configclass
class CurriculumCfg:
    """go2arm 任务的课程配置。"""

    # 课程在具体 go2arm env 中注册；基类里先留空。
    go2arm_reaching_stages = None


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """go2arm 任务共享的环境主配置。"""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 控制频率相关设置：每 8 个物理步执行一次策略。
        self.decimation = 8
        # 单个 episode 的时长。
        self.episode_length_s = 20.0
        # 物理仿真步长，当前为 400Hz。
        self.sim.dt = 0.0025
        # 渲染间隔与控制间隔保持一致。
        self.sim.render_interval = self.decimation
        # 直接复用场景地形材质作为仿真默认材质。
        self.sim.physics_material = self.scene.terrain.physics_material
        # 提高 GPU 侧 rigid patch 上限，避免大规模并行时溢出。
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # 传感器更新周期与当前控制/物理频率对齐。
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # 当前 go2arm 基线任务不启用 terrain curriculum。
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """把权重为 0 的奖励项直接置空，避免无效项继续参与 manager 注册。"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)

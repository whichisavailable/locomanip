# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# Derived from go2_piper_description_mjc_NoGripper.urdf:
# - each foot collision sphere has origin xyz="-0.002 0 0" in the corresponding *_foot body frame
GO2ARM_FOOT_SPHERE_CENTER_OFFSET_B = (
    (-0.002, 0.0, 0.0),
    (-0.002, 0.0, 0.0),
    (-0.002, 0.0, 0.0),
    (-0.002, 0.0, 0.0),
)
GO2ARM_FOOT_SPHERE_RADIUS = 0.022
GO2ARM_FOOT_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GO2ARM_FOOT_SENSOR_NAMES = ("FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact")
_GO2ARM_FOOT_OFFSETS_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_GO2ARM_PHASE_OFFSETS_CACHE: dict[tuple[torch.device, torch.dtype, tuple[float, ...]], torch.Tensor] = {}


def _go2arm_foot_offsets(device: torch.device, dtype: torch.dtype, num_feet: int) -> torch.Tensor:
    if num_feet != len(GO2ARM_FOOT_SPHERE_CENTER_OFFSET_B):
        raise ValueError(f"Expected 4 Go2Arm feet, but got {num_feet}.")
    cache_key = (device, dtype, num_feet)
    if cache_key not in _GO2ARM_FOOT_OFFSETS_CACHE:
        _GO2ARM_FOOT_OFFSETS_CACHE[cache_key] = torch.tensor(
            GO2ARM_FOOT_SPHERE_CENTER_OFFSET_B, device=device, dtype=dtype
        )
    return _GO2ARM_FOOT_OFFSETS_CACHE[cache_key]


def _go2arm_phase_offsets(device: torch.device, dtype: torch.dtype, phase_offsets: tuple[float, ...]) -> torch.Tensor:
    cache_key = (device, dtype, phase_offsets)
    if cache_key not in _GO2ARM_PHASE_OFFSETS_CACHE:
        _GO2ARM_PHASE_OFFSETS_CACHE[cache_key] = torch.tensor(phase_offsets, device=device, dtype=dtype).unsqueeze(0)
    return _GO2ARM_PHASE_OFFSETS_CACHE[cache_key]


def get_go2arm_foot_sphere_centers_from_bodies(asset: Articulation, body_ids: list[int] | torch.Tensor) -> torch.Tensor:
    """Return foot collision-sphere centers from the four foot rigid bodies."""
    if isinstance(body_ids, torch.Tensor):
        body_ids = body_ids.tolist()
    body_names = tuple(asset.body_names[int(body_id)] for body_id in body_ids)
    if body_names != GO2ARM_FOOT_BODY_NAMES:
        raise ValueError(f"Unsupported Go2Arm body layout for foot centers: {body_names}")
    foot_offsets_b = _go2arm_foot_offsets(asset.device, asset.data.body_pos_w.dtype, len(body_ids))
    foot_pos_w = asset.data.body_pos_w[:, body_ids, :]
    foot_quat_w = asset.data.body_quat_w[:, body_ids, :]
    foot_offsets_w = quat_apply(
        foot_quat_w.reshape(-1, 4),
        foot_offsets_b.unsqueeze(0).expand(asset.num_instances, -1, -1).reshape(-1, 3),
    ).reshape(asset.num_instances, len(body_ids), 3)
    return foot_pos_w + foot_offsets_w


def get_go2arm_foot_center_linear_velocities_from_bodies(
    asset: Articulation, body_ids: list[int] | torch.Tensor
) -> torch.Tensor:
    """Return foot collision-sphere center linear velocities from the four foot rigid bodies."""
    if isinstance(body_ids, torch.Tensor):
        body_ids = body_ids.tolist()
    body_names = tuple(asset.body_names[int(body_id)] for body_id in body_ids)
    if body_names != GO2ARM_FOOT_BODY_NAMES:
        raise ValueError(f"Unsupported Go2Arm body layout for foot velocities: {body_names}")
    foot_offsets_b = _go2arm_foot_offsets(asset.device, asset.data.body_lin_vel_w.dtype, len(body_ids))
    foot_quat_w = asset.data.body_quat_w[:, body_ids, :]
    foot_lin_vel_w = asset.data.body_lin_vel_w[:, body_ids, :]
    foot_ang_vel_w = asset.data.body_ang_vel_w[:, body_ids, :]
    foot_offsets_w = quat_apply(
        foot_quat_w.reshape(-1, 4),
        foot_offsets_b.unsqueeze(0).expand(asset.num_instances, -1, -1).reshape(-1, 3),
    ).reshape(asset.num_instances, len(body_ids), 3)
    return foot_lin_vel_w + torch.cross(foot_ang_vel_w, foot_offsets_w, dim=-1)


def _get_go2arm_foot_kinematics(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> dict[str, torch.Tensor]:
    """Cache foot-sphere centers and center velocities for the current step."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        raise ValueError("Go2Arm foot kinematics require explicit foot body_ids.")
    cache_key = (getattr(env, "common_step_counter", -1), asset_cfg.name, tuple(int(body_id) for body_id in body_ids))
    cached_data = getattr(env, "_go2arm_foot_kinematics_cache", None)
    if cached_data is not None and cached_data.get("key") == cache_key:
        return cached_data["value"]

    kinematics = {
        "foot_sphere_centers_w": get_go2arm_foot_sphere_centers_from_bodies(asset, body_ids),
        "foot_center_lin_vel_w": get_go2arm_foot_center_linear_velocities_from_bodies(asset, body_ids),
    }
    env._go2arm_foot_kinematics_cache = {"key": cache_key, "value": kinematics}
    return kinematics


def _get_go2arm_ground_height_data(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Cache per-foot scanner height means and validity masks for the current step."""
    cache_key = (getattr(env, "common_step_counter", -1), sensor_names)
    cached_data = getattr(env, "_go2arm_ground_height_cache", None)
    if cached_data is not None and cached_data.get("key") == cache_key:
        return cached_data["value"]

    ground_height_values: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    for sensor_name in sensor_names:
        sensor: RayCaster = env.scene.sensors[sensor_name]
        ray_hits_z = sensor.data.ray_hits_w[..., 2]
        ground_height_values.append(torch.mean(ray_hits_z, dim=1))
        valid_masks.append(
            ~(
                torch.isnan(ray_hits_z).any(dim=1)
                | torch.isinf(ray_hits_z).any(dim=1)
                | (torch.max(torch.abs(ray_hits_z), dim=1)[0] > 1e6)
            )
        )

    ground_height_data = {
        "ground_height_w": torch.stack(ground_height_values, dim=1),
        "is_valid": torch.stack(valid_masks, dim=1),
    }
    env._go2arm_ground_height_cache = {"key": cache_key, "value": ground_height_data}
    return ground_height_data


def _use_go2arm_precise_contact(env: ManagerBasedEnv) -> bool:
    """Whether the current go2arm env exposes the required per-foot sensors."""
    return bool(getattr(env, "_go2arm_has_foot_sensors", False))


def _get_go2arm_precise_foot_sensor_data(env: ManagerBasedEnv) -> dict[str, torch.Tensor] | None:
    """Read precise foot contact directly from the four dedicated foot sensors."""
    if not _use_go2arm_precise_contact(env):
        return None

    cache_key = (getattr(env, "common_step_counter", -1), "go2arm_precise_foot_sensor_data")
    cached_data = getattr(env, "_go2arm_precise_foot_sensor_cache", None)
    if cached_data is not None and cached_data.get("key") == cache_key:
        return cached_data["value"]

    foot_contact_sensors = getattr(env, "_go2arm_foot_contact_sensors", None)
    if foot_contact_sensors is None:
        if any(sensor_name not in env.scene.sensors for sensor_name in GO2ARM_FOOT_SENSOR_NAMES):
            return None
        foot_contact_sensors = tuple(env.scene.sensors[sensor_name] for sensor_name in GO2ARM_FOOT_SENSOR_NAMES)
        env._go2arm_foot_contact_sensors = foot_contact_sensors

    asset: Articulation = env.scene["robot"]
    dtype = asset.data.body_pos_w.dtype
    force_vectors_per_foot: list[torch.Tensor] = []
    air_time_per_foot: list[torch.Tensor] = []
    contact_time_per_foot: list[torch.Tensor] = []

    for foot_contact_sensor in foot_contact_sensors:
        if foot_contact_sensor.data.force_matrix_w is not None:
            force_vectors = torch.sum(foot_contact_sensor.data.force_matrix_w[:, 0, :, :], dim=1)
        else:
            force_vectors = foot_contact_sensor.data.net_forces_w[:, 0, :]
        force_vectors_per_foot.append(force_vectors.to(dtype))
        air_time_per_foot.append(foot_contact_sensor.data.current_air_time[:, 0].to(dtype))
        contact_time_per_foot.append(foot_contact_sensor.data.current_contact_time[:, 0].to(dtype))

    precise_foot_force_vectors = torch.stack(force_vectors_per_foot, dim=1)
    precise_foot_normal_forces = torch.abs(precise_foot_force_vectors[..., 2])
    foot_sensor_data = {
        "foot_force_vectors_w": precise_foot_force_vectors,
        "foot_normal_forces": precise_foot_normal_forces,
        "current_air_time": torch.stack(air_time_per_foot, dim=1),
        "current_contact_time": torch.stack(contact_time_per_foot, dim=1),
    }
    env._go2arm_precise_foot_sensor_cache = {"key": cache_key, "value": foot_sensor_data}
    return foot_sensor_data


def get_go2arm_precise_foot_contact_forces(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor | None:
    """Return precise legal foot contact forces in world frame for go2arm."""
    del sensor_cfg
    del asset_cfg
    contact_data = _get_go2arm_precise_foot_sensor_data(env)
    if contact_data is None:
        return None
    return contact_data["foot_force_vectors_w"]


def get_go2arm_precise_foot_normal_forces(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor | None:
    """Return precise legal foot-pad contact-force magnitudes for go2arm."""
    del sensor_cfg
    del asset_cfg
    contact_data = _get_go2arm_precise_foot_sensor_data(env)
    if contact_data is None:
        return None
    return contact_data["foot_normal_forces"]


def get_go2arm_precise_foot_contact_timers(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> dict[str, torch.Tensor] | None:
    """Return precise go2arm foot contact state and per-foot air/contact timers."""
    del sensor_cfg
    del asset_cfg
    foot_sensor_data = _get_go2arm_precise_foot_sensor_data(env)
    if foot_sensor_data is None:
        return None
    in_contact = foot_sensor_data["foot_normal_forces"] > threshold
    timer_data = {
        "in_contact": in_contact,
        "current_air_time": foot_sensor_data["current_air_time"],
        "current_contact_time": foot_sensor_data["current_contact_time"],
    }
    return timer_data


def feet_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """返回每个足端是否接触地面的二值状态。"""

    precise_normal_forces = get_go2arm_precise_foot_normal_forces(env, sensor_cfg=sensor_cfg)
    if precise_normal_forces is not None:
        # go2arm 下优先使用“合法足端 patch 聚合后的精确法向力”来判断是否触地。
        return (precise_normal_forces > threshold).float()

    # 其它机器人仍然沿用 body 级接触力阈值判定。
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.norm(net_forces, dim=-1) > threshold).float()


def _get_external_wrench_b(asset: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-effort accessor for buffered external wrenches across IsaacLab versions."""
    if hasattr(asset, "permanent_wrench_composer"):
        composer = asset.permanent_wrench_composer
        force = getattr(composer, "composed_force_as_torch", None)
        torque = getattr(composer, "composed_torque_as_torch", None)
        if force is not None and torque is not None:
            return (
                force.view(asset.num_instances, asset.num_bodies, 3).to(asset.device),
                torque.view(asset.num_instances, asset.num_bodies, 3).to(asset.device),
            )

    force = getattr(asset, "_external_force_b", None)
    torque = getattr(asset, "_external_torque_b", None)
    if force is not None and torque is not None:
        return force.to(asset.device), torque.to(asset.device)

    zeros = torch.zeros(
        (asset.num_instances, asset.num_bodies, 3), device=asset.device, dtype=asset.data.root_pos_w.dtype
    )
    return zeros, zeros


def feet_air_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回每个足端当前已经连续离地的时间。"""

    precise_timer_data = get_go2arm_precise_foot_contact_timers(env, sensor_cfg=sensor_cfg)
    if precise_timer_data is not None:
        # go2arm 下优先返回按“合法足端 patch 接触状态”更新的精确 air time。
        return precise_timer_data["current_air_time"]

    # 读取接触传感器。
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # current_air_time 的形状为 (num_envs, num_bodies)。
    return contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]


def static_friction(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回指定刚体对应的静摩擦系数。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]
    # PhysX 材料属性的最后一维依次为静摩擦、动摩擦和回弹系数。
    materials = asset.root_physx_view.get_material_properties()

    # 取出需要查询的 body 索引。
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(*body_ids.indices(asset.num_bodies)))

    # PhysX 的材料是按 shape 存的，不是按 body 存的。
    # 这里先统计每个 body 有多少个 shape，用于把 body_id 映射到 shape_id。
    num_shapes_per_body = []
    for link_path in asset.root_physx_view.link_paths[0]:
        link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)
        num_shapes_per_body.append(link_physx_view.max_shapes)

    # 这里约定读取每个 body 的第一个 shape 的静摩擦系数。
    static_frictions = []
    for body_id in body_ids:
        shape_idx = sum(num_shapes_per_body[:body_id])
        static_frictions.append(materials[:, shape_idx, 0])

    # 拼成 (num_envs, num_bodies)。
    return torch.stack(static_frictions, dim=1).to(asset.device)


def base_external_wrench(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回基座上的外力和外力矩，表达在 body frame。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]
    # 如果没有显式指定 body，则默认取第一个刚体作为基座。
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = [0]

    external_force_b, external_torque_b = _get_external_wrench_b(asset)
    external_force = external_force_b[:, body_ids, :].flatten(1)
    external_torque = external_torque_b[:, body_ids, :].flatten(1)
    return torch.cat([external_force, external_torque], dim=-1)


def base_external_push_velocity(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回最近一次基座速度扰动，并转换到 base frame。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]

    # 该缓存由 push event 写入；如果本步没有 push，则直接返回 0。
    delta_w = getattr(env, "_last_push_delta_w", None)
    if delta_w is None or getattr(env, "_last_push_step", None) != env._sim_step_counter:
        return torch.zeros((env.scene.num_envs, 6), device=asset.device, dtype=asset.data.root_vel_w.dtype)

    # 确保缓存张量和当前设备一致。
    if delta_w.device != asset.device:
        delta_w = delta_w.to(asset.device)
        setattr(env, "_last_push_delta_w", delta_w)

    # 速度扰动最开始记录在世界系下，这里旋转到 base frame。
    root_quat_w = asset.data.root_quat_w
    lin_b = quat_apply_inverse(root_quat_w, delta_w[:, :3])
    ang_b = quat_apply_inverse(root_quat_w, delta_w[:, 3:6])
    return torch.cat([lin_b, ang_b], dim=-1)


def base_mass_disturbance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回基座质量扰动，即当前质量减去默认质量。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]
    # 如果没有显式指定 body，则默认第一个刚体是基座。
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = [0]
    body_id = body_ids[0] if isinstance(body_ids, list) else body_ids

    # 当前生效质量。
    current_mass = asset.root_physx_view.get_masses()[:, body_id].to(asset.device)
    # 首次调用时缓存默认质量，后续直接做差。
    cache_name = f"_default_mass_{asset_cfg.name}_{body_id}"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, asset.data.default_mass[:, body_id].clone().to(asset.device))
    default_mass = getattr(env, cache_name)
    if default_mass.device != asset.device:
        default_mass = default_mass.to(asset.device)
        setattr(env, cache_name, default_mass)

    return (current_mass - default_mass).unsqueeze(-1)


def ee_external_wrench(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回末端执行器上的外力和外力矩，表达在 body frame。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]
    # 末端必须显式指定 body 名称，不能直接使用默认 slice。
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        raise ValueError("ee_external_wrench requires explicit body_names for the end-effector")

    external_force_b, external_torque_b = _get_external_wrench_b(asset)
    external_force = external_force_b[:, body_ids, :].flatten(1)
    external_torque = external_torque_b[:, body_ids, :].flatten(1)
    return torch.cat([external_force, external_torque], dim=-1)


def ee_mass_disturbance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回末端执行器的质量扰动。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]
    # 末端必须显式指定 body 名称。
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        raise ValueError("ee_mass_disturbance requires explicit body_names for the end-effector")
    body_id = body_ids[0] if isinstance(body_ids, list) else body_ids

    # 当前生效质量。
    current_mass = asset.root_physx_view.get_masses()[:, body_id].to(asset.device)
    # 首次调用时缓存默认质量。
    cache_name = f"_default_mass_ee_{asset_cfg.name}_{body_id}"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, asset.data.default_mass[:, body_id].clone().to(asset.device))
    default_mass = getattr(env, cache_name)
    if default_mass.device != asset.device:
        default_mass = default_mass.to(asset.device)
        setattr(env, cache_name, default_mass)

    return (current_mass - default_mass).unsqueeze(-1)


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """返回相对默认位姿的关节位置，并把轮关节清零。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """返回基础相位编码 [sin, cos]。"""

    # 某些分析脚本里环境可能还没初始化该缓存，这里做一次兜底。
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # 相位定义为当前 episode 时间除以步态周期。
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    return torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)


def end_effector_pose_b(
    env: ManagerBasedEnv,
    ee_body_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回末端执行器在 base frame 下的位姿，格式为 3 维位置加 9 维旋转矩阵。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[ee_body_cfg.name]
    # 读取基座和末端在世界系下的位姿。
    base_pos_w = asset.data.body_pos_w[:, base_body_cfg.body_ids[0]]
    base_quat_w = asset.data.body_quat_w[:, base_body_cfg.body_ids[0]]
    ee_pos_w = asset.data.body_pos_w[:, ee_body_cfg.body_ids[0]]
    ee_quat_w = asset.data.body_quat_w[:, ee_body_cfg.body_ids[0]]

    # 把末端位姿变换到 base frame。
    ee_pos_b, ee_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
    # 把四元数转成旋转矩阵，并展平为 9 维。
    ee_rotmat_b = matrix_from_quat(ee_quat_b)
    return torch.cat([ee_pos_b, ee_rotmat_b.reshape(env.num_envs, -1)], dim=-1)


def command_term_reference_tracking_error(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """读取命令项内部维护的参考跟踪误差。"""

    # 这里读取的是命令项对象本身，而不是 12 维命令张量。
    command_term = env.command_manager.get_term(command_name)
    return command_term.reference_tracking_error.unsqueeze(-1)


def command_term_cumulative_tracking_error(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """读取命令项内部维护的累积跟踪误差。"""

    # 这里和参考误差读取的是同一个命令项对象，只是字段不同。
    command_term = env.command_manager.get_term(command_name)
    return command_term.cumulative_tracking_error.unsqueeze(-1)


def trot_phase_sin(
    env: ManagerBasedRLEnv, cycle_time: float, phase_offsets: tuple[float, float, float, float]
) -> torch.Tensor:
    """返回四个足端的 trot 相位正弦信号。"""

    # 如果环境还没准备好 episode 计数缓存，这里做一次兜底初始化。
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # 当前相位使用 episode 内经过的时间除以步态周期得到。
    phase_t = env.episode_length_buf.to(torch.float32) * env.step_dt / cycle_time
    # 每条腿各自叠加一个固定的 trot 相位偏置。
    offsets = _go2arm_phase_offsets(env.device, torch.float32, tuple(phase_offsets))
    return torch.sin(2.0 * torch.pi * (phase_t.unsqueeze(-1) + offsets))


def base_height_from_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回基座相对当前地面的高度。"""

    # 读取基座高度扫描器。
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # 对所有射线的击中点高度取平均，得到当前基座下方的局部地形高度。
    ground_height_w = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
    # 传感器自身高度减去地形高度，得到基座离地高度。
    return sensor.data.pos_w[:, 2].unsqueeze(-1) - ground_height_w


def base_height_on_plane(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return exact base height above a flat plane at z=0 without ray casting."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2].unsqueeze(-1)


def foot_heights_from_scanners(
    env: ManagerBasedEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回每个足端相对地面的高度。"""

    asset: Articulation = env.scene[asset_cfg.name]
    del asset
    foot_sphere_centers_w = _get_go2arm_foot_kinematics(env, asset_cfg)["foot_sphere_centers_w"]

    # 逐个足端扫描器读取地面高度，顺序与传入的 sensor_names 一致。
    ground_height_data = _get_go2arm_ground_height_data(env, tuple(sensor_names))
    contact_point_height_w = foot_sphere_centers_w[..., 2] - GO2ARM_FOOT_SPHERE_RADIUS
    effective_ground_height_w = torch.where(
        ground_height_data["is_valid"],
        ground_height_data["ground_height_w"],
        contact_point_height_w,
    )
    return contact_point_height_w - effective_ground_height_w


def foot_heights_on_plane(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return exact foot contact-point heights above a flat plane at z=0."""
    foot_sphere_centers_w = _get_go2arm_foot_kinematics(env, asset_cfg)["foot_sphere_centers_w"]
    return foot_sphere_centers_w[..., 2] - GO2ARM_FOOT_SPHERE_RADIUS


def feet_contact_forces(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回四个足端的三维接触力，展平后为 12 维。"""

    precise_forces = get_go2arm_precise_foot_contact_forces(env, sensor_cfg=sensor_cfg)
    if precise_forces is not None:
        # go2arm 下返回的是“只统计合法足端 patch 后”的精确足端接触力。
        return precise_forces.reshape(env.num_envs, -1)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return net_forces.reshape(env.num_envs, -1)


def body_velocity_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回指定刚体在 base frame 下的线速度和角速度。"""

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]
    # 用基座姿态把世界系速度旋转到 base frame。
    base_quat_w = asset.data.body_quat_w[:, base_body_cfg.body_ids[0]]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0]]
    body_lin_vel_b = quat_apply_inverse(base_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(base_quat_w, body_ang_vel_w)
    return torch.cat([body_lin_vel_b, body_ang_vel_b], dim=-1)


def feet_planar_velocities_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """返回四个足端在世界坐标系下的平面速度，每只脚保留 (vx, vy)。"""

    # 当前 teacher 任务里，这个特权量按世界系水平速度使用。
    del base_body_cfg

    # 读取机器人资产。
    asset: Articulation = env.scene[asset_cfg.name]
    del asset
    # 用 foot 刚体和球心局部偏移恢复足端碰撞球心的世界系线速度。
    feet_lin_vel_w = _get_go2arm_foot_kinematics(env, asset_cfg)["foot_center_lin_vel_w"]
    num_envs = feet_lin_vel_w.shape[0]
    # 只保留每只脚的世界系平面速度 (vx, vy)，四只脚一共 8 维。
    return feet_lin_vel_w[..., :2].reshape(num_envs, -1)


def observation_delay(
    env: ManagerBasedEnv,
    attr_name: str = "_observation_delay",
) -> torch.Tensor:
    """返回观测延迟接口值，当前默认不启用时为 0。"""

    # 如果环境对象上还没有这个属性，说明当前没有启用观测延迟。
    delay_value = getattr(env, attr_name, None)
    if delay_value is None:
        return torch.zeros((env.num_envs, 1), device=env.device)

    # 如果是 Python 标量，则扩成每个并行环境各一份。
    if not torch.is_tensor(delay_value):
        return torch.full((env.num_envs, 1), float(delay_value), device=env.device)

    # 如果已经是张量，则规范成 (num_envs, 1)。
    delay_value = delay_value.to(env.device)
    if delay_value.ndim == 1:
        delay_value = delay_value.unsqueeze(-1)
    return delay_value

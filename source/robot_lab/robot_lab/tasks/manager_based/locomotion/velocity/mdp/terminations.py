# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp


def _persistent_violation(
    env,
    violation_mask: torch.Tensor,
    counter_name: str,
    consecutive_steps: int,
) -> torch.Tensor:
    """对每个并行环境维护连续违规计数，并在达到阈值时返回终止标志。"""

    if not hasattr(env, counter_name):
        setattr(env, counter_name, torch.zeros(env.num_envs, dtype=torch.int32, device=env.device))

    counter = getattr(env, counter_name)
    if hasattr(env, "episode_length_buf"):
        # Reset per-slot history at the start of a new episode. Without this, a slot that
        # terminated after N soft violations can immediately terminate again after reset.
        new_episode_mask = env.episode_length_buf <= 1
        counter = torch.where(new_episode_mask, torch.zeros_like(counter), counter)
    counter = torch.where(violation_mask, counter + 1, torch.zeros_like(counter))
    setattr(env, counter_name, counter)
    return counter >= consecutive_steps


def _root_height_over_terrain(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None,
) -> torch.Tensor:
    """计算机身相对地面的高度。"""

    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None:
        root_height = asset.data.root_pos_w[:, 2]
    else:
        try:
            if len(body_ids) == 0:
                root_height = asset.data.root_pos_w[:, 2]
            else:
                root_height = asset.data.body_pos_w[:, body_ids, 2]
        except TypeError:
            root_height = asset.data.body_pos_w[:, body_ids, 2]
        if root_height.ndim > 1:
            root_height = torch.mean(root_height, dim=1)

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if not (torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6):
            root_height = root_height - torch.mean(ray_hits, dim=1)

    return root_height


def contact_termination(
    env,
    sensor_cfg: SceneEntityCfg,
    soft_force_threshold: float,
    hard_force_threshold: float,
    consecutive_steps: int = 3,
) -> torch.Tensor:
    """非法碰撞终止：hard 立即终止，soft 连续多步终止。"""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    max_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
    hard_violation = torch.any(max_force > hard_force_threshold, dim=1)
    soft_violation = torch.any(max_force > soft_force_threshold, dim=1)
    return hard_violation | _persistent_violation(
        env=env,
        violation_mask=soft_violation,
        counter_name="_termination_contact_soft_counter",
        consecutive_steps=consecutive_steps,
    )


def base_orientation_termination(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_roll_pitch_limit: float = 0.75,
    hard_roll_pitch_limit: float = 1.10,
    consecutive_steps: int = 3,
) -> torch.Tensor:
    """姿态终止：根据 base frame 下的重力投影恢复 roll/pitch。"""

    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    roll = torch.atan2(gravity_b[:, 1], -gravity_b[:, 2])
    pitch = torch.atan2(-gravity_b[:, 0], torch.sqrt(gravity_b[:, 1] ** 2 + gravity_b[:, 2] ** 2))

    hard_violation = (torch.abs(roll) > hard_roll_pitch_limit) | (torch.abs(pitch) > hard_roll_pitch_limit)
    soft_violation = (torch.abs(roll) > soft_roll_pitch_limit) | (torch.abs(pitch) > soft_roll_pitch_limit)
    return hard_violation | _persistent_violation(
        env=env,
        violation_mask=soft_violation,
        counter_name="_termination_orientation_soft_counter",
        consecutive_steps=consecutive_steps,
    )


def base_height_termination(
    env,
    soft_minimum_height: float,
    hard_minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    consecutive_steps: int = 3,
) -> torch.Tensor:
    """基座高度终止：hard 立即终止，soft 连续多步终止。"""

    root_height = _root_height_over_terrain(env=env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    hard_violation = root_height < hard_minimum_height
    soft_violation = root_height < soft_minimum_height
    return hard_violation | _persistent_violation(
        env=env,
        violation_mask=soft_violation,
        counter_name="_termination_height_soft_counter",
        consecutive_steps=consecutive_steps,
    )


def joint_position_termination(
    env,
    soft_max_violation: float,
    hard_max_violation: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    consecutive_steps: int = 3,
) -> torch.Tensor:
    """关节位置越界终止。"""

    violation = mdp.joint_pos_limits(env, asset_cfg=asset_cfg)
    hard_violation = violation > hard_max_violation
    soft_violation = violation > soft_max_violation
    return hard_violation | _persistent_violation(
        env=env,
        violation_mask=soft_violation,
        counter_name="_termination_joint_pos_soft_counter",
        consecutive_steps=consecutive_steps,
    )


def joint_velocity_termination(
    env,
    soft_max_violation: float,
    hard_max_violation: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 1.0,
    consecutive_steps: int = 3,
) -> torch.Tensor:
    """关节速度越界终止。"""

    violation = mdp.joint_vel_limits(env, asset_cfg=asset_cfg, soft_ratio=soft_ratio)
    hard_violation = violation > hard_max_violation
    soft_violation = violation > soft_max_violation
    return hard_violation | _persistent_violation(
        env=env,
        violation_mask=soft_violation,
        counter_name="_termination_joint_vel_soft_counter",
        consecutive_steps=consecutive_steps,
    )


def joint_torque_termination(
    env,
    soft_max_ratio: float | None = None,
    hard_max_ratio: float | None = None,
    soft_max_violation: float | None = None,
    hard_max_violation: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    consecutive_steps: int = 3,
) -> torch.Tensor:
    """Terminate on sustained actuator torque clipping."""

    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    torque_clipping = torch.abs(
        asset.data.computed_torque[:, joint_ids] - asset.data.applied_torque[:, joint_ids]
    )

    if soft_max_ratio is not None:
        torque_dtype = asset.data.computed_torque.dtype
        effort_limits = torch.full_like(asset.data.computed_torque, torch.inf)
        for actuator in asset.actuators.values():
            actuator_joint_ids = torch.as_tensor(actuator.joint_indices, device=env.device, dtype=torch.long)
            actuator_effort_limit = actuator.effort_limit
            if not isinstance(actuator_effort_limit, torch.Tensor):
                actuator_effort_limit = torch.full(
                    (env.num_envs, actuator_joint_ids.numel()),
                    float(actuator_effort_limit),
                    device=env.device,
                    dtype=torque_dtype,
                )
            else:
                actuator_effort_limit = actuator_effort_limit.to(device=env.device, dtype=torque_dtype)
                if actuator_effort_limit.dim() == 0:
                    actuator_effort_limit = actuator_effort_limit.reshape(1, 1).expand(
                        env.num_envs, actuator_joint_ids.numel()
                    )
                elif actuator_effort_limit.dim() == 1:
                    actuator_effort_limit = actuator_effort_limit.unsqueeze(0).expand(env.num_envs, -1)
            effort_limits[:, actuator_joint_ids] = actuator_effort_limit

        selected_effort_limits = torch.clamp(effort_limits[:, joint_ids], min=1.0e-6)
        violation = torch.max(torque_clipping / selected_effort_limits, dim=1)[0]
        soft_threshold = soft_max_ratio
        hard_threshold = hard_max_ratio
    else:
        if soft_max_violation is None:
            raise ValueError("joint_torque_termination requires either soft_max_ratio or soft_max_violation.")
        violation = torch.sum(torque_clipping, dim=1)
        soft_threshold = soft_max_violation
        hard_threshold = hard_max_violation

    hard_violation = (
        violation > hard_threshold if hard_threshold is not None else torch.zeros_like(violation, dtype=torch.bool)
    )
    soft_violation = violation > soft_threshold
    return hard_violation | _persistent_violation(
        env=env,
        violation_mask=soft_violation,
        counter_name="_termination_joint_torque_soft_counter",
        consecutive_steps=consecutive_steps,
    )


def task_success_termination(
    env,
    command_name: str,
    base_lin_vel_threshold: float,
    base_ang_vel_threshold: float,
    arm_joint_vel_threshold: float,
    ee_tracking_error_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_joint_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    consecutive_steps: int = 3,
) -> torch.Tensor:
    """成功终止：机身和机械臂都稳定静止，且末端误差足够小。"""

    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    base_lin_vel_ok = torch.linalg.norm(asset.data.root_lin_vel_b, dim=1) < base_lin_vel_threshold
    base_ang_vel_ok = torch.linalg.norm(asset.data.root_ang_vel_b, dim=1) < base_ang_vel_threshold
    arm_joint_vel_ok = (
        torch.max(torch.abs(asset.data.joint_vel[:, arm_joint_cfg.joint_ids]), dim=1)[0] < arm_joint_vel_threshold
    )
    ee_tracking_ok = command_term.tracking_error < ee_tracking_error_threshold

    success_mask = base_lin_vel_ok & base_ang_vel_ok & arm_joint_vel_ok & ee_tracking_ok
    return _persistent_violation(
        env=env,
        violation_mask=success_mask,
        counter_name="_termination_success_counter",
        consecutive_steps=consecutive_steps,
    )

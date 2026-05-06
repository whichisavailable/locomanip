# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _lerp_value(start: float, end: float, progress: float) -> float:
    """线性插值标量参数。"""
    return start + (end - start) * progress


def _lerp_tuple(start: Sequence[float], end: Sequence[float], progress: float) -> tuple[float, ...]:
    """线性插值 tuple 参数。"""
    return tuple(_lerp_value(float(s), float(e), progress) for s, e in zip(start, end, strict=True))


def _reward_based_progress(env: ManagerBasedRLEnv, reward_term_name: str) -> float:
    """Estimate curriculum progress from logged episodic reward statistics."""
    extras = getattr(env, "extras", {}) or {}
    episode_metrics = extras.get("episode", {}) if isinstance(extras, dict) else {}
    reward_value = episode_metrics.get(f"Reward/{reward_term_name}")
    if reward_value is None:
        return 1.0
    if isinstance(reward_value, torch.Tensor):
        reward_value = float(reward_value.mean().item())
    else:
        reward_value = float(reward_value)
    return max(0.0, min(1.0, reward_value))


def _scale_range(range_values: Sequence[float], scale: float) -> tuple[float, float]:
    """Scale a symmetric command range."""
    return float(range_values[0]) * scale, float(range_values[1]) * scale


def _clamp_progress(step: int, start_step: int, end_step: int) -> float:
    """Compute curriculum progress in [0, 1] from step bounds."""
    if end_step <= start_step:
        return 1.0
    return max(0.0, min(1.0, (float(step) - float(start_step)) / float(end_step - start_step)))


def _frontloaded_progress(step: float, start_step: float, end_step: float, reach_fraction: float) -> float:
    """Reach the target before the interval ends, then keep it fixed."""
    base_progress = _clamp_progress(step, start_step, end_step)
    fraction = max(float(reach_fraction), 1.0e-6)
    return min(base_progress / fraction, 1.0)


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """Progressively expand linear velocity command ranges."""
    del env_ids
    if not hasattr(env.cfg.commands, "base_velocity"):
        return torch.tensor(0.0, device=env.device)

    progress = _reward_based_progress(env, reward_term_name)
    scale = _lerp_value(float(range_multiplier[0]), float(range_multiplier[1]), progress)
    ranges = env.cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = _scale_range(ranges.lin_vel_x, scale)
    ranges.lin_vel_y = _scale_range(ranges.lin_vel_y, scale)
    return torch.tensor(scale, device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z_exp",
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> torch.Tensor:
    """Progressively expand angular velocity command ranges."""
    del env_ids
    if not hasattr(env.cfg.commands, "base_velocity"):
        return torch.tensor(0.0, device=env.device)

    progress = _reward_based_progress(env, reward_term_name)
    scale = _lerp_value(float(range_multiplier[0]), float(range_multiplier[1]), progress)
    ranges = env.cfg.commands.base_velocity.ranges
    ranges.ang_vel_z = _scale_range(ranges.ang_vel_z, scale)
    return torch.tensor(scale, device=env.device)


def go2arm_reaching_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "ee_pose",
    steps_per_iteration: int = 24,
    total_reward_term_name: str = "total_reward",
    loco_stage_end_iteration: int = 0,
    stage1_end_iteration: int = 200,
    stage2_hold_end_iteration: int = 700,
    stage2_expand_end_iteration: int = 1500,
    stage2_ratio_end_iteration: int = 2500,
    stage3_xy_end_iteration: int = 3000,
    stage2_expand_reach_fraction: float = 0.5,
    stage2_ratio_reach_fraction: float = 0.5,
    workspace_position_std_stage1_end_iteration: int = 300,
    workspace_position_std_stage2_end_iteration: int = 700,
    workspace_position_std_stage1: float = 0.5,
    workspace_position_std_stage2: float = 0.25,
    workspace_position_std_stage3: float = 0.1,
    position_range_b_loco_stage: Sequence[float] = (0.70, 1.20, 0.0, 0.0, 0.0, 0.0),
    world_z_range_loco_stage: Sequence[float] = (0.6926649548, 0.6926649548),
    euler_xyz_range_b_loco_stage: Sequence[float] = (0.0, 0.0, 1.5008926535, 1.5008926535, 0.0, 0.0),
    position_range_b_stage1_start: Sequence[float] | None = None,
    position_range_b_stage1: Sequence[float] = (0.10, 0.24, -0.12, 0.12, 0.0, 0.0),
    position_range_b_stage2_allowed_start: Sequence[float] = (0.08, 0.30, -0.14, 0.14, 0.0, 0.0),
    position_range_b_stage3: Sequence[float] = (0.05, 2.00, -0.35, 0.35, 0.0, 0.0),
    world_z_range_stage1_start: Sequence[float] | None = None,
    world_z_range_stage1: Sequence[float] = (0.85, 0.95),
    world_z_range_stage2_allowed_start: Sequence[float] = (0.70, 1.06),
    world_z_range_stage3: Sequence[float] = (0.02, 1.20),
    euler_xyz_range_b_stage1_start: Sequence[float] | None = None,
    euler_xyz_range_b_stage1: Sequence[float] = (-0.50, 0.50, -0.50, 0.50, -3.14159, 3.14159),
    euler_xyz_range_b_stage2_allowed: Sequence[float] = (-0.50, 0.50, -0.50, 0.50, -3.14159, 3.14159),
    euler_xyz_range_b_stage3: Sequence[float] = (-0.50, 0.50, -0.50, 0.50, -3.14159, 3.14159),
    position_range_b_hard_low_start: Sequence[float] = (0.22, 0.38, -0.10, 0.10, 0.0, 0.0),
    position_range_b_hard_low_final: Sequence[float] = (0.22, 0.42, -0.12, 0.12, 0.0, 0.0),
    world_z_range_hard_low_start: Sequence[float] = (0.60, 0.76),
    world_z_range_hard_low_final: Sequence[float] = (0.02, 0.50),
    position_range_b_hard_high_start: Sequence[float] = (0.10, 0.26, -0.10, 0.10, 0.0, 0.0),
    position_range_b_hard_high_final: Sequence[float] = (0.10, 0.30, -0.12, 0.12, 0.0, 0.0),
    world_z_range_hard_high_start: Sequence[float] = (1.00, 1.08),
    world_z_range_hard_high_final: Sequence[float] = (1.02, 1.20),
    euler_xyz_range_b_hard_low: Sequence[float] = (-0.50, 0.50, -0.50, 0.50, -3.14159, 3.14159),
    euler_xyz_range_b_hard_high: Sequence[float] = (-0.50, 0.50, -0.50, 0.50, -3.14159, 3.14159),
    hard_low_sample_prob_stage2_base: float = 0.08,
    hard_high_sample_prob_stage2_base: float = 0.08,
    hard_low_sample_prob_stage2_final: float = 0.22,
    hard_high_sample_prob_stage2_final: float = 0.18,
    reset_joint_position_range_stage1: Sequence[float] = (-0.02, 0.02),
    reset_joint_position_range_stage2: Sequence[float] = (-0.03, 0.03),
    reset_joint_position_range_stage3: Sequence[float] = (-0.05, 0.05),
    reset_joint_velocity_range_stage1: Sequence[float] = (-0.03, 0.03),
    reset_joint_velocity_range_stage2: Sequence[float] = (-0.05, 0.05),
    reset_joint_velocity_range_stage3: Sequence[float] = (-0.08, 0.08),
    reset_root_x_range_stage1: Sequence[float] = (-0.02, 0.02),
    reset_root_x_range_stage2: Sequence[float] = (-0.03, 0.03),
    reset_root_x_range_stage3: Sequence[float] = (-0.06, 0.06),
    reset_root_y_range_stage1: Sequence[float] = (-0.02, 0.02),
    reset_root_y_range_stage2: Sequence[float] = (-0.03, 0.03),
    reset_root_y_range_stage3: Sequence[float] = (-0.06, 0.06),
    reset_root_yaw_range_stage1: Sequence[float] = (-0.06, 0.06),
    reset_root_yaw_range_stage2: Sequence[float] = (-0.10, 0.10),
    reset_root_yaw_range_stage3: Sequence[float] = (-0.18, 0.18),
    base_height_termination_soft_normal: float | None = None,
    base_height_termination_hard_normal: float | None = None,
    base_height_termination_soft_low: float | None = None,
    base_height_termination_hard_low: float | None = None,
) -> torch.Tensor:
    """go2arm staged curriculum: optional loco-only warmup, then manipulation range expansion."""
    del env_ids

    command_term = env.command_manager.get_term(command_name)
    command_cfg = command_term.cfg
    step = int(getattr(env, "common_step_counter", 0))
    current_iteration = float(step) / float(max(steps_per_iteration, 1))
    if current_iteration < workspace_position_std_stage1_end_iteration:
        current_workspace_position_std = float(workspace_position_std_stage1)
    elif current_iteration < workspace_position_std_stage2_end_iteration:
        current_workspace_position_std = float(workspace_position_std_stage2)
    else:
        current_workspace_position_std = float(workspace_position_std_stage3)

    if current_iteration < loco_stage_end_iteration:
        stage_progress = _clamp_progress(current_iteration, 0, loco_stage_end_iteration)
        current_position_range_b = tuple(float(v) for v in position_range_b_loco_stage)
        current_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_loco_stage)
        current_world_z_range = tuple(float(v) for v in world_z_range_loco_stage)
        current_secondary_position_range_b = None
        current_secondary_euler_xyz_range_b = None
        current_secondary_world_z_range = None
        current_secondary_sample_prob = 0.0
        current_tertiary_position_range_b = None
        current_tertiary_euler_xyz_range_b = None
        current_tertiary_world_z_range = None
        current_tertiary_sample_prob = 0.0
        current_reset_joint_position_range = tuple(float(v) for v in reset_joint_position_range_stage1)
        current_reset_joint_velocity_range = tuple(float(v) for v in reset_joint_velocity_range_stage1)
        current_reset_root_x_range = tuple(float(v) for v in reset_root_x_range_stage1)
        current_reset_root_y_range = tuple(float(v) for v in reset_root_y_range_stage1)
        current_reset_root_yaw_range = tuple(float(v) for v in reset_root_yaw_range_stage1)
        current_gating_fixed_d = None
        current_height_termination_progress = 0.0
        stage_value = stage_progress
    elif current_iteration < stage1_end_iteration:
        stage_progress = _clamp_progress(current_iteration, loco_stage_end_iteration, stage1_end_iteration)
        if position_range_b_stage1_start is not None:
            current_position_range_b = _lerp_tuple(
                position_range_b_stage1_start, position_range_b_stage1, stage_progress
            )
        else:
            current_position_range_b = tuple(float(v) for v in position_range_b_stage1)
        if euler_xyz_range_b_stage1_start is not None:
            current_euler_xyz_range_b = _lerp_tuple(
                euler_xyz_range_b_stage1_start, euler_xyz_range_b_stage1, stage_progress
            )
        else:
            current_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_stage1)
        if world_z_range_stage1_start is not None:
            current_world_z_range = _lerp_tuple(world_z_range_stage1_start, world_z_range_stage1, stage_progress)
        else:
            current_world_z_range = tuple(float(v) for v in world_z_range_stage1)
        current_secondary_position_range_b = None
        current_secondary_euler_xyz_range_b = None
        current_secondary_world_z_range = None
        current_secondary_sample_prob = 0.0
        current_tertiary_position_range_b = None
        current_tertiary_euler_xyz_range_b = None
        current_tertiary_world_z_range = None
        current_tertiary_sample_prob = 0.0
        current_reset_joint_position_range = tuple(float(v) for v in reset_joint_position_range_stage1)
        current_reset_joint_velocity_range = tuple(float(v) for v in reset_joint_velocity_range_stage1)
        current_reset_root_x_range = tuple(float(v) for v in reset_root_x_range_stage1)
        current_reset_root_y_range = tuple(float(v) for v in reset_root_y_range_stage1)
        current_reset_root_yaw_range = tuple(float(v) for v in reset_root_yaw_range_stage1)
        current_gating_fixed_d = None
        current_height_termination_progress = 0.0
        stage_value = 1.0 + stage_progress
    elif current_iteration < stage2_hold_end_iteration:
        stage_progress = _clamp_progress(current_iteration, stage1_end_iteration, stage2_hold_end_iteration)
        current_position_range_b = tuple(float(v) for v in position_range_b_stage2_allowed_start)
        current_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_stage2_allowed)
        current_world_z_range = tuple(float(v) for v in world_z_range_stage2_allowed_start)
        current_secondary_position_range_b = tuple(float(v) for v in position_range_b_hard_low_start)
        current_secondary_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_hard_low)
        current_secondary_world_z_range = tuple(float(v) for v in world_z_range_hard_low_start)
        current_secondary_sample_prob = float(hard_low_sample_prob_stage2_base)
        current_tertiary_position_range_b = tuple(float(v) for v in position_range_b_hard_high_start)
        current_tertiary_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_hard_high)
        current_tertiary_world_z_range = tuple(float(v) for v in world_z_range_hard_high_start)
        current_tertiary_sample_prob = float(hard_high_sample_prob_stage2_base)
        current_reset_joint_position_range = _lerp_tuple(
            reset_joint_position_range_stage1, reset_joint_position_range_stage2, stage_progress
        )
        current_reset_joint_velocity_range = _lerp_tuple(
            reset_joint_velocity_range_stage1, reset_joint_velocity_range_stage2, stage_progress
        )
        current_reset_root_x_range = _lerp_tuple(reset_root_x_range_stage1, reset_root_x_range_stage2, stage_progress)
        current_reset_root_y_range = _lerp_tuple(reset_root_y_range_stage1, reset_root_y_range_stage2, stage_progress)
        current_reset_root_yaw_range = _lerp_tuple(
            reset_root_yaw_range_stage1, reset_root_yaw_range_stage2, stage_progress
        )
        current_gating_fixed_d = None
        current_height_termination_progress = stage_progress
        stage_value = 2.0 + stage_progress
    elif current_iteration < stage2_expand_end_iteration:
        stage_progress = _frontloaded_progress(
            current_iteration, stage2_hold_end_iteration, stage2_expand_end_iteration, stage2_expand_reach_fraction
        )
        current_position_range_b = tuple(float(v) for v in position_range_b_stage2_allowed_start)
        current_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_stage2_allowed)
        current_world_z_range = _lerp_tuple(world_z_range_stage2_allowed_start, world_z_range_stage3, stage_progress)
        current_secondary_position_range_b = _lerp_tuple(
            position_range_b_hard_low_start, position_range_b_hard_low_final, stage_progress
        )
        current_secondary_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_hard_low)
        current_secondary_world_z_range = _lerp_tuple(
            world_z_range_hard_low_start, world_z_range_hard_low_final, stage_progress
        )
        current_secondary_sample_prob = float(hard_low_sample_prob_stage2_base)
        current_tertiary_position_range_b = _lerp_tuple(
            position_range_b_hard_high_start, position_range_b_hard_high_final, stage_progress
        )
        current_tertiary_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_hard_high)
        current_tertiary_world_z_range = _lerp_tuple(
            world_z_range_hard_high_start, world_z_range_hard_high_final, stage_progress
        )
        current_tertiary_sample_prob = float(hard_high_sample_prob_stage2_base)
        current_reset_joint_position_range = _lerp_tuple(
            reset_joint_position_range_stage1, reset_joint_position_range_stage2, stage_progress
        )
        current_reset_joint_velocity_range = _lerp_tuple(
            reset_joint_velocity_range_stage1, reset_joint_velocity_range_stage2, stage_progress
        )
        current_reset_root_x_range = _lerp_tuple(reset_root_x_range_stage1, reset_root_x_range_stage2, stage_progress)
        current_reset_root_y_range = _lerp_tuple(reset_root_y_range_stage1, reset_root_y_range_stage2, stage_progress)
        current_reset_root_yaw_range = _lerp_tuple(
            reset_root_yaw_range_stage1, reset_root_yaw_range_stage2, stage_progress
        )
        current_gating_fixed_d = None
        current_height_termination_progress = 1.0
        stage_value = 3.0 + stage_progress
    elif current_iteration < stage2_ratio_end_iteration:
        stage_progress = _frontloaded_progress(
            current_iteration, stage2_expand_end_iteration, stage2_ratio_end_iteration, stage2_ratio_reach_fraction
        )
        current_position_range_b = tuple(float(v) for v in position_range_b_stage2_allowed_start)
        current_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_stage2_allowed)
        current_world_z_range = tuple(float(v) for v in world_z_range_stage3)
        current_secondary_position_range_b = tuple(float(v) for v in position_range_b_hard_low_final)
        current_secondary_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_hard_low)
        current_secondary_world_z_range = tuple(float(v) for v in world_z_range_hard_low_final)
        current_secondary_sample_prob = _lerp_value(
            hard_low_sample_prob_stage2_base, hard_low_sample_prob_stage2_final, stage_progress
        )
        current_tertiary_position_range_b = tuple(float(v) for v in position_range_b_hard_high_final)
        current_tertiary_euler_xyz_range_b = tuple(float(v) for v in euler_xyz_range_b_hard_high)
        current_tertiary_world_z_range = tuple(float(v) for v in world_z_range_hard_high_final)
        current_tertiary_sample_prob = _lerp_value(
            hard_high_sample_prob_stage2_base, hard_high_sample_prob_stage2_final, stage_progress
        )
        current_reset_joint_position_range = tuple(float(v) for v in reset_joint_position_range_stage2)
        current_reset_joint_velocity_range = tuple(float(v) for v in reset_joint_velocity_range_stage2)
        current_reset_root_x_range = tuple(float(v) for v in reset_root_x_range_stage2)
        current_reset_root_y_range = tuple(float(v) for v in reset_root_y_range_stage2)
        current_reset_root_yaw_range = tuple(float(v) for v in reset_root_yaw_range_stage2)
        current_gating_fixed_d = None
        current_height_termination_progress = 1.0
        stage_value = 4.0 + stage_progress
    else:
        stage_progress = _clamp_progress(current_iteration, stage2_ratio_end_iteration, stage3_xy_end_iteration)
        current_position_range_b = (
            _lerp_value(position_range_b_stage2_allowed_start[0], position_range_b_stage3[0], stage_progress),
            _lerp_value(position_range_b_stage2_allowed_start[1], position_range_b_stage3[1], stage_progress),
            _lerp_value(position_range_b_stage2_allowed_start[2], position_range_b_stage3[2], stage_progress),
            _lerp_value(position_range_b_stage2_allowed_start[3], position_range_b_stage3[3], stage_progress),
            float(position_range_b_stage3[4]),
            float(position_range_b_stage3[5]),
        )
        current_euler_xyz_range_b = _lerp_tuple(
            euler_xyz_range_b_stage2_allowed, euler_xyz_range_b_stage3, stage_progress
        )
        current_world_z_range = tuple(float(v) for v in world_z_range_stage3)
        current_secondary_position_range_b = None
        current_secondary_euler_xyz_range_b = None
        current_secondary_world_z_range = None
        current_secondary_sample_prob = 0.0
        current_tertiary_position_range_b = None
        current_tertiary_euler_xyz_range_b = None
        current_tertiary_world_z_range = None
        current_tertiary_sample_prob = 0.0
        current_reset_joint_position_range = _lerp_tuple(
            reset_joint_position_range_stage2, reset_joint_position_range_stage3, stage_progress
        )
        current_reset_joint_velocity_range = _lerp_tuple(
            reset_joint_velocity_range_stage2, reset_joint_velocity_range_stage3, stage_progress
        )
        current_reset_root_x_range = _lerp_tuple(reset_root_x_range_stage2, reset_root_x_range_stage3, stage_progress)
        current_reset_root_y_range = _lerp_tuple(reset_root_y_range_stage2, reset_root_y_range_stage3, stage_progress)
        current_reset_root_yaw_range = _lerp_tuple(
            reset_root_yaw_range_stage2, reset_root_yaw_range_stage3, stage_progress
        )
        current_gating_fixed_d = None
        current_height_termination_progress = 1.0
        stage_value = 5.0 + stage_progress

    command_cfg.position_range_b = current_position_range_b
    command_cfg.sample_z_in_world_frame = True
    command_cfg.world_z_range = current_world_z_range
    command_cfg.euler_xyz_range_b = current_euler_xyz_range_b
    command_cfg.secondary_position_range_b = current_secondary_position_range_b
    command_cfg.secondary_euler_xyz_range_b = current_secondary_euler_xyz_range_b
    command_cfg.secondary_world_z_range = current_secondary_world_z_range
    command_cfg.secondary_sample_prob = current_secondary_sample_prob
    command_cfg.tertiary_position_range_b = current_tertiary_position_range_b
    command_cfg.tertiary_euler_xyz_range_b = current_tertiary_euler_xyz_range_b
    command_cfg.tertiary_world_z_range = current_tertiary_world_z_range
    command_cfg.tertiary_sample_prob = current_tertiary_sample_prob

    env.cfg.commands.ee_pose.position_range_b = current_position_range_b
    env.cfg.commands.ee_pose.sample_z_in_world_frame = True
    env.cfg.commands.ee_pose.world_z_range = current_world_z_range
    env.cfg.commands.ee_pose.euler_xyz_range_b = current_euler_xyz_range_b
    env.cfg.commands.ee_pose.secondary_position_range_b = current_secondary_position_range_b
    env.cfg.commands.ee_pose.secondary_euler_xyz_range_b = current_secondary_euler_xyz_range_b
    env.cfg.commands.ee_pose.secondary_world_z_range = current_secondary_world_z_range
    env.cfg.commands.ee_pose.secondary_sample_prob = current_secondary_sample_prob
    env.cfg.commands.ee_pose.tertiary_position_range_b = current_tertiary_position_range_b
    env.cfg.commands.ee_pose.tertiary_euler_xyz_range_b = current_tertiary_euler_xyz_range_b
    env.cfg.commands.ee_pose.tertiary_world_z_range = current_tertiary_world_z_range
    env.cfg.commands.ee_pose.tertiary_sample_prob = current_tertiary_sample_prob
    if hasattr(env.cfg.rewards, total_reward_term_name):
        getattr(env.cfg.rewards, total_reward_term_name).params["gating_fixed_d"] = current_gating_fixed_d
        getattr(env.cfg.rewards, total_reward_term_name).params["workspace_position_std"] = (
            current_workspace_position_std
        )
    total_reward_term_cfg = env.reward_manager.get_term_cfg(total_reward_term_name)
    total_reward_term_cfg.params["gating_fixed_d"] = current_gating_fixed_d
    total_reward_term_cfg.params["workspace_position_std"] = current_workspace_position_std

    if (
        base_height_termination_soft_normal is not None
        and base_height_termination_hard_normal is not None
        and base_height_termination_soft_low is not None
        and base_height_termination_hard_low is not None
    ):
        current_soft_minimum_height = _lerp_value(
            base_height_termination_soft_normal,
            base_height_termination_soft_low,
            current_height_termination_progress,
        )
        current_hard_minimum_height = _lerp_value(
            base_height_termination_hard_normal,
            base_height_termination_hard_low,
            current_height_termination_progress,
        )
        if hasattr(env.cfg.terminations, "base_height_termination"):
            env.cfg.terminations.base_height_termination.params["soft_minimum_height"] = current_soft_minimum_height
            env.cfg.terminations.base_height_termination.params["hard_minimum_height"] = current_hard_minimum_height
        if hasattr(env, "termination_manager"):
            try:
                base_height_term_cfg = env.termination_manager.get_term_cfg("base_height_termination")
            except Exception:
                base_height_term_cfg = None
            if base_height_term_cfg is not None:
                base_height_term_cfg.params["soft_minimum_height"] = current_soft_minimum_height
                base_height_term_cfg.params["hard_minimum_height"] = current_hard_minimum_height

    env.cfg.events.randomize_reset_joints.params["position_range"] = current_reset_joint_position_range
    env.cfg.events.randomize_reset_joints.params["velocity_range"] = current_reset_joint_velocity_range
    env.cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": current_reset_root_x_range,
        "y": current_reset_root_y_range,
        "yaw": current_reset_root_yaw_range,
    }
    return torch.tensor(stage_value, device=env.device)

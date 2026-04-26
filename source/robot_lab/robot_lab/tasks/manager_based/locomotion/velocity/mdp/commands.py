# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """


class EndEffectorPoseCommand(CommandTerm):
    """固定世界系末端位姿命令，并在每一步转换到 base frame。"""

    cfg: EndEffectorPoseCommandCfg

    def __init__(self, cfg: EndEffectorPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # 取出机器人资产，后面所有位姿、速度、关节量都从这里读取。
        self.robot: Articulation = env.scene[cfg.asset_name]
        # 缓存末端执行器和基座对应的 body 索引，避免每步重复查字符串。
        self.ee_body_idx = self.robot.body_names.index(cfg.ee_body_name)
        self.base_body_idx = self.robot.body_names.index(cfg.base_body_name)

        # 世界系下固定目标位置。每个 episode 采样一次，episode 内保持不变。
        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # 世界系下固定目标姿态，内部用四元数存。
        self.target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        # 四元数单位元，表示无旋转。
        self.target_quat_w[:, 0] = 1.0

        # 暴露给策略网络的命令缓存：3 维位置 + 9 维旋转矩阵。
        self.command_buffer = torch.zeros(self.num_envs, 12, device=self.device)
        # 当前末端在 base frame 下的位置。
        self.ee_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_ee_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        # 缓存每个 episode reset 后首帧的末端初始姿态欧拉角，便于直接从训练日志读取初始位姿。
        self.initial_ee_euler_b = torch.zeros(self.num_envs, 3, device=self.device)
        # 当前末端在 base frame 下的旋转矩阵。
        self.ee_rotmat_b = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        # 目标末端在 base frame 下的位置。
        self.target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        # 缓存每个 episode 采样瞬间的 base-frame 命令位置；日志按它区分近端/远端任务。
        self.sampled_target_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        # 目标末端在 base frame 下的旋转矩阵。
        self.target_rotmat_b = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)

        # 当前时刻的跟踪误差 e_track(t)。
        self.position_tracking_error = torch.zeros(self.num_envs, device=self.device)
        self.orientation_tracking_error = torch.zeros(self.num_envs, device=self.device)
        self.tracking_error = torch.zeros(self.num_envs, device=self.device)
        # 每个 episode 开始时记录的 e_track(0)。
        self.initial_tracking_error = torch.zeros(self.num_envs, device=self.device)
        # 参考跟踪误差 e_ref(t) = max(e_track(0) - v t, 0)。
        self.reference_tracking_error = torch.zeros(self.num_envs, device=self.device)
        # 累积误差，当前先用固定系数做逐步累加。
        self.cumulative_tracking_error = torch.zeros(self.num_envs, device=self.device)
        # 标记哪些环境还没有完成 e_track(0) 的初始化。
        self._needs_error_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # 这些指标会被记录到 extras / logger 中，方便调试和后续做 reward。
        self.metrics["tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cumulative_tracking_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cumulative_error_gate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        # 返回当前命令张量，供 generated_commands 这类观测接口直接读取。
        return self.command_buffer

    def _resample_command(self, env_ids: Sequence[int]):
        # 没有需要重采样的环境时直接返回。
        if len(env_ids) == 0:
            return

        # 统一转成 tensor，便于后面按索引回写。
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        # 本次需要采样的环境数量。
        num_samples = len(env_ids)
        # 优先使用 base frame 采样；未配置时退回到世界系采样。
        sample_in_base_frame = self.cfg.position_range_b is not None
        # 位置采样范围，格式为 xmin xmax ymin ymax zmin zmax。
        primary_pos_range_values = self.cfg.position_range_b if sample_in_base_frame else self.cfg.position_range_w
        primary_pos_ranges = torch.tensor(primary_pos_range_values, dtype=torch.float32, device=self.device)
        # 若启用“xy 用 base frame、z 用 world frame”的混合采样，则 z 单独从世界系范围采样。
        sample_world_z = (
            sample_in_base_frame and self.cfg.sample_z_in_world_frame and self.cfg.world_z_range is not None
        )
        primary_world_z_ranges = (
            torch.tensor(self.cfg.world_z_range, dtype=torch.float32, device=self.device) if sample_world_z else None
        )
        # 欧拉角采样范围，格式为 roll_min roll_max pitch_min pitch_max yaw_min yaw_max。
        primary_euler_range_values = (
            self.cfg.euler_xyz_range_b
            if sample_in_base_frame and self.cfg.euler_xyz_range_b is not None
            else self.cfg.euler_xyz_range
        )
        primary_euler_ranges = torch.tensor(primary_euler_range_values, dtype=torch.float32, device=self.device)
        # 可选的第二/第三采样分布，仅用于显式课程混采，例如低 z / 高 z 高难区。
        has_secondary_distribution = (
            sample_in_base_frame
            and self.cfg.secondary_position_range_b is not None
            and self.cfg.secondary_euler_xyz_range_b is not None
            and self.cfg.secondary_sample_prob > 0.0
        )
        has_tertiary_distribution = (
            sample_in_base_frame
            and self.cfg.tertiary_position_range_b is not None
            and self.cfg.tertiary_euler_xyz_range_b is not None
            and self.cfg.tertiary_sample_prob > 0.0
        )
        if has_secondary_distribution:
            secondary_pos_ranges = torch.tensor(
                self.cfg.secondary_position_range_b, dtype=torch.float32, device=self.device
            )
            secondary_euler_ranges = torch.tensor(
                self.cfg.secondary_euler_xyz_range_b, dtype=torch.float32, device=self.device
            )
            secondary_world_z_ranges = (
                torch.tensor(self.cfg.secondary_world_z_range, dtype=torch.float32, device=self.device)
                if sample_world_z and self.cfg.secondary_world_z_range is not None
                else None
            )
        else:
            secondary_pos_ranges = None
            secondary_euler_ranges = None
            secondary_world_z_ranges = None
        if has_tertiary_distribution:
            tertiary_pos_ranges = torch.tensor(
                self.cfg.tertiary_position_range_b, dtype=torch.float32, device=self.device
            )
            tertiary_euler_ranges = torch.tensor(
                self.cfg.tertiary_euler_xyz_range_b, dtype=torch.float32, device=self.device
            )
            tertiary_world_z_ranges = (
                torch.tensor(self.cfg.tertiary_world_z_range, dtype=torch.float32, device=self.device)
                if sample_world_z and self.cfg.tertiary_world_z_range is not None
                else None
            )
        else:
            tertiary_pos_ranges = None
            tertiary_euler_ranges = None
            tertiary_world_z_ranges = None
        if has_secondary_distribution or has_tertiary_distribution:
            sample_selector = torch.rand(num_samples, device=self.device)
            secondary_prob = float(self.cfg.secondary_sample_prob) if has_secondary_distribution else 0.0
            tertiary_prob = float(self.cfg.tertiary_sample_prob) if has_tertiary_distribution else 0.0
            use_secondary_distribution = (
                (sample_selector < secondary_prob)
                if has_secondary_distribution
                else torch.zeros(num_samples, dtype=torch.bool, device=self.device)
            )
            use_tertiary_distribution = (
                ((sample_selector >= secondary_prob) & (sample_selector < secondary_prob + tertiary_prob))
                if has_tertiary_distribution
                else torch.zeros(num_samples, dtype=torch.bool, device=self.device)
            )
        else:
            use_secondary_distribution = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
            use_tertiary_distribution = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        # 每个并行环境自己的世界原点；这里同时用作地面高度参考。
        env_origins = self._env.scene.env_origins[env_ids_tensor]
        # base-relative 采样时，先读取 reset 后当前 base 的世界系位姿，再把局部目标映射到世界系。
        if sample_in_base_frame:
            base_pos_w = self.robot.data.body_pos_w[env_ids_tensor, self.base_body_idx]
            base_quat_w = self.robot.data.body_quat_w[env_ids_tensor, self.base_body_idx]

        # 记录哪些环境已经采到了合法目标。
        valid_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        # 暂存采样到的世界系目标位置。
        sampled_pos = torch.zeros(num_samples, 3, device=self.device)
        # 暂存采样瞬间的 base-frame 目标位置，不随 episode 内 base 运动更新。
        sampled_pos_b = torch.zeros(num_samples, 3, device=self.device)
        # 暂存采样到的世界系目标姿态四元数。
        sampled_quat = torch.zeros(num_samples, 4, device=self.device)
        # 默认先设成单位四元数，防止有环境最后没采成功时留下非法值。
        sampled_quat[:, 0] = 1.0

        # 允许多次重采样，直到全部环境拿到合法目标，或者达到最大尝试次数。
        attempts = 0
        while not torch.all(valid_mask) and attempts < self.cfg.max_sampling_tries:
            # 只对尚未成功的环境继续采样。
            pending_mask = ~valid_mask
            pending_count = int(pending_mask.sum().item())
            pending_indices = torch.where(pending_mask)[0]
            pending_use_secondary = use_secondary_distribution[pending_indices]
            pending_use_tertiary = use_tertiary_distribution[pending_indices]

            pending_pos_ranges = primary_pos_ranges.unsqueeze(0).repeat(pending_count, 1)
            pending_euler_ranges = primary_euler_ranges.unsqueeze(0).repeat(pending_count, 1)
            pending_world_z_ranges = (
                primary_world_z_ranges.unsqueeze(0).repeat(pending_count, 1)
                if sample_world_z and primary_world_z_ranges is not None
                else None
            )
            if has_secondary_distribution and secondary_pos_ranges is not None and secondary_euler_ranges is not None:
                pending_pos_ranges[pending_use_secondary] = secondary_pos_ranges
                pending_euler_ranges[pending_use_secondary] = secondary_euler_ranges
                if pending_world_z_ranges is not None and secondary_world_z_ranges is not None:
                    pending_world_z_ranges[pending_use_secondary] = secondary_world_z_ranges
            if has_tertiary_distribution and tertiary_pos_ranges is not None and tertiary_euler_ranges is not None:
                pending_pos_ranges[pending_use_tertiary] = tertiary_pos_ranges
                pending_euler_ranges[pending_use_tertiary] = tertiary_euler_ranges
                if pending_world_z_ranges is not None and tertiary_world_z_ranges is not None:
                    pending_world_z_ranges[pending_use_tertiary] = tertiary_world_z_ranges

            # 在世界系范围内均匀采样目标位置。
            sampled_world_z = (
                (
                    torch.empty(pending_count, device=self.device).uniform_(0.0, 1.0)
                    * (pending_world_z_ranges[:, 1] - pending_world_z_ranges[:, 0])
                    + pending_world_z_ranges[:, 0]
                )
                if pending_world_z_ranges is not None
                else None
            )
            sampled_x_local = (
                torch.empty(pending_count, device=self.device).uniform_(0.0, 1.0)
                * (pending_pos_ranges[:, 1] - pending_pos_ranges[:, 0])
                + pending_pos_ranges[:, 0]
            )
            sampled_y_local = (
                torch.empty(pending_count, device=self.device).uniform_(0.0, 1.0)
                * (pending_pos_ranges[:, 3] - pending_pos_ranges[:, 2])
                + pending_pos_ranges[:, 2]
            )
            sampled_z_local = (
                (
                    torch.empty(pending_count, device=self.device).uniform_(0.0, 1.0)
                    * (pending_pos_ranges[:, 5] - pending_pos_ranges[:, 4])
                    + pending_pos_ranges[:, 4]
                )
                if sampled_world_z is None
                else None
            )
            pos_local = torch.stack(
                [
                    sampled_x_local,
                    sampled_y_local,
                    sampled_z_local if sampled_z_local is not None else torch.zeros(pending_count, device=self.device),
                ],
                dim=-1,
            )
            # 先在欧拉角空间采样，再转换成姿态四元数，避免直接采随机旋转矩阵。
            euler_xyz = torch.stack(
                [
                    torch.empty(pending_count, device=self.device).uniform_(0.0, 1.0)
                    * (pending_euler_ranges[:, 1] - pending_euler_ranges[:, 0])
                    + pending_euler_ranges[:, 0],
                    torch.empty(pending_count, device=self.device).uniform_(0.0, 1.0)
                    * (pending_euler_ranges[:, 3] - pending_euler_ranges[:, 2])
                    + pending_euler_ranges[:, 2],
                    torch.empty(pending_count, device=self.device).uniform_(0.0, 1.0)
                    * (pending_euler_ranges[:, 5] - pending_euler_ranges[:, 4])
                    + pending_euler_ranges[:, 4],
                ],
                dim=-1,
            )

            # 找到这轮对应的真实环境索引。
            pending_env_origins = env_origins[pending_indices]
            if sample_in_base_frame:
                if sampled_world_z is not None:
                    # 混合坐标采样时，xy 是 base frame 水平位移，z 单独由世界系高度控制。
                    xy_local = pos_local.clone()
                    xy_local[:, 2] = 0.0
                    sampled_pos_w = base_pos_w[pending_indices] + quat_apply(base_quat_w[pending_indices], xy_local)
                    sampled_pos_w[:, 2] = sampled_world_z
                    identity_quat = torch.zeros(pending_count, 4, device=self.device)
                    identity_quat[:, 0] = 1.0
                    pos_local, _ = subtract_frame_transforms(
                        base_pos_w[pending_indices], base_quat_w[pending_indices], sampled_pos_w, identity_quat
                    )
                else:
                    # 纯 base frame 采样时，直接做刚体变换即可。
                    sampled_pos_w = base_pos_w[pending_indices] + quat_apply(base_quat_w[pending_indices], pos_local)
            else:
                # 世界系采样时，只需要加上各自环境原点。
                sampled_pos_w = pending_env_origins + pos_local

            # 初始认为本轮采到的点都合法。
            keep_mask = torch.ones(pending_count, dtype=torch.bool, device=self.device)
            if self.cfg.reject_position_cuboid is not None:
                # 混合坐标采样时，先构造真实世界目标点，再反算回 base frame 做拒绝判断。
                reject = torch.tensor(self.cfg.reject_position_cuboid, dtype=torch.float32, device=self.device)
                in_reject = (
                    (pos_local[:, 0] >= reject[0])
                    & (pos_local[:, 0] <= reject[1])
                    & (pos_local[:, 1] >= reject[2])
                    & (pos_local[:, 1] <= reject[3])
                    & (pos_local[:, 2] >= reject[4])
                    & (pos_local[:, 2] <= reject[5])
                )
                keep_mask &= ~in_reject

            # 平地操作时，目标不能落到地面以下。
            keep_mask &= sampled_pos_w[:, 2] >= pending_env_origins[:, 2]
            # 只保留合法采样结果对应的环境索引。
            accepted_indices = pending_indices[keep_mask]
            if accepted_indices.numel() > 0:
                # 提取这些环境对应的欧拉角。
                accepted_euler = euler_xyz[keep_mask]
                # 欧拉角转四元数，先得到采样坐标系下的目标姿态。
                sampled_local_quat = quat_from_euler_xyz(
                    accepted_euler[:, 0], accepted_euler[:, 1], accepted_euler[:, 2]
                )
                if sample_in_base_frame:
                    # 先在 base frame 中采样，再映射到世界系，保证目标相对基座可达。
                    accepted_base_quat_w = base_quat_w[accepted_indices]
                    sampled_pos[accepted_indices] = sampled_pos_w[keep_mask]
                    sampled_pos_b[accepted_indices] = pos_local[keep_mask]
                    sampled_quat[accepted_indices] = quat_mul(accepted_base_quat_w, sampled_local_quat)
                else:
                    # 把相对环境原点的采样位置转成世界坐标。
                    sampled_pos[accepted_indices] = sampled_pos_w[keep_mask]
                    sampled_pos_b[accepted_indices] = pos_local[keep_mask]
                    sampled_quat[accepted_indices] = sampled_local_quat
                # 标记这些环境采样成功。
                valid_mask[accepted_indices] = True
            attempts += 1

        if not torch.all(valid_mask):
            # 如果有环境始终采不到合法点，就退化到采样范围中心，保证命令有效。
            remaining = torch.where(~valid_mask)[0]
            remaining_pos_ranges = primary_pos_ranges.unsqueeze(0).repeat(remaining.numel(), 1)
            remaining_world_z_ranges = (
                primary_world_z_ranges.unsqueeze(0).repeat(remaining.numel(), 1)
                if sample_world_z and primary_world_z_ranges is not None
                else None
            )
            if has_secondary_distribution and secondary_pos_ranges is not None:
                remaining_use_secondary = use_secondary_distribution[remaining]
                remaining_pos_ranges[remaining_use_secondary] = secondary_pos_ranges
                if remaining_world_z_ranges is not None and secondary_world_z_ranges is not None:
                    remaining_world_z_ranges[remaining_use_secondary] = secondary_world_z_ranges
            if has_tertiary_distribution and tertiary_pos_ranges is not None:
                remaining_use_tertiary = use_tertiary_distribution[remaining]
                remaining_pos_ranges[remaining_use_tertiary] = tertiary_pos_ranges
                if remaining_world_z_ranges is not None and tertiary_world_z_ranges is not None:
                    remaining_world_z_ranges[remaining_use_tertiary] = tertiary_world_z_ranges
            fallback_local = torch.stack(
                [
                    0.5 * (remaining_pos_ranges[:, 0] + remaining_pos_ranges[:, 1]),
                    0.5 * (remaining_pos_ranges[:, 2] + remaining_pos_ranges[:, 3]),
                    torch.zeros(remaining.numel(), device=self.device)
                    if remaining_world_z_ranges is not None
                    else 0.5 * (remaining_pos_ranges[:, 4] + remaining_pos_ranges[:, 5]),
                ],
                dim=-1,
            )
            if sample_in_base_frame:
                sampled_pos[remaining] = base_pos_w[remaining] + quat_apply(base_quat_w[remaining], fallback_local)
                sampled_pos_b[remaining] = fallback_local
                if remaining_world_z_ranges is not None:
                    sampled_pos[remaining, 2] = 0.5 * (remaining_world_z_ranges[:, 0] + remaining_world_z_ranges[:, 1])
                    sampled_pos_b[remaining, 2] = sampled_pos[remaining, 2] - base_pos_w[remaining, 2]
            else:
                sampled_pos[remaining] = env_origins[remaining] + fallback_local
                sampled_pos_b[remaining] = fallback_local
            sampled_pos[remaining, 2] = torch.maximum(sampled_pos[remaining, 2], env_origins[remaining, 2])

        # 回写新的世界系目标。
        self.target_pos_w[env_ids_tensor] = sampled_pos
        self.sampled_target_pos_b[env_ids_tensor] = sampled_pos_b
        self.target_quat_w[env_ids_tensor] = sampled_quat
        # 新 episode 开始，误差相关量清零并等待用第一帧真实状态初始化。
        self.position_tracking_error[env_ids_tensor] = 0.0
        self.orientation_tracking_error[env_ids_tensor] = 0.0
        self.initial_tracking_error[env_ids_tensor] = 0.0
        self.reference_tracking_error[env_ids_tensor] = 0.0
        self.cumulative_tracking_error[env_ids_tensor] = 0.0
        self._needs_error_init[env_ids_tensor] = True

    def _update_command(self):
        # 读取当前基座与末端的世界系位姿。
        base_pos_w = self.robot.data.body_pos_w[:, self.base_body_idx]
        base_quat_w = self.robot.data.body_quat_w[:, self.base_body_idx]
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.ee_body_idx]

        # 把当前末端位姿变换到 base frame。
        self.ee_pos_b, ee_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, ee_pos_w, ee_quat_w)
        # 把固定世界系目标变换到当前时刻的 base frame。
        self.target_pos_b, target_quat_b = subtract_frame_transforms(
            base_pos_w, base_quat_w, self.target_pos_w, self.target_quat_w
        )
        # 四元数转旋转矩阵，便于按 9 维矩阵形式输出姿态命令和当前姿态。
        self.ee_rotmat_b = matrix_from_quat(ee_quat_b)
        self.target_rotmat_b = matrix_from_quat(target_quat_b)
        # 命令最终拼成 12 维张量。
        self.command_buffer = torch.cat(
            [self.target_pos_b, self.target_rotmat_b.reshape(self.num_envs, -1)],
            dim=-1,
        )

        # 每一步只计算一次位置误差和姿态误差，后续 observation / reward 直接复用缓存。
        self.position_tracking_error, self.orientation_tracking_error = self._compute_tracking_errors(
            ee_pos_w, ee_quat_w
        )
        self.tracking_error = (
            self.cfg.position_error_weight * self.position_tracking_error
            + self.cfg.orientation_error_weight * self.orientation_tracking_error
        )

        # 记录哪些环境当前还是 episode 的第一帧。
        init_mask = self._needs_error_init.clone()
        init_ids = torch.where(init_mask)[0]
        if init_ids.numel() > 0:
            # 第一帧时，用 reset 后的当前误差作为 e_track(0)。
            self.initial_tracking_error[init_ids] = self.tracking_error[init_ids]
            self.initial_ee_pos_b[init_ids] = self.ee_pos_b[init_ids]
            self.initial_ee_euler_b[init_ids] = self._quat_to_euler_xyz(ee_quat_b[init_ids])
            # 累积误差从第一帧开始累计。
            # 完成初始化后，后续这些环境就进入常规累计分支。
            self._needs_error_init[init_ids] = False

        # 非第一帧环境继续按固定系数累加跟踪误差。
        active_ids = torch.where(~self._needs_error_init & ~init_mask)[0]

        # episode 内时间 t，定义为已执行步数乘以 step_dt。
        episode_time = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
        # 按你定义的公式更新参考跟踪误差 e_ref(t)。
        self.reference_tracking_error = torch.clamp(
            self.initial_tracking_error - self.cfg.reference_error_velocity * episode_time,
            min=0.0,
        )

        # 累积误差项的每步增量权重改成与 mani 一致的门控 (1 - D)。
        cumulative_error_gate = 1.0 - torch.sigmoid(
            (5.0 / self.cfg.cumulative_error_gating_l)
            * (self.reference_tracking_error - self.cfg.cumulative_error_gating_mu)
        )
        if init_ids.numel() > 0:
            self.cumulative_tracking_error[init_ids] = (
                self.cfg.cumulative_error_weight * cumulative_error_gate[init_ids] * self.tracking_error[init_ids]
            )
        if active_ids.numel() > 0:
            self.cumulative_tracking_error[active_ids] += (
                self.cfg.cumulative_error_weight * cumulative_error_gate[active_ids] * self.tracking_error[active_ids]
            )

    def _update_metrics(self):
        # 把当前误差量同步到 metrics，便于 logger 或调试工具直接读。
        self.metrics["tracking_error"] = self.tracking_error
        self.metrics["position_tracking_error"] = self.position_tracking_error
        self.metrics["orientation_tracking_error"] = self.orientation_tracking_error
        self.metrics["reference_tracking_error"] = self.reference_tracking_error
        self.metrics["cumulative_tracking_error"] = self.cumulative_tracking_error
        self.metrics["cumulative_error_gate"] = 1.0 - torch.sigmoid(
            (5.0 / self.cfg.cumulative_error_gating_l)
            * (self.reference_tracking_error - self.cfg.cumulative_error_gating_mu)
        )

    def _compute_tracking_errors(
        self, ee_pos_w: torch.Tensor, ee_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 位置偏差：当前位置与目标位置差的二范数。
        pos_error = torch.linalg.norm(ee_pos_w - self.target_pos_w, dim=-1)

        # 当前姿态和目标姿态先转成旋转矩阵。
        ee_rot = matrix_from_quat(ee_quat_w)
        target_rot = matrix_from_quat(self.target_quat_w)
        # 旋转误差矩阵：R* R^T。
        rot_delta = torch.matmul(target_rot, ee_rot.transpose(-1, -2))
        # 由旋转矩阵迹恢复旋转角。
        trace = torch.diagonal(rot_delta, dim1=-2, dim2=-1).sum(dim=-1)
        cos_theta = torch.clamp(0.5 * (trace - 1.0), -1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)
        # 对 SO(3) 对数映射的 F 范数，等价于 sqrt(2) * theta。
        rot_error = torch.sqrt(torch.tensor(2.0, device=self.device)) * theta

        # 分别返回两个原子误差，避免后续 reward 再重复计算。
        return pos_error, rot_error

    def _quat_to_euler_xyz(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        # 将 wxyz 四元数转成 xyz 欧拉角，日志里直接输出这个量更便于人工看初始姿态。
        w, x, y, z = quat_wxyz.unbind(dim=-1)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack((roll, pitch, yaw), dim=-1)


@configclass
class EndEffectorPoseCommandCfg(CommandTermCfg):
    """固定世界系末端位姿命令的配置。"""

    class_type: type = EndEffectorPoseCommand

    # 机器人资产名。
    asset_name: str = "robot"
    # 末端执行器 body 名称。
    ee_body_name: str = MISSING
    # 基座 body 名称。
    base_body_name: str = MISSING

    # 世界系位置采样范围：xmin xmax ymin ymax zmin zmax。
    position_range_w: tuple[float, float, float, float, float, float] = (0.2, 0.5, -0.3, 0.3, 0.1, 0.5)
    # 世界系欧拉角采样范围：roll_min roll_max pitch_min pitch_max yaw_min yaw_max。
    euler_xyz_range: tuple[float, float, float, float, float, float] = (
        -0.4,
        0.4,
        -0.4,
        0.4,
        -3.14159,
        3.14159,
    )
    # 可选的 base frame 位置采样范围；配置后优先使用它采样，再映射到世界系。
    position_range_b: tuple[float, float, float, float, float, float] | None = None
    # 是否启用“xy 用 base frame、z 用 world frame”的混合采样。
    sample_z_in_world_frame: bool = False
    # 主分布的世界系 z 采样范围；启用混合采样时使用它替代 position_range_b 里的 z。
    world_z_range: tuple[float, float] | None = None
    # 可选的 base frame 欧拉角采样范围；配置后与 position_range_b 配套生效。
    euler_xyz_range_b: tuple[float, float, float, float, float, float] | None = None
    # 可选的拒绝采样长方体，用于排除机械臂不可达或不合理目标。
    reject_position_cuboid: tuple[float, float, float, float, float, float] | None = None
    # 可选的第二 base-frame 采样范围；用于课程中按比例混采另一类命令分布。
    secondary_position_range_b: tuple[float, float, float, float, float, float] | None = None
    # 第二 base-frame 姿态采样范围；与 secondary_position_range_b 配套。
    secondary_euler_xyz_range_b: tuple[float, float, float, float, float, float] | None = None
    # 第二分布的世界系 z 采样范围。
    secondary_world_z_range: tuple[float, float] | None = None
    # 第二采样分布的占比，取值 [0, 1]。
    secondary_sample_prob: float = 0.0
    # 可选的第三 base-frame 采样范围；用于再叠加一类高难命令分布。
    tertiary_position_range_b: tuple[float, float, float, float, float, float] | None = None
    # 第三 base-frame 姿态采样范围；与 tertiary_position_range_b 配套。
    tertiary_euler_xyz_range_b: tuple[float, float, float, float, float, float] | None = None
    # 第三分布的世界系 z 采样范围。
    tertiary_world_z_range: tuple[float, float] | None = None
    # 第三采样分布的占比，取值 [0, 1]。
    tertiary_sample_prob: float = 0.0
    # 单次重采样最大尝试次数。
    max_sampling_tries: int = 128

    # 跟踪误差中位置项的权重。
    position_error_weight: float = 1.0
    # 跟踪误差中姿态项的权重。
    orientation_error_weight: float = 1.0
    # 参考误差下降速度 v。
    reference_error_velocity: float = 0.0
    # 累积误差固定权重，后面可替换成动态权重。
    cumulative_error_weight: float = 1.0
    # 累积误差每步增量的门控中心，使用与 mani 相同的 D(x) 形式。
    cumulative_error_gating_mu: float = 1.5
    # 累积误差每步增量的门控尺度。
    cumulative_error_gating_l: float = 1.0

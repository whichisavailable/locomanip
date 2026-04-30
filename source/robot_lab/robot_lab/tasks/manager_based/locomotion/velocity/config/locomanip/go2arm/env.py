# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp


class Go2ArmManagerBasedRLEnv(ManagerBasedRLEnv):
    """go2arm 额外调试日志环境。"""

    _GO2ARM_TRACKING_TERMS = {
        "gate_d",
        "tracking_error",
        "position_tracking_error",
        "orientation_tracking_error",
        "reference_tracking_error",
        "cumulative_tracking_error",
    }

    _GO2ARM_MANI_MASKED_TERMS = {
        "position_tracking_error",
        "orientation_tracking_error",
        "reference_tracking_error",
        "cumulative_tracking_error",
        "mani_reward",
        "support_roll_penalty",
        "support_feet_slide_penalty",
        "support_foot_air_penalty",
        "support_non_foot_contact_penalty",
        "target_height_pitch_penalty",
        "min_base_height_penalty",
        "posture_deviation_penalty",
        "joint_limit_safety_penalty",
        "support_left_right_x_symmetry_penalty",
        "support_left_right_y_symmetry_penalty",
        "support_foot_xy_range_penalty",
        "mani_regularization_raw",
        "mani_regularization",
        "ee_tracking_potential",
    }

    _GO2ARM_MANI_UNMASKED_TERMS = {
        "workspace_position_penalty",
    }

    _GO2ARM_LOCO_MASKED_TERMS = {
        "loco_reward",
        "locomotion_tracking",
        "moving_arm_default_deviation_penalty",
        "moving_arm_joint_velocity_penalty",
        "base_height_penalty",
        "base_roll_penalty",
        "base_pitch_penalty",
        "base_roll_ang_vel_penalty",
        "base_pitch_ang_vel_penalty",
        "base_z_vel_penalty",
        "base_lateral_vel_penalty",
        "leg_posture_deviation_penalty",
        "touchdown_left_right_x_symmetry_penalty",
        "touchdown_left_right_y_symmetry_penalty",
        "touchdown_foot_y_distance_penalty",
        "diagonal_foot_symmetry_penalty",
        "feet_contact_soft_trot_weighted_gate",
        "loco_regularization_base_raw",
        "loco_regularization",
    }

    _GO2ARM_REWARD_LOG_ORDER = [
        "gate_d",
        "tracking_error",
        "position_tracking_error",
        "orientation_tracking_error",
        "reference_tracking_error",
        "cumulative_tracking_error",
        "mani_reward",
        "loco_reward",
        "basic_reward",
        "support_roll_penalty",
        "support_feet_slide_penalty",
        "support_foot_air_penalty",
        "support_non_foot_contact_penalty",
        "target_height_pitch_penalty",
        "min_base_height_penalty",
        "posture_deviation_penalty",
        "joint_limit_safety_penalty",
        "support_left_right_x_symmetry_penalty",
        "support_left_right_y_symmetry_penalty",
        "support_foot_xy_range_penalty",
        "mani_regularization_raw",
        "mani_regularization",
        "ee_tracking_potential",
        "workspace_position_penalty",
        "locomotion_tracking",
        "moving_arm_default_deviation_penalty",
        "moving_arm_joint_velocity_penalty",
        "base_height_penalty",
        "base_roll_penalty",
        "base_pitch_penalty",
        "base_roll_ang_vel_penalty",
        "base_pitch_ang_vel_penalty",
        "base_z_vel_penalty",
        "base_lateral_vel_penalty",
        "leg_posture_deviation_penalty",
        "touchdown_left_right_x_symmetry_penalty",
        "touchdown_left_right_y_symmetry_penalty",
        "touchdown_foot_y_distance_penalty",
        "diagonal_foot_symmetry_penalty",
        "feet_contact_soft_trot_weighted_gate",
        "loco_regularization_base_raw",
        "loco_regularization",
        "basic_is_alive",
        "basic_termination_penalty",
        "basic_collision_penalty",
        "basic_action_smoothness_first",
        "basic_action_smoothness_second",
        "basic_joint_torque_sq_penalty",
        "basic_joint_power_penalty",
        "total_reward_debug",
    ]

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._debug_zero_action = bool(getattr(cfg, "debug_zero_action", False))
        self._enable_debug_reward_logging = bool(getattr(cfg, "enable_debug_reward_logging", False))
        self._enable_collision_group_logging = bool(getattr(cfg, "enable_collision_group_logging", False))
        self._enable_contact_verification_logging = bool(getattr(cfg, "enable_contact_verification_logging", False))
        self._enable_termination_debug_logging = bool(getattr(cfg, "enable_termination_debug_logging", False))
        self._enable_play_termination_reason_logging = bool(
            getattr(cfg, "enable_play_termination_reason_logging", False)
        )
        self._episode_log_key_prefixes = tuple(getattr(cfg, "episode_log_key_prefixes", ()) or ())
        reward_log_interval_iterations = getattr(cfg, "reward_log_interval_iterations", None)
        reward_log_steps_per_iteration = int(getattr(cfg, "reward_log_steps_per_iteration", 24))
        if reward_log_interval_iterations is not None:
            self._reward_log_interval = max(1, int(reward_log_interval_iterations) * reward_log_steps_per_iteration)
        else:
            self._reward_log_interval = max(1, int(getattr(cfg, "reward_log_interval", 100)))
        self._reward_log_counter = 0
        self._reward_log_sums: dict[str, torch.Tensor] = {}
        self._reward_log_counts: dict[str, torch.Tensor] = {}
        self._go2arm_reward_cache = None
        self._go2arm_reward_cache_term_name = None
        self.action_manager.prev_prev_action = torch.zeros_like(self.action_manager.action)
        self._validate_go2arm_precise_foot_bodies()

    def _validate_go2arm_precise_foot_bodies(self) -> None:
        """Ensure the current go2arm asset exposes the four feet and dedicated foot sensors."""
        try:
            foot_body_ids, _ = self.scene["robot"].find_bodies(mdp.GO2ARM_FOOT_BODY_NAMES, preserve_order=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Go2Arm precise foot contact requires the robot asset to expose "
                "FL_foot/FR_foot/RL_foot/RR_foot as articulation bodies. "
                "The current asset does not provide that layout."
            ) from exc
        if len(foot_body_ids) != 4:
            raise RuntimeError(
                f"Go2Arm precise foot contact expected 4 articulation foot bodies, but found {len(foot_body_ids)}."
            )
        missing_sensors = [
            sensor_name for sensor_name in mdp.GO2ARM_FOOT_SENSOR_NAMES if sensor_name not in self.scene.sensors
        ]
        if missing_sensors:
            raise RuntimeError(
                "Go2Arm precise foot contact requires the four dedicated foot sensors, "
                f"but the following sensors are missing: {missing_sensors}."
            )
        self._go2arm_has_foot_sensors = True
        self._go2arm_foot_contact_sensors = tuple(
            self.scene.sensors[sensor_name] for sensor_name in mdp.GO2ARM_FOOT_SENSOR_NAMES
        )

    def _reward_log_key(self, term_name: str) -> str:
        if term_name in self._GO2ARM_TRACKING_TERMS:
            group = "tracking"
        elif term_name in self._GO2ARM_MANI_MASKED_TERMS or term_name in self._GO2ARM_MANI_UNMASKED_TERMS:
            group = "mani"
        elif term_name in self._GO2ARM_LOCO_MASKED_TERMS:
            group = "loco"
        elif term_name.startswith("basic_"):
            group = "basic"
        else:
            group = "misc"
        return f"R/{group}/{term_name}"

    def _as_log_tensor(self, value: float | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.device, dtype=torch.float32)
        return torch.tensor(float(value), device=self.device, dtype=torch.float32)

    def _accumulate_scalar_log(
        self, episode_dict: dict[str, float | torch.Tensor], key: str, value: float | torch.Tensor, count: float = 1.0
    ) -> None:
        episode_dict[key] = self._as_log_tensor(value)
        self._reward_log_sums[key] = self._reward_log_sums.get(key, self._as_log_tensor(0.0)) + self._as_log_tensor(
            value
        )
        self._reward_log_counts[key] = self._reward_log_counts.get(key, self._as_log_tensor(0.0)) + self._as_log_tensor(
            count
        )

    def _accumulate_log_only(self, key: str, value: float, count: float = 1.0) -> None:
        self._reward_log_sums[key] = self._reward_log_sums.get(key, self._as_log_tensor(0.0)) + self._as_log_tensor(
            value
        )
        self._reward_log_counts[key] = self._reward_log_counts.get(key, self._as_log_tensor(0.0)) + self._as_log_tensor(
            count
        )

    def _accumulate_tensor_mean_log(
        self,
        episode_dict: dict[str, float | torch.Tensor],
        key: str,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        write_episode: bool = True,
    ) -> None:
        valid_values = values[mask] if mask is not None else values.reshape(-1)
        valid_count = int(valid_values.numel())
        if valid_count > 0:
            sum_value = valid_values.sum()
            mean_value = sum_value / float(valid_count)
        else:
            sum_value = self._as_log_tensor(0.0)
            mean_value = self._as_log_tensor(0.0)
        if write_episode:
            episode_dict[key] = mean_value
        self._accumulate_log_only(key, sum_value, count=float(valid_count))

    def _command_term(self, command_name: str = "ee_pose"):
        return self.command_manager.get_term(command_name)

    def _classify_episode_bucket(
        self, sampled_target_pos_b: torch.Tensor, target_pos_w: torch.Tensor, command_cfg
    ) -> str:
        x_tag = "x_near" if float(sampled_target_pos_b[0]) <= 0.5 else "x_far"
        z_world = float(target_pos_w[2])
        low_range = getattr(command_cfg, "secondary_world_z_range", None)
        high_range = getattr(command_cfg, "tertiary_world_z_range", None)
        z_tag = "z_normal"
        if low_range is not None and float(low_range[0]) <= z_world <= float(low_range[1]):
            z_tag = "z_low_hard"
        elif high_range is not None and float(high_range[0]) <= z_world <= float(high_range[1]):
            z_tag = "z_high_hard"
        return f"{x_tag}/{z_tag}"

    def _accumulate_done_episode_stats(
        self,
        episode_dict: dict[str, float | torch.Tensor],
        done_mask: torch.Tensor,
        prev_episode_length_buf: torch.Tensor,
        prev_sampled_target_pos_b: torch.Tensor,
        prev_target_pos_w: torch.Tensor,
        terminal_tracking_errors: dict[str, torch.Tensor] | None,
    ) -> None:
        if not torch.any(done_mask):
            return

        done_ids = torch.where(done_mask)[0]
        done_episode_lengths = prev_episode_length_buf[done_ids].to(torch.float32) + 1.0
        if terminal_tracking_errors is not None:
            for term_name, term_value in terminal_tracking_errors.items():
                self._accumulate_tensor_mean_log(
                    episode_dict,
                    f"R/tracking/terminal_{term_name}",
                    term_value[done_ids],
                    write_episode=True,
                )
        self._accumulate_log_only(
            "R/misc/terminal_episode_length",
            done_episode_lengths.sum().item(),
            count=float(done_episode_lengths.numel()),
        )
        episode_dict["R/misc/terminal_episode_length"] = done_episode_lengths.sum() / float(
            done_episode_lengths.numel()
        )

        command_cfg = self.cfg.commands.ee_pose
        bucket_to_lengths: dict[str, list[float]] = {}
        for local_idx, env_id in enumerate(done_ids.tolist()):
            bucket = self._classify_episode_bucket(
                prev_sampled_target_pos_b[env_id],
                prev_target_pos_w[env_id],
                command_cfg,
            )
            bucket_to_lengths.setdefault(bucket, []).append(float(done_episode_lengths[local_idx].item()))

        for bucket, lengths in bucket_to_lengths.items():
            bucket_sum = float(sum(lengths))
            bucket_count = float(len(lengths))
            bucket_key = f"Len/{bucket}"
            episode_dict[bucket_key] = self._as_log_tensor(bucket_sum / bucket_count)
            self._accumulate_log_only(bucket_key, bucket_sum, count=bucket_count)

    def _filter_episode_log_dict(self, log_dict: dict[str, float | torch.Tensor]) -> dict[str, float | torch.Tensor]:
        if not log_dict or not self._episode_log_key_prefixes:
            return log_dict
        return {
            key: value
            for key, value in log_dict.items()
            if any(key.startswith(prefix) for prefix in self._episode_log_key_prefixes)
        }

    def _merge_reset_logs_into_episode(self, extras: dict, episode_dict: dict[str, float | torch.Tensor]) -> None:
        log_dict = extras.get("log")
        if not isinstance(log_dict, dict):
            return
        filtered_log_dict = self._filter_episode_log_dict(log_dict)
        if filtered_log_dict:
            episode_dict.update(filtered_log_dict)
        extras["log"] = filtered_log_dict

    def _get_collision_group_masks(self, body_names: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        groups = {
            "base": [],
            "thigh": [],
            "calflower": [],
            "calf": [],
            "foot": [],
            "arm": [],
            "other": [],
        }
        for index, body_name in enumerate(body_names):
            lower_name = body_name.lower()
            if "base" in lower_name:
                groups["base"].append(index)
            elif "thigh" in lower_name or "_hip" in lower_name:
                groups["thigh"].append(index)
            elif "calflower" in lower_name:
                groups["calflower"].append(index)
            elif "calf" in lower_name:
                groups["calf"].append(index)
            elif "foot" in lower_name:
                groups["foot"].append(index)
            elif "link" in lower_name or "gripper" in lower_name or "joint" in lower_name or "arm" in lower_name:
                groups["arm"].append(index)
            else:
                groups["other"].append(index)

        group_masks: dict[str, torch.Tensor] = {}
        num_bodies = len(body_names)
        for group_name, body_indices in groups.items():
            mask = torch.zeros(num_bodies, dtype=torch.bool, device=device)
            if body_indices:
                mask[body_indices] = True
            group_masks[group_name] = mask
        return group_masks

    def _log_collision_groups(self, episode_dict: dict[str, float | torch.Tensor]) -> None:
        contact_sensor: ContactSensor = self.scene.sensors["contact_forces"]
        body_names = contact_sensor.body_names
        group_masks = self._get_collision_group_masks(body_names, device=self.device)

        net_contact_forces = contact_sensor.data.net_forces_w_history
        max_force_per_body = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
        in_contact = max_force_per_body > 1.0

        for group_name, mask in group_masks.items():
            if not torch.any(mask):
                continue
            group_force = max_force_per_body[:, mask]
            group_contact = in_contact[:, mask]
            self._accumulate_scalar_log(
                episode_dict,
                f"Go2ArmCollision/{group_name}_force",
                torch.sum(group_force, dim=1).mean().item(),
            )
            self._accumulate_scalar_log(
                episode_dict,
                f"Go2ArmCollision/{group_name}_count",
                group_contact.float().sum(dim=1).mean().item(),
            )

    def _log_precise_contact_verification(self, episode_dict: dict[str, float | torch.Tensor]) -> None:
        contact_sensor: ContactSensor = self.scene.sensors["contact_forces"]
        try:
            foot_body_ids, _ = contact_sensor.find_bodies(mdp.GO2ARM_FOOT_BODY_NAMES, preserve_order=True)
        except Exception:  # noqa: BLE001
            return
        if len(foot_body_ids) != 4:
            return

        global_force_vectors = contact_sensor.data.net_forces_w
        foot_force_vectors = global_force_vectors[:, foot_body_ids, :]
        foot_force_norm = torch.linalg.norm(foot_force_vectors, dim=-1)
        self._accumulate_scalar_log(
            episode_dict,
            "Go2ArmVerify/global_foot_force_sum",
            torch.sum(foot_force_norm, dim=1).mean().item(),
        )
        self._accumulate_scalar_log(
            episode_dict,
            "Go2ArmVerify/global_foot_fz_sum",
            torch.sum(torch.abs(foot_force_vectors[..., 2]), dim=1).mean().item(),
        )

        filtered_force_vectors_per_foot: list[torch.Tensor] = []
        for sensor_name in ("FL_foot_contact", "FR_foot_contact", "RL_foot_contact", "RR_foot_contact"):
            if sensor_name not in self.scene.sensors:
                return
            foot_contact_sensor: ContactSensor = self.scene.sensors[sensor_name]
            if foot_contact_sensor.data.force_matrix_w is not None:
                filtered_force_vectors_per_foot.append(
                    torch.sum(foot_contact_sensor.data.force_matrix_w[:, 0, :, :], dim=1)
                )
            else:
                filtered_force_vectors_per_foot.append(foot_contact_sensor.data.net_forces_w[:, 0, :])

        filtered_force_vectors = torch.stack(filtered_force_vectors_per_foot, dim=1)
        filtered_force_norm = torch.linalg.norm(filtered_force_vectors, dim=-1)
        self._accumulate_scalar_log(
            episode_dict,
            "Go2ArmVerify/filtered_foot_force_sum",
            torch.sum(filtered_force_norm, dim=1).mean().item(),
        )
        self._accumulate_scalar_log(
            episode_dict,
            "Go2ArmVerify/filtered_foot_fz_sum",
            torch.sum(torch.abs(filtered_force_vectors[..., 2]), dim=1).mean().item(),
        )
        self._accumulate_scalar_log(
            episode_dict,
            "Go2ArmVerify/filtered_foot_contact_count",
            (filtered_force_norm > 1.0).float().sum(dim=1).mean().item(),
        )
        self._accumulate_scalar_log(
            episode_dict,
            "Go2ArmVerify/filtered_vs_global_foot_vector_residual",
            torch.linalg.norm(filtered_force_vectors - foot_force_vectors, dim=-1).mean().item(),
        )
        self._accumulate_scalar_log(
            episode_dict,
            "Go2ArmVerify/filtered_vs_global_foot_force_gap",
            torch.abs(filtered_force_norm - foot_force_norm).mean().item(),
        )

    def _log_termination_terms(
        self,
        episode_dict: dict[str, float | torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        self._accumulate_scalar_log(episode_dict, "Go2ArmTermination/terminated", terminated.float().mean().item())
        self._accumulate_scalar_log(episode_dict, "Go2ArmTermination/truncated", truncated.float().mean().item())

        done_mask = terminated | truncated
        for term_name in self.termination_manager.active_terms:
            term_value = self.termination_manager.get_term(term_name).float()
            self._accumulate_scalar_log(
                episode_dict,
                f"Go2ArmTermination/{term_name}",
                term_value.mean().item(),
            )
            if torch.any(done_mask):
                self._accumulate_scalar_log(
                    episode_dict,
                    f"Go2ArmTerminationOnDone/{term_name}",
                    term_value[done_mask].mean().item(),
                )

    def _log_final_termination_terms(
        self,
        episode_dict: dict[str, float | torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Log only the terminal causes on envs that ended this step."""
        done_mask = terminated | truncated
        if not torch.any(done_mask):
            return

        done_count = int(done_mask.sum().item())
        self._accumulate_log_only(
            "Term/done_count",
            float(done_count),
            count=1.0,
        )
        episode_dict["Term/done_count"] = self._as_log_tensor(float(done_count))
        self._accumulate_log_only(
            "Term/terminated",
            terminated[done_mask].float().mean().item(),
            count=1.0,
        )
        episode_dict["Term/terminated"] = terminated[done_mask].float().mean()
        self._accumulate_log_only(
            "Term/truncated",
            truncated[done_mask].float().mean().item(),
            count=1.0,
        )
        episode_dict["Term/truncated"] = truncated[done_mask].float().mean()

        for term_name in self.termination_manager.active_terms:
            term_value = self.termination_manager.get_term(term_name).float()
            self._accumulate_log_only(
                f"Term/{term_name}",
                term_value[done_mask].mean().item(),
                count=1.0,
            )
            episode_dict[f"Term/{term_name}"] = term_value[done_mask].mean()

    def _collect_termination_reasons(self, terminated: torch.Tensor, truncated: torch.Tensor) -> list[dict]:
        """Collect per-env terminal causes for play-time diagnostics."""
        done_mask = terminated | truncated
        if not torch.any(done_mask):
            return []

        term_values = {
            term_name: self.termination_manager.get_term(term_name).detach().to(device="cpu", dtype=torch.bool)
            for term_name in self.termination_manager.active_terms
        }
        terminated_cpu = terminated.detach().to(device="cpu", dtype=torch.bool)
        truncated_cpu = truncated.detach().to(device="cpu", dtype=torch.bool)

        reasons = []
        for env_id in torch.where(done_mask.detach().to(device="cpu"))[0].tolist():
            active_reasons = [term_name for term_name, term_value in term_values.items() if bool(term_value[env_id])]
            if not active_reasons:
                if bool(truncated_cpu[env_id]):
                    active_reasons.append("truncated")
                elif bool(terminated_cpu[env_id]):
                    active_reasons.append("terminated")
            reasons.append(
                {
                    "env_id": int(env_id),
                    "terminated": bool(terminated_cpu[env_id]),
                    "truncated": bool(truncated_cpu[env_id]),
                    "reasons": active_reasons,
                }
            )
        return reasons

    def step(self, action: torch.Tensor):
        self._go2arm_reward_cache = None
        self._go2arm_reward_cache_term_name = None
        self.action_manager.prev_prev_action = self.action_manager.prev_action.clone()
        prev_episode_length_buf = self.episode_length_buf.clone()
        command_term = self._command_term("ee_pose")
        prev_sampled_target_pos_b = getattr(command_term, "sampled_target_pos_b", command_term.target_pos_b).clone()
        prev_target_pos_w = command_term.target_pos_w.clone()

        if self._debug_zero_action:
            action = torch.zeros_like(action)

        obs, rew, terminated, truncated, extras = super().step(action)
        done_mask = terminated | truncated
        if torch.any(done_mask):
            self.action_manager.prev_prev_action[done_mask] = 0.0
        if self._enable_play_termination_reason_logging and torch.any(done_mask):
            extras["go2arm_termination_reasons"] = self._collect_termination_reasons(
                terminated=terminated, truncated=truncated
            )

        episode_dict: dict[str, float | torch.Tensor] = extras.setdefault("episode", {})
        self._merge_reset_logs_into_episode(extras, episode_dict)

        reward_term_name = "total_reward"
        terminal_tracking_errors = None
        if self._enable_debug_reward_logging and reward_term_name in self.reward_manager.active_terms:
            debug_terms = mdp.go2arm_reward_debug_terms(self, total_reward_term_name=reward_term_name)
            gate = debug_terms.get("gate_d")
            gate_low_mask = gate < 0.1 if gate is not None else None
            gate_high_mask = gate > 0.9 if gate is not None else None
            terminal_tracking_errors = {
                term_name: debug_terms[term_name]
                for term_name in (
                    "tracking_error",
                    "position_tracking_error",
                    "orientation_tracking_error",
                    "reference_tracking_error",
                    "cumulative_tracking_error",
                )
                if term_name in debug_terms
            }

            for name in self._GO2ARM_REWARD_LOG_ORDER:
                if name not in debug_terms:
                    continue
                key = self._reward_log_key(name)
                if name == "tracking_error":
                    continue
                if name in self._GO2ARM_MANI_MASKED_TERMS:
                    self._accumulate_tensor_mean_log(
                        episode_dict, key, debug_terms[name], mask=gate_low_mask, write_episode=True
                    )
                elif name in self._GO2ARM_LOCO_MASKED_TERMS:
                    self._accumulate_tensor_mean_log(
                        episode_dict, key, debug_terms[name], mask=gate_high_mask, write_episode=True
                    )
                else:
                    self._accumulate_tensor_mean_log(episode_dict, key, debug_terms[name], write_episode=True)

        self._accumulate_done_episode_stats(
            episode_dict,
            done_mask=done_mask,
            prev_episode_length_buf=prev_episode_length_buf,
            prev_sampled_target_pos_b=prev_sampled_target_pos_b,
            prev_target_pos_w=prev_target_pos_w,
            terminal_tracking_errors=terminal_tracking_errors,
        )
        self._log_final_termination_terms(episode_dict, terminated=terminated, truncated=truncated)

        next_reward_log_counter = self._reward_log_counter + 1
        should_emit_log = next_reward_log_counter >= self._reward_log_interval

        if should_emit_log and self._enable_collision_group_logging:
            self._log_collision_groups(episode_dict)
        if should_emit_log and self._enable_contact_verification_logging:
            self._log_precise_contact_verification(episode_dict)
        if should_emit_log and self._enable_termination_debug_logging:
            self._log_termination_terms(episode_dict, terminated=terminated, truncated=truncated)

        self._reward_log_counter = next_reward_log_counter

        if should_emit_log:
            log_dict = extras.setdefault("log", {})
            for name, value_sum in self._reward_log_sums.items():
                count = self._reward_log_counts.get(name, self._as_log_tensor(0.0))
                log_dict[name] = (value_sum / count).item() if float(count.item()) > 0.0 else 0.0
            extras["log"] = self._filter_episode_log_dict(log_dict)
            self._reward_log_sums.clear()
            self._reward_log_counts.clear()
            self._reward_log_counter = 0

        extras["episode"] = self._filter_episode_log_dict(episode_dict)

        return obs, rew, terminated, truncated, extras

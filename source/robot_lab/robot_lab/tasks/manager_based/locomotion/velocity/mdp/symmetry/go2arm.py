# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""World-XZ mirror symmetry for the Unitree Go2Arm task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = object

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


_POLAR_SIGN = (1.0, -1.0, 1.0)
_AXIAL_SIGN = (-1.0, 1.0, -1.0)
_FOOT_LEFT_RIGHT_INDEX = (1, 0, 3, 2)
_GO2ARM_ACTION_SIGN = (
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str | None = None,
    is_critic: bool | None = None,
):
    """Augment observations and actions with a global world-XZ mirror.

    The mirror is the reflection ``y -> -y`` in the world frame. Observation
    group semantics are preserved: world-frame privileged vectors remain
    world-frame vectors after the mirrored sample is generated.
    """
    del env

    obs_aug = _augment_observations(obs, obs_type=obs_type, is_critic=is_critic) if obs is not None else None
    actions_aug = _augment_actions(actions) if actions is not None else None
    return obs_aug, actions_aug


def _augment_observations(
    obs: TensorDict | torch.Tensor,
    *,
    obs_type: str | None = None,
    is_critic: bool | None = None,
) -> TensorDict | torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return _augment_flat_observations(obs, obs_type=obs_type, is_critic=is_critic)

    batch_size = obs.batch_size[0]
    obs_aug = obs.repeat(2)

    if "policy" in obs.keys():
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _transform_policy_obs(obs["policy"])

    if "privileged" in obs.keys():
        obs_aug["privileged"][:batch_size] = obs["privileged"]
        obs_aug["privileged"][batch_size:] = _transform_privileged_obs(obs["privileged"])

    if "critic_extra" in obs.keys():
        obs_aug["critic_extra"][:batch_size] = obs["critic_extra"]
        obs_aug["critic_extra"][batch_size:] = obs["critic_extra"]

    return obs_aug


def _augment_flat_observations(
    obs: torch.Tensor,
    *,
    obs_type: str | None = None,
    is_critic: bool | None = None,
) -> torch.Tensor:
    batch_size = obs.shape[0]
    obs_aug = torch.empty(batch_size * 2, obs.shape[1], device=obs.device, dtype=obs.dtype)
    obs_aug[:batch_size] = obs

    if obs_type is None:
        obs_type = "critic" if is_critic else "policy"

    if obs.shape[1] == 92:
        obs_aug[batch_size:] = _transform_policy_obs(obs)
        return obs_aug

    if obs.shape[1] in (148, 149):
        mirrored = obs.clone()
        mirrored[:, 0:92] = _transform_policy_obs(obs[:, 0:92])
        mirrored[:, 92:148] = _transform_privileged_obs(obs[:, 92:148])
        if obs_type == "critic" and obs.shape[1] > 148:
            mirrored[:, 148:] = obs[:, 148:]
        obs_aug[batch_size:] = mirrored
        return obs_aug

    raise ValueError(f"Unsupported Go2Arm flat observation shape for symmetry: {tuple(obs.shape)}.")


def _augment_actions(actions: torch.Tensor) -> torch.Tensor:
    batch_size = actions.shape[0]
    actions_aug = torch.empty(batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype)
    actions_aug[:batch_size] = actions
    actions_aug[batch_size:] = _transform_joint_data(actions)
    return actions_aug


def _transform_policy_obs(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()

    # joint_pos, joint_vel
    obs[:, 0:18] = _transform_joint_data(obs[:, 0:18])
    obs[:, 18:36] = _transform_joint_data(obs[:, 18:36])
    # current ee pose: 3D position + row-major 3x3 rotation matrix
    obs[:, 36:48] = _transform_pose_b(obs[:, 36:48])
    # last actions
    obs[:, 48:66] = _transform_joint_data(obs[:, 48:66])
    # ee command: 3D position + row-major 3x3 rotation matrix
    obs[:, 66:78] = _transform_pose_b(obs[:, 66:78])
    # base linear velocity, angular velocity, projected gravity
    obs[:, 78:81] = _transform_polar_vector(obs[:, 78:81])
    obs[:, 81:84] = _transform_axial_vector(obs[:, 81:84])
    obs[:, 84:87] = _transform_polar_vector(obs[:, 84:87])
    # reference_tracking_error is scalar and unchanged at 87:88.
    obs[:, 88:92] = _switch_feet_scalar(obs[:, 88:92])

    return obs


def _transform_privileged_obs(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()

    # base_height is scalar and unchanged at 0:1.
    obs[:, 1:5] = _switch_feet_scalar(obs[:, 1:5])
    obs[:, 5:17] = _switch_feet_vectors(obs[:, 5:17], vector_dim=3, axial=False)
    obs[:, 17:21] = _switch_feet_scalar(obs[:, 17:21])
    obs[:, 21:27] = _transform_wrench(obs[:, 21:27])
    obs[:, 27:33] = _transform_twist(obs[:, 27:33])
    # base_mass_disturbance and ee_mass_disturbance are unchanged at 33:35.
    obs[:, 35:41] = _transform_wrench(obs[:, 35:41])
    obs[:, 41:47] = _transform_twist(obs[:, 41:47])
    obs[:, 47:55] = _switch_feet_vectors(obs[:, 47:55], vector_dim=2, axial=False)
    # observation_delay is scalar and unchanged at 55:56.

    return obs


def _transform_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_switched = torch.empty_like(joint_data)
    # [FL, FR, RL, RR] x [hip, thigh, calf], then [joint1..joint6].
    joint_data_switched[..., 0:3] = joint_data[..., 3:6]
    joint_data_switched[..., 3:6] = joint_data[..., 0:3]
    joint_data_switched[..., 6:9] = joint_data[..., 9:12]
    joint_data_switched[..., 9:12] = joint_data[..., 6:9]
    joint_data_switched[..., 12:18] = joint_data[..., 12:18]

    signs = torch.tensor(_GO2ARM_ACTION_SIGN, device=joint_data.device, dtype=joint_data.dtype)
    return joint_data_switched * signs


def _transform_pose_b(pose: torch.Tensor) -> torch.Tensor:
    pose = pose.clone()
    pose[:, 0:3] = _transform_polar_vector(pose[:, 0:3])
    rot = pose[:, 3:12].reshape(-1, 3, 3)
    sign = torch.tensor(_POLAR_SIGN, device=pose.device, dtype=pose.dtype)
    rot = rot * sign.view(1, 3, 1) * sign.view(1, 1, 3)
    pose[:, 3:12] = rot.reshape(-1, 9)
    return pose


def _transform_twist(twist: torch.Tensor) -> torch.Tensor:
    twist = twist.clone()
    twist[:, 0:3] = _transform_polar_vector(twist[:, 0:3])
    twist[:, 3:6] = _transform_axial_vector(twist[:, 3:6])
    return twist


def _transform_wrench(wrench: torch.Tensor) -> torch.Tensor:
    wrench = wrench.clone()
    wrench[:, 0:3] = _transform_polar_vector(wrench[:, 0:3])
    wrench[:, 3:6] = _transform_axial_vector(wrench[:, 3:6])
    return wrench


def _transform_polar_vector(vector: torch.Tensor) -> torch.Tensor:
    sign = torch.tensor(_POLAR_SIGN[: vector.shape[-1]], device=vector.device, dtype=vector.dtype)
    return vector * sign


def _transform_axial_vector(vector: torch.Tensor) -> torch.Tensor:
    sign = torch.tensor(_AXIAL_SIGN[: vector.shape[-1]], device=vector.device, dtype=vector.dtype)
    return vector * sign


def _switch_feet_scalar(values: torch.Tensor) -> torch.Tensor:
    return values[..., list(_FOOT_LEFT_RIGHT_INDEX)]


def _switch_feet_vectors(values: torch.Tensor, *, vector_dim: int, axial: bool) -> torch.Tensor:
    values_by_foot = values.reshape(values.shape[0], 4, vector_dim)
    values_by_foot = values_by_foot[:, list(_FOOT_LEFT_RIGHT_INDEX), :]
    if axial:
        values_by_foot = _transform_axial_vector(values_by_foot)
    else:
        values_by_foot = _transform_polar_vector(values_by_foot)
    return values_by_foot.reshape(values.shape[0], 4 * vector_dim)

# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.rsl_rl_compat import (
    MLP,
    EmpiricalNormalization,
)


def _activation_module(activation: str) -> nn.Module:
    activation = activation.lower()
    if activation == "elu":
        return nn.ELU()
    if activation == "selu":
        return nn.SELU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "crelu":
        return nn.ReLU()
    if activation == "lrelu":
        return nn.LeakyReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "identity":
        return nn.Identity()
    raise ValueError(f"Invalid activation function '{activation}'.")


def _feature_mlp(input_dim: int, hidden_dims: Sequence[int], activation: str) -> nn.Module:
    if len(hidden_dims) == 0:
        return nn.Identity()

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(_activation_module(activation))
        prev_dim = hidden_dim
    return nn.Sequential(*layers)


def _head_mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int], activation: str) -> nn.Module:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(_activation_module(activation))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class SplitLegArmActor(nn.Module):
    """Actor with a shared post-fusion encoder and separate leg/arm action heads."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        shared_hidden_dims: Sequence[int],
        leg_head_hidden_dims: Sequence[int],
        arm_head_hidden_dims: Sequence[int],
        activation: str,
        leg_action_dim: int = 12,
    ) -> None:
        super().__init__()
        if not 0 < leg_action_dim < num_actions:
            raise ValueError(f"leg_action_dim must be in (0, {num_actions}), got {leg_action_dim}.")

        self.leg_action_dim = leg_action_dim
        self.arm_action_dim = num_actions - leg_action_dim
        self.shared_encoder = _feature_mlp(input_dim, shared_hidden_dims, activation)
        shared_output_dim = shared_hidden_dims[-1] if len(shared_hidden_dims) > 0 else input_dim
        self.leg_head = _head_mlp(shared_output_dim, self.leg_action_dim, leg_head_hidden_dims, activation)
        self.arm_head = _head_mlp(shared_output_dim, self.arm_action_dim, arm_head_hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.shared_encoder(obs)
        return torch.cat([self.leg_head(latent), self.arm_head(latent)], dim=-1)


class PrivilegedTeacherActorCritic(nn.Module):
    """Legacy rsl_rl 3.x actor-critic with a privileged encoder for Go2Arm."""

    is_recurrent = False

    # Observation dimensions defined by the Go2Arm env configuration.
    POLICY_OBS_DIM = 92
    PRIVILEGED_OBS_DIM = 56
    CRITIC_EXTRA_OBS_DIM = 1

    def __init__(
        self,
        num_actor_obs: int | object,
        num_critic_obs: int | object,
        num_actions: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str,
        init_noise_std: float | Sequence[float] | torch.Tensor,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        privileged_encoder_hidden_dims: list[int] | None = None,
        privileged_feature_dim: int = 32,
        actor_shared_hidden_dims: list[int] | None = None,
        actor_leg_head_hidden_dims: list[int] | None = None,
        actor_arm_head_hidden_dims: list[int] | None = None,
        leg_action_dim: int = 12,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.actor_group_names = ("policy", "privileged")
        self.critic_group_names = ("policy", "privileged", "critic_extra")

        if self._looks_like_obs_and_obs_groups(num_actor_obs, num_critic_obs):
            obs = num_actor_obs
            obs_groups = num_critic_obs
            self.actor_group_names = tuple(obs_groups.get("actor", self.actor_group_names))
            self.critic_group_names = tuple(obs_groups.get("critic", self.critic_group_names))
            num_actor_obs = self._resolve_obs_dim_from_groups(obs, self.actor_group_names)
            num_critic_obs = self._resolve_obs_dim_from_groups(obs, self.critic_group_names)
        else:
            num_actor_obs = self._resolve_obs_dim(num_actor_obs, self.actor_group_names, direct_key="actor")
            num_critic_obs = self._resolve_obs_dim(num_critic_obs, self.critic_group_names, direct_key="critic")

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.privileged_feature_dim = privileged_feature_dim

        if privileged_encoder_hidden_dims is None:
            privileged_encoder_hidden_dims = [128, 64]

        expected_actor_obs = self.POLICY_OBS_DIM + self.PRIVILEGED_OBS_DIM
        expected_critic_obs = expected_actor_obs + self.CRITIC_EXTRA_OBS_DIM
        if num_actor_obs != expected_actor_obs or num_critic_obs != expected_critic_obs:
            raise ValueError(
                "Unexpected Go2Arm observation dimensions for legacy privileged actor-critic: "
                f"actor={num_actor_obs}, critic={num_critic_obs}, "
                f"expected actor={expected_actor_obs}, critic={expected_critic_obs}."
            )

        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )

        self.actor_privileged_encoder = MLP(
            self.PRIVILEGED_OBS_DIM,
            privileged_feature_dim,
            privileged_encoder_hidden_dims,
            activation,
        )
        self.critic_privileged_encoder = MLP(
            self.PRIVILEGED_OBS_DIM,
            privileged_feature_dim,
            privileged_encoder_hidden_dims,
            activation,
        )

        if actor_shared_hidden_dims is None:
            actor_shared_hidden_dims = actor_hidden_dims[:-1] if len(actor_hidden_dims) > 1 else actor_hidden_dims
        if actor_leg_head_hidden_dims is None:
            actor_leg_head_hidden_dims = actor_hidden_dims[-1:] if len(actor_hidden_dims) > 1 else []
        if actor_arm_head_hidden_dims is None:
            actor_arm_head_hidden_dims = actor_leg_head_hidden_dims

        self.actor = SplitLegArmActor(
            self.POLICY_OBS_DIM + privileged_feature_dim,
            num_actions,
            shared_hidden_dims=actor_shared_hidden_dims,
            leg_head_hidden_dims=actor_leg_head_hidden_dims,
            arm_head_hidden_dims=actor_arm_head_hidden_dims,
            activation=activation,
            leg_action_dim=leg_action_dim,
        )
        self.critic = MLP(
            self.POLICY_OBS_DIM + privileged_feature_dim + self.CRITIC_EXTRA_OBS_DIM,
            1,
            critic_hidden_dims,
            activation,
        )

        self.std = nn.Parameter(self._resolve_init_std(init_noise_std, num_actions))
        Normal.set_default_validate_args(False)
        self.distribution: Normal | None = None

    def reset(self, dones=None) -> None:
        del dones

    def update_normalization(self, observations) -> None:
        """Compatibility hook for newer rsl_rl runners."""
        with torch.no_grad():
            if hasattr(self.actor_obs_normalizer, "update"):
                actor_obs = self._flatten_obs(observations, self.actor_group_names, direct_key="actor")
                self.actor_obs_normalizer.update(actor_obs)

            if hasattr(self.critic_obs_normalizer, "update"):
                critic_obs = self._flatten_obs(observations, self.critic_group_names, direct_key="critic")
                self.critic_obs_normalizer.update(critic_obs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev  # type: ignore[union-attr]

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)  # type: ignore[union-attr]

    def update_distribution(self, observations) -> None:
        actor_obs = self._prepare_actor_obs(observations)
        mean = self.actor(actor_obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    @staticmethod
    def _resolve_init_std(init_noise_std: float | Sequence[float] | torch.Tensor, num_actions: int) -> torch.Tensor:
        init_std_tensor = torch.as_tensor(init_noise_std, dtype=torch.float32)
        if init_std_tensor.ndim == 0:
            return init_std_tensor.repeat(num_actions)
        init_std_tensor = init_std_tensor.reshape(-1)
        if init_std_tensor.numel() != num_actions:
            raise ValueError(
                f"Expected init_noise_std to provide {num_actions} values, but got {init_std_tensor.numel()}."
            )
        return init_std_tensor

    def act(self, observations, **kwargs) -> torch.Tensor:
        del kwargs
        self.update_distribution(observations)
        return self.distribution.sample()  # type: ignore[union-attr]

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)  # type: ignore[union-attr]

    def act_inference(self, observations) -> torch.Tensor:
        actor_obs = self._prepare_actor_obs(observations)
        return self.actor(actor_obs)

    def evaluate(self, critic_observations, **kwargs) -> torch.Tensor:
        del kwargs
        critic_obs = self._prepare_critic_obs(critic_observations)
        return self.critic(critic_obs)

    def as_jit(self) -> nn.Module:
        """返回 TorchScript 导出包装。

        输入约定保持为训练时的原始 teacher actor 观测，也就是 `policy + privileged` 拼接后的 148 维向量。
        """
        return _TorchPrivilegedTeacherActorExporter(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """返回 ONNX 导出包装。

        输入约定同样保持为训练时的原始 teacher actor 观测，避免导出时走错通用导出器的输入假设。
        """
        return _OnnxPrivilegedTeacherActorExporter(self, verbose)

    def _prepare_actor_obs(self, observations) -> torch.Tensor:
        obs = self._flatten_obs(observations, self.actor_group_names, direct_key="actor")
        obs = self.actor_obs_normalizer(obs)
        policy_obs = obs[:, : self.POLICY_OBS_DIM]
        privileged_obs = obs[:, self.POLICY_OBS_DIM :]
        privileged_latent = self.actor_privileged_encoder(privileged_obs)
        return torch.cat([policy_obs, privileged_latent], dim=-1)

    def _prepare_critic_obs(self, observations) -> torch.Tensor:
        obs = self._flatten_obs(observations, self.critic_group_names, direct_key="critic")
        obs = self.critic_obs_normalizer(obs)
        policy_obs = obs[:, : self.POLICY_OBS_DIM]
        privileged_start = self.POLICY_OBS_DIM
        privileged_end = privileged_start + self.PRIVILEGED_OBS_DIM
        privileged_obs = obs[:, privileged_start:privileged_end]
        critic_extra_obs = obs[:, privileged_end:]
        privileged_latent = self.critic_privileged_encoder(privileged_obs)
        return torch.cat([policy_obs, privileged_latent, critic_extra_obs], dim=-1)

    def _flatten_obs(self, observations, group_names: tuple[str, ...], direct_key: str) -> torch.Tensor:
        if isinstance(observations, torch.Tensor):
            return observations
        if isinstance(observations, Mapping):
            obs_parts = [
                self._coerce_obs_value_to_tensor(observations[group_name], group_name)
                for group_name in group_names
                if group_name in observations
            ]
            if obs_parts:
                return torch.cat(obs_parts, dim=-1)
            if direct_key in observations:
                return self._coerce_obs_value_to_tensor(observations[direct_key], direct_key)
        raise TypeError(f"Unsupported observation type for PrivilegedTeacherActorCritic: {type(observations)!r}")

    @staticmethod
    def _resolve_obs_dim(observations, group_names: tuple[str, ...], direct_key: str) -> int:
        """兼容新旧 rsl_rl：既支持直接传入维度，也支持传入 observation 容器。"""

        if isinstance(observations, int):
            return observations
        if isinstance(observations, torch.Size):
            if len(observations) == 0:
                raise ValueError(f"Observation shape for '{direct_key}' is empty.")
            return int(observations[-1])
        if isinstance(observations, torch.Tensor):
            if observations.ndim == 0:
                return int(observations.item())
            return int(observations.shape[-1])
        if isinstance(observations, (list, tuple)):
            obs_tensor = PrivilegedTeacherActorCritic._coerce_obs_value_to_tensor(observations, direct_key)
            return int(obs_tensor.shape[-1])
        if isinstance(observations, Mapping):
            obs_parts = [
                PrivilegedTeacherActorCritic._coerce_obs_value_to_tensor(observations[group_name], group_name)
                for group_name in group_names
                if group_name in observations
            ]
            if obs_parts:
                return int(sum(part.shape[-1] for part in obs_parts))
            if direct_key in observations:
                return int(
                    PrivilegedTeacherActorCritic._coerce_obs_value_to_tensor(
                        observations[direct_key], direct_key
                    ).shape[-1]
                )
        raise TypeError(
            f"Unsupported observation specification for '{direct_key}' in PrivilegedTeacherActorCritic: "
            f"{type(observations)!r}"
        )

    @staticmethod
    def _coerce_obs_value_to_tensor(value, value_name: str) -> torch.Tensor:
        """把旧版接口里的 observation 值统一展平成单个 2D tensor。"""

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            tensor_parts = [
                PrivilegedTeacherActorCritic._coerce_obs_value_to_tensor(item, value_name) for item in value
            ]
            if not tensor_parts:
                raise ValueError(f"Observation value '{value_name}' is an empty list/tuple.")
            return torch.cat(tensor_parts, dim=-1)
        raise TypeError(
            f"Unsupported observation value type for '{value_name}' in PrivilegedTeacherActorCritic: {type(value)!r}"
        )

    @staticmethod
    def _resolve_obs_dim_from_groups(observations, group_names: tuple[str, ...]) -> int:
        """按 observation groups 从观测容器里解析拼接后的总维度。"""

        if not isinstance(observations, Mapping):
            raise TypeError(
                "Observation container must be a mapping when resolving dimensions from observation groups. "
                f"Received: {type(observations)!r}"
            )
        obs_parts = [
            PrivilegedTeacherActorCritic._coerce_obs_value_to_tensor(observations[group_name], group_name)
            for group_name in group_names
            if group_name in observations
        ]
        if not obs_parts:
            raise ValueError(f"No observation groups found for {group_names!r}.")
        return int(sum(part.shape[-1] for part in obs_parts))

    @staticmethod
    def _looks_like_obs_and_obs_groups(obs_candidate, obs_groups_candidate) -> bool:
        """判断是否是新版 model-style 构造协议：(obs, obs_groups, ...)。"""

        if not isinstance(obs_candidate, Mapping) or not isinstance(obs_groups_candidate, Mapping):
            return False
        actor_groups = obs_groups_candidate.get("actor")
        critic_groups = obs_groups_candidate.get("critic")
        return isinstance(actor_groups, (list, tuple)) and isinstance(critic_groups, (list, tuple))


class _ExportPrivilegedTeacherActorBase(nn.Module):
    """legacy privileged-teacher actor 的导出基类。

    这里单独封装导出逻辑，而不是直接复用 IsaacLab 的通用导出器。
    原因是当前模型的 `actor` 只是最终的 actor head，它期望的输入是
    `policy_obs + privileged_latent`，也就是 124 维编码后特征；
    但训练时保存下来的 `actor_obs_normalizer` 处理的是编码前的 148 维原始观测。
    因此导出时必须先做“原始观测归一化 + privileged encoder”，再送入 actor head。
    """

    def __init__(self, model: PrivilegedTeacherActorCritic) -> None:
        super().__init__()
        # 复制 actor 侧的原始观测归一化器，保证导出模型与训练时使用同一套统计量。
        self.actor_obs_normalizer = copy.deepcopy(model.actor_obs_normalizer)
        # 复制 privileged encoder，保证导出模型内部仍然从原始 privileged 观测构造 latent。
        self.actor_privileged_encoder = copy.deepcopy(model.actor_privileged_encoder)
        # 复制最终 actor head，它真正消费的是 `policy_obs + privileged_latent`。
        self.actor = copy.deepcopy(model.actor)
        # 记录公共观测维度，后面按固定切片位置拆分原始输入。
        self.policy_obs_dim = model.POLICY_OBS_DIM
        # 记录原始 privileged 观测维度，后面用于切出 privileged 片段。
        self.privileged_obs_dim = model.PRIVILEGED_OBS_DIM
        # 导出模型的输入维度定义为原始 actor 观测总维度，也就是 148。
        self.input_size = model.num_actor_obs

    def _build_latent(self, x: torch.Tensor) -> torch.Tensor:
        # 先对编码前的原始 actor 观测做归一化，这一步必须与训练完全一致。
        x = self.actor_obs_normalizer(x)
        # 取出原始公共观测部分；这一段会直接保留到最终 latent 中。
        policy_obs = x[:, : self.policy_obs_dim]
        # 取出原始 privileged 观测部分；这一段需要先经过 privileged encoder。
        privileged_obs = x[:, self.policy_obs_dim : self.policy_obs_dim + self.privileged_obs_dim]
        # 把原始 privileged 观测压缩成固定维度的 latent 特征。
        privileged_latent = self.actor_privileged_encoder(privileged_obs)
        # 按训练时 actor head 的真实输入格式拼接成 124 维 latent。
        return torch.cat([policy_obs, privileged_latent], dim=-1)


class _TorchPrivilegedTeacherActorExporter(_ExportPrivilegedTeacherActorBase):
    """legacy privileged-teacher actor 的 TorchScript 导出包装。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先把原始 148 维观测变成 actor head 需要的 124 维 latent。
        latent = self._build_latent(x)
        # 再执行最终 actor head，输出动作均值。
        return self.actor(latent)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxPrivilegedTeacherActorExporter(_ExportPrivilegedTeacherActorBase):
    """legacy privileged-teacher actor 的 ONNX 导出包装。"""

    is_recurrent: bool = False

    def __init__(self, model: PrivilegedTeacherActorCritic, verbose: bool) -> None:
        super().__init__(model)
        # 保留 verbose 配置，接口形式与 IsaacLab 现有 ONNX 导出包装保持一致。
        self.verbose = verbose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ONNX 导出时的前向流程与 TorchScript 完全一致。
        latent = self._build_latent(x)
        return self.actor(latent)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

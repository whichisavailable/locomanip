# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories
from tensordict import TensorDict

from robot_lab.tasks.manager_based.locomotion.velocity.config.locomanip.go2arm.agents.rsl_rl_compat import (
    MLP,
    EmpiricalNormalization,
    HiddenState,
)


class PrivilegedEncoderMLPModel(MLPModel):
    """带特权编码器的 MLP 模型。

    这个模型用于 go2arm 的 teacher 阶段：
    1. 把共享观测、本体感知以外的原始特权观测、critic 额外项分别取出。
    2. 原始特权观测先经过一个独立的 MLP，压缩成固定 32 维特征。
    3. 再把共享观测、32 维特权特征、critic 额外项拼接后送入主干网络。

    注意：
    - actor 和 critic 共用这个模型类，但通过不同 group 名称控制输入组成。
    - 累积误差不进入特权编码器，而是作为 critic 的额外输入直接拼接。
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        proprio_group_names: tuple[str, ...] | list[str] = ("policy",),
        privileged_group_names: tuple[str, ...] | list[str] = ("privileged",),
        extra_group_names: tuple[str, ...] | list[str] = (),
        privileged_encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        privileged_feature_dim: int = 32,
    ) -> None:
        # 不走父类默认初始化，因为这里的 latent 构造逻辑与普通 MLPModel 不同。
        nn.Module.__init__(self)

        # 保存 observation set 名称和分组配置，方便调试和导出。
        self.obs_set = obs_set
        self.obs_groups = obs_groups

        # 统一转成 list，便于后面循环处理。
        self.proprio_group_names = list(proprio_group_names)
        self.privileged_group_names = list(privileged_group_names)
        self.extra_group_names = list(extra_group_names)

        # 计算三类原始输入的维度。
        self.proprio_dim = self._get_group_dim(obs, self.proprio_group_names)
        self.privileged_dim = self._get_group_dim(obs, self.privileged_group_names)
        self.extra_dim = self._get_group_dim(obs, self.extra_group_names)

        # 归一化器看到的是编码前的原始输入。
        self.obs_dim = self.proprio_dim + self.privileged_dim + self.extra_dim
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        # 特权编码器只处理原始特权观测，不碰 critic 额外项。
        self.privileged_feature_dim = privileged_feature_dim if self.privileged_dim > 0 else 0
        if self.privileged_dim > 0:
            self.privileged_encoder = MLP(
                self.privileged_dim,
                self.privileged_feature_dim,
                privileged_encoder_hidden_dims,
                activation,
            )
        else:
            self.privileged_encoder = nn.Identity()

        # 主干网络输入维度 = 共享观测 + 编码后的特权特征 + critic 额外项。
        self.latent_dim = self.proprio_dim + self.privileged_feature_dim + self.extra_dim

        # 分布头配置与普通 MLPModel 保持一致。
        self.distribution_cfg = copy.deepcopy(distribution_cfg)
        if distribution_cfg is not None:
            distribution_cfg = copy.deepcopy(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # 创建最终 actor / critic 主干。
        self.mlp = MLP(self.latent_dim, mlp_output_dim, hidden_dims, activation)

        # 如果有分布头，则沿用 rsl_rl 默认的分布头初始化方式。
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """前向传播。"""

        # 如果来自 RNN 流程的是 padding 过的观测，而当前模型本身不是 RNN，则先去掉 padding。
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # 构造编码后的 latent。
        latent = self.get_latent(obs, masks, hidden_state)
        # 送入主干网络。
        mlp_output = self.mlp(latent)
        # 如果配置了分布头，则按需采样；否则直接返回确定性输出。
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """构造最终送入主干网络的 latent。"""

        # 先按三类取出原始观测。
        proprio_obs = self._concat_groups(obs, self.proprio_group_names)
        privileged_obs = self._concat_groups(obs, self.privileged_group_names)
        extra_obs = self._concat_groups(obs, self.extra_group_names)

        # 归一化作用在编码前的原始输入上。
        raw_obs_parts = [part for part in (proprio_obs, privileged_obs, extra_obs) if part is not None]
        if len(raw_obs_parts) > 0:
            normalized_raw_obs = self.obs_normalizer(torch.cat(raw_obs_parts, dim=-1))
        else:
            batch_size = next(iter(obs.values())).shape[0]
            normalized_raw_obs = torch.zeros((batch_size, 0), device=next(iter(obs.values())).device)

        # 再按原始维度切回三类。
        start = 0
        proprio_obs = normalized_raw_obs[:, start : start + self.proprio_dim]
        start += self.proprio_dim
        privileged_obs = normalized_raw_obs[:, start : start + self.privileged_dim]
        start += self.privileged_dim
        extra_obs = normalized_raw_obs[:, start : start + self.extra_dim]

        # 原始特权先编码成固定维度，再与其它输入拼接。
        latent_parts = [proprio_obs]
        if self.privileged_dim > 0:
            latent_parts.append(self.privileged_encoder(privileged_obs))
        if self.extra_dim > 0:
            latent_parts.append(extra_obs)
        return torch.cat(latent_parts, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """更新经验归一化器。"""

        # 未启用归一化时直接返回。
        if not self.obs_normalization:
            return

        # 用编码前的原始输入更新统计量。
        raw_obs_parts = []
        for group_names in (self.proprio_group_names, self.privileged_group_names, self.extra_group_names):
            part = self._concat_groups(obs, group_names)
            if part is not None:
                raw_obs_parts.append(part)
        if len(raw_obs_parts) == 0:
            return
        self.obs_normalizer.update(torch.cat(raw_obs_parts, dim=-1))  # type: ignore[arg-type]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """当前模型不是 RNN，所以无需重置隐藏状态。"""

    def get_hidden_state(self) -> HiddenState:
        """当前模型没有隐藏状态。"""

        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """当前模型没有隐藏状态，因此这里为空实现。"""

    def as_jit(self) -> nn.Module:
        """返回可用于 TorchScript 导出的包装模型。"""

        return _TorchPrivilegedEncoderMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """返回可用于 ONNX 导出的包装模型。"""

        return _OnnxPrivilegedEncoderMLPModel(self, verbose)

    def _get_group_dim(self, obs: TensorDict, group_names: list[str]) -> int:
        """统计若干观测组拼接后的总维度。"""

        total_dim = 0
        for group_name in group_names:
            if group_name not in obs:
                raise ValueError(
                    f"Observation group '{group_name}' not found in environment outputs. "
                    f"Available groups: {list(obs.keys())}"
                )
            if len(obs[group_name].shape) != 2:
                raise ValueError(
                    f"PrivilegedEncoderMLPModel 只支持 1D 观测，但 '{group_name}' 的形状是 {obs[group_name].shape}。"
                )
            total_dim += obs[group_name].shape[-1]
        return total_dim

    def _concat_groups(self, obs: TensorDict, group_names: list[str]) -> torch.Tensor | None:
        """按给定顺序把若干观测组拼接起来。"""

        if len(group_names) == 0:
            return None
        obs_parts = [obs[group_name] for group_name in group_names]
        return torch.cat(obs_parts, dim=-1)


class _ExportPrivilegedEncoderMLPModelBase(nn.Module):
    """特权编码模型的导出基类。

    导出时的输入约定：
    - 输入是当前 observation set 对应的“原始拼接观测”，也就是编码前的扁平向量。
    - actor 导出时对应 `policy + privileged`
    - critic 导出时对应 `policy + privileged + critic_extra`
    """

    def __init__(self, model: PrivilegedEncoderMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.privileged_encoder = copy.deepcopy(model.privileged_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.proprio_dim = model.proprio_dim
        self.privileged_dim = model.privileged_dim
        self.extra_dim = model.extra_dim
        self.input_size = model.obs_dim

    def _build_latent(self, x: torch.Tensor) -> torch.Tensor:
        """把扁平原始输入转成主干网络需要的 latent。"""

        # 先用训练时同样的归一化器处理原始输入。
        x = self.obs_normalizer(x)

        # 按原始维度切出三类输入。
        start = 0
        proprio_obs = x[:, start : start + self.proprio_dim]
        start += self.proprio_dim
        privileged_obs = x[:, start : start + self.privileged_dim]
        start += self.privileged_dim
        extra_obs = x[:, start : start + self.extra_dim]

        # 先编码原始特权，再和其它部分拼接。
        latent_parts = [proprio_obs]
        if self.privileged_dim > 0:
            latent_parts.append(self.privileged_encoder(privileged_obs))
        if self.extra_dim > 0:
            latent_parts.append(extra_obs)
        return torch.cat(latent_parts, dim=-1)


class _TorchPrivilegedEncoderMLPModel(_ExportPrivilegedEncoderMLPModelBase):
    """用于 TorchScript 导出的包装。"""

    def __init__(self, model: PrivilegedEncoderMLPModel) -> None:
        super().__init__(model)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对已经扁平化的原始输入做确定性推理。"""

        latent = self._build_latent(x)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """与普通导出模型保持一致，这里为空实现。"""


class _OnnxPrivilegedEncoderMLPModel(_ExportPrivilegedEncoderMLPModelBase):
    """用于 ONNX 导出的包装。"""

    is_recurrent: bool = False

    def __init__(self, model: PrivilegedEncoderMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对已经扁平化的原始输入做确定性推理。"""

        latent = self._build_latent(x)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """返回 ONNX tracing 需要的示例输入。"""

        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """ONNX 输入名。"""

        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """ONNX 输出名。"""

        return ["actions"]

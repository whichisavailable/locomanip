# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn

try:
    # 优先复用当前环境里 rsl_rl 已经提供的实现，避免重复定义。
    from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
except ImportError:
    try:
        # 兼容部分版本：类存在于子模块，但没有从 modules 顶层重新导出。
        from rsl_rl.modules.mlp import MLP
        from rsl_rl.modules.normalization import EmpiricalNormalization
        from rsl_rl.modules.rnn import HiddenState
    except ImportError:
        # 再向下兼容更老版本：在 robot_lab 内提供与当前需求等价的最小实现。
        HiddenState = tuple[torch.Tensor, ...] | torch.Tensor | None

        def _resolve_nn_activation(act_name: str | None) -> nn.Module:
            """把字符串激活函数名解析成 torch 模块。"""

            if act_name is None:
                return nn.Identity()
            act_dict: dict[str, nn.Module] = {
                "elu": nn.ELU(),
                "selu": nn.SELU(),
                "relu": nn.ReLU(),
                "crelu": nn.ReLU(),
                "lrelu": nn.LeakyReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(),
            }
            key = act_name.lower()
            if key not in act_dict:
                raise ValueError(
                    f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}"
                )
            return act_dict[key]

        class MLP(nn.Sequential):
            """兼容旧版 rsl_rl 缺失 MLP 时的等价前馈网络实现。"""

            def __init__(
                self,
                input_dim: int,
                output_dim: int | tuple[int, ...] | list[int],
                hidden_dims: tuple[int, ...] | list[int],
                activation: str = "elu",
                last_activation: str | None = None,
            ) -> None:
                super().__init__()

                activation_mod = _resolve_nn_activation(activation)
                last_activation_mod = _resolve_nn_activation(last_activation) if last_activation is not None else None
                hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

                layers: list[nn.Module] = []
                layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
                layers.append(activation_mod)

                for layer_index in range(len(hidden_dims_processed) - 1):
                    layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
                    layers.append(activation_mod)

                if isinstance(output_dim, int):
                    layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
                else:
                    total_out_dim = reduce(lambda x, y: x * y, output_dim)
                    layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
                    layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

                if last_activation_mod is not None:
                    layers.append(last_activation_mod)

                for idx, layer in enumerate(layers):
                    self.add_module(f"{idx}", layer)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """顺序执行每一层，与新版 rsl_rl.modules.MLP 保持一致。"""

                for layer in self:
                    x = layer(x)
                return x

        class EmpiricalNormalization(nn.Module):
            """兼容旧版 rsl_rl 缺失归一化模块时的等价实现。"""

            def __init__(
                self,
                shape: int | tuple[int, ...] | list[int],
                eps: float = 1e-2,
                until: int | None = None,
            ) -> None:
                super().__init__()
                self.eps = eps
                self.until = until
                self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
                self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
                self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
                self.register_buffer("count", torch.tensor(0, dtype=torch.long))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """按经验均值和方差做归一化。"""

                return (x - self._mean) / (self._std + self.eps)

            @torch.jit.unused
            def update(self, x: torch.Tensor) -> None:
                """更新经验统计量。"""

                if not self.training:
                    return
                if self.until is not None and self.count >= self.until:
                    return

                count_x = x.shape[0]
                self.count += count_x
                rate = count_x / self.count
                var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
                mean_x = torch.mean(x, dim=0, keepdim=True)
                delta_mean = mean_x - self._mean
                self._mean += rate * delta_mean
                self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
                self._std = torch.sqrt(self._var)

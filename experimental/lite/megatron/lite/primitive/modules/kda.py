# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kimi Delta Attention composed from Lite tensor-parallel primitives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.lite.primitive.ops.kda import kda
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
)
from megatron.lite.primitive.utils import ensure_divisible

try:
    from fla.modules.convolution import (
        causal_conv1d as _fla_causal_conv1d,  # pyright: ignore[reportMissingImports]
    )

    _HAS_FLA_CONV = True
except ImportError:
    _HAS_FLA_CONV = False


class KimiDeltaAttention(nn.Module):
    """Trainable KDA with TP-sharded heads and K3's full-rank output gate."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        short_conv_kernel_size: int,
        gate_lower_bound: float,
        rms_norm_eps: float,
        ps: ParallelState,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.ps = ps
        self.num_heads = num_heads
        self.num_heads_local = ensure_divisible(num_heads, ps.tp_size)
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.projection_size_local = self.num_heads_local * head_dim
        self.short_conv_kernel_size = short_conv_kernel_size
        self.gate_lower_bound = gate_lower_bound
        self.deterministic = bool(deterministic)

        self.q_proj = ColumnParallelLinear(
            hidden_size, self.projection_size, ps, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, self.projection_size, ps, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, self.projection_size, ps, bias=False
        )
        self.q_conv1d = self._make_conv()
        self.k_conv1d = self._make_conv()
        self.v_conv1d = self._make_conv()
        self.A_log = nn.Parameter(
            torch.zeros(self.num_heads_local, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.zeros(
                self.num_heads_local,
                head_dim,
                dtype=torch.float32,
            )
        )
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = ColumnParallelLinear(
            head_dim, self.projection_size, ps, bias=False
        )
        self.b_proj = ColumnParallelLinear(hidden_size, num_heads, ps, bias=False)
        self.g_proj = ColumnParallelLinear(
            hidden_size, self.projection_size, ps, bias=False
        )
        self.o_norm = te.RMSNorm(head_dim, eps=rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.projection_size, hidden_size, ps, bias=False
        )

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse=recurse)
        for parameter in (self.A_log, self.dt_bias):
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
        return module

    def _make_conv(self) -> nn.Conv1d:
        return nn.Conv1d(
            self.projection_size_local,
            self.projection_size_local,
            self.short_conv_kernel_size,
            groups=self.projection_size_local,
            bias=False,
        )

    def _causal_conv(self, values: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Apply the existing FLA causal-convolution primitive when available."""
        if _HAS_FLA_CONV and not self.deterministic:
            output, _ = _fla_causal_conv1d(
                x=values,
                weight=conv.weight.squeeze(1),
                bias=None,
                activation="silu",
            )
            return output
        channels_first = values.transpose(1, 2)
        channels_first = F.pad(channels_first, (self.short_conv_kernel_size - 1, 0))
        return F.silu(conv(channels_first).transpose(1, 2))

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        return values.view(
            *values.shape[:-1],
            self.num_heads_local,
            self.head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        del position_ids
        if packed_seq_params is not None:
            raise NotImplementedError(
                "KimiDeltaAttention packed THD execution is not validated yet."
            )

        # Lite model activations are [sequence, batch, hidden]. The KDA and
        # causal-convolution operators consume [batch, sequence, ...].
        q = self.q_proj(x).transpose(0, 1).contiguous()
        k = self.k_proj(x).transpose(0, 1).contiguous()
        v = self.v_proj(x).transpose(0, 1).contiguous()
        q = self._heads(self._causal_conv(q, self.q_conv1d))
        k = self._heads(self._causal_conv(k, self.k_conv1d))
        v = self._heads(self._causal_conv(v, self.v_conv1d))

        feature_gate = self._heads(
            self.f_b_proj(self.f_a_proj(x)).transpose(0, 1).contiguous()
        )
        beta = self.b_proj(x).transpose(0, 1).contiguous()
        output, _ = kda(
            q,
            k,
            v,
            feature_gate,
            beta,
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            output_final_state=False,
            scale=self.head_dim**-0.5,
        )
        output_gate = torch.sigmoid(
            self._heads(self.g_proj(x).transpose(0, 1).contiguous())
        )
        output = self.o_norm(output) * output_gate
        output = output.flatten(-2).transpose(0, 1).contiguous()
        return self.o_proj(output)


__all__ = ["KimiDeltaAttention"]

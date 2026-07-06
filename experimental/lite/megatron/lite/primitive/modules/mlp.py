"""Model-agnostic tensor-parallel SwiGLU MLP primitives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
)


class SwiGLUMLP(nn.Module):
    """Bias-free tensor-parallel SwiGLU MLP.

    This primitive is suitable for both dense decoder MLPs and always-on
    shared experts.  Optional output gating belongs in a separate composition
    module because it is not part of the SwiGLU operation itself.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ps: ParallelState,
    ) -> None:
        super().__init__()
        self.gate_up = ColumnParallelLinear(
            hidden_size,
            intermediate_size * 2,
            ps,
            bias=False,
        )
        self.down = RowParallelLinear(
            intermediate_size,
            hidden_size,
            ps,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


__all__ = ["SwiGLUMLP"]

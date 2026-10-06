# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
import torch.nn as nn

from megatron.lite.primitive.modules.experts import swiglu_with_probs


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = float(swiglu_limit or 0.0)

    @classmethod
    def from_projections(cls, w1, w2, w3, *, swiglu_limit=0.0):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.w1, model.w2, model.w3 = w1, w2, w3
        model.swiglu_limit = swiglu_limit
        return model

    def forward(self, x: torch.Tensor, weights=None) -> torch.Tensor:
        if hasattr(self, 'gate_up'):
            gate_up, down = self.gate_up(x), self.down
        else:
            gate_up = torch.cat((self.w1(x).float(), self.w3(x).float()), dim=-1)
            down = self.w2
        y = swiglu_with_probs(gate_up, weights, self.swiglu_limit)
        return down(y.to(dtype=x.dtype))

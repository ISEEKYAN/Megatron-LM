# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.primitive.precision import (
    PrecisionCoverage,
    PrecisionImplementation,
    PrecisionPhase,
    PrimitiveCapability,
    SemanticSite,
    active_precision,
    precision_site_forward_context,
)


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        swiglu_limit: float = 0.0,
        precision_coverage: PrecisionCoverage | None = None,
    ):
        super().__init__()
        self._precision_implementation = self._bind_bf16_weight_precision(
            precision_coverage,
            dimensions=(hidden_size, 2 * intermediate_size, intermediate_size),
        )
        self.gate_up = te.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        self.down = te.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        if precision_coverage is not None:
            precision_coverage.claim(
                self.gate_up,
                SemanticSite.DENSE_MLP,
                PrimitiveCapability.TE_LINEAR,
            )
            precision_coverage.claim(
                self.down,
                SemanticSite.DENSE_MLP,
                PrimitiveCapability.TE_LINEAR,
            )
        self.swiglu_limit = float(swiglu_limit or 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self._precision_context(x):
            gate_up = self.gate_up(x)
        y = swiglu_with_probs(gate_up, None, self.swiglu_limit)
        with self._precision_context(y):
            return self.down(y.to(dtype=x.dtype))

    @staticmethod
    def _bind_bf16_weight_precision(
        coverage: PrecisionCoverage | None,
        *,
        dimensions: tuple[int, ...],
    ) -> PrecisionImplementation | None:
        if coverage is None:
            return None
        implementation = active_precision(PrecisionPhase.MODEL_INIT)
        if coverage.implementation is not implementation:
            raise RuntimeError("precision coverage is bound to a different implementation")
        for value in dimensions:
            if value % 128 != 0:
                raise ValueError(
                    "Dense MLP features must be divisible by 128 for Hopper "
                    f"blockwise FP8; got {value}."
                )
        return implementation

    def _precision_context(self, x: torch.Tensor):
        if self._precision_implementation is None:
            return nullcontext()
        if x.dim() < 2 or x.shape[-1] % 128 != 0:
            raise ValueError(
                "dense_mlp input must have rank at least 2 and a last dimension "
                "divisible by 128"
            )
        rows = x.numel() // x.shape[-1]
        if rows % 128 != 0:
            raise ValueError(
                f"dense_mlp input row product must be divisible by 128; got {rows}."
            )
        return precision_site_forward_context(
            self._precision_implementation, SemanticSite.DENSE_MLP
        )

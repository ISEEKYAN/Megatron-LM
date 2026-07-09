# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mixed-precision policy for the MLite-owned M-FSDP buffers.

This is the BF16/FP16 closure of MCore's Megatron-FSDP policy.  FP8 cache and
quantized initialization belong to a separate primitive and are intentionally
outside this optimizer's supported surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MixedPrecisionPolicy:
    compute_dtype: torch.dtype
    main_params_dtype: torch.dtype = torch.float32
    main_grads_dtype: torch.dtype = torch.float32
    grad_comm_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        for name in (
            "compute_dtype",
            "main_params_dtype",
            "main_grads_dtype",
            "grad_comm_dtype",
        ):
            dtype = getattr(self, name)
            if not torch.empty((), dtype=dtype).is_floating_point():
                raise ValueError(f"M-FSDP {name} must be floating point, got {dtype}.")


__all__ = ["MixedPrecisionPolicy"]

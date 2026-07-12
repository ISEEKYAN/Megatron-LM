# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]


def ensure_divisible(numerator: int, denominator: int, msg: str = "") -> int:
    if numerator % denominator != 0:
        detail = f" ({msg})" if msg else ""
        raise ValueError(f"{numerator} is not divisible by {denominator}{detail}")
    return numerator // denominator


def log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[megatron.lite] {msg}", flush=True)


__all__ = ["ensure_divisible", "log_rank0"]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-FSDP optimizer primitive package."""

from __future__ import annotations

from megatron.lite.primitive.optimizers.mfsdp.optimizer import (
    build_mfsdp_training_optimizer,
)

__all__ = [
    "build_mfsdp_training_optimizer",
]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optimizer backend registry."""

from __future__ import annotations

import importlib

from megatron.lite.primitive.optimizers.runtime_adapter import DEFAULT_RUNTIME_ADAPTER

BACKENDS = {
    "dist_opt": "megatron.lite.primitive.optimizers.megatron_wrap",
    "fsdp2": "megatron.lite.primitive.optimizers.fsdp2",
    "mfsdp": "megatron.lite.primitive.optimizers.mfsdp.backend",
}


def get_optimizer_backend(name: str | None):
    if name in (None, "none"):
        return DEFAULT_RUNTIME_ADAPTER
    if name not in BACKENDS:
        raise ValueError(f"Unknown Megatron Lite optimizer backend: {name!r}.")
    return importlib.import_module(BACKENDS[name]).BACKEND


__all__ = ["get_optimizer_backend"]

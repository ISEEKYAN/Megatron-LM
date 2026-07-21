# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optimizer backend registry."""

from __future__ import annotations

import importlib

# One entry per selectable optimizer backend. Registering "mfsdp" here (the
# minimal wiring the standalone M-FSDP optimizer needs) is what makes it
# selectable by name via ``get_optimizer_backend`` on the same footing as the
# existing dist_opt / fsdp2 backends; the implementation lives entirely under
# ``optimizers/mfsdp/``.
BACKENDS = {
    "dist_opt": "megatron.lite.primitive.optimizers.megatron_wrap",
    "fsdp2": "megatron.lite.primitive.optimizers.fsdp2",
    "mfsdp": "megatron.lite.primitive.optimizers.mfsdp.backend",
}


def get_optimizer_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown Megatron Lite optimizer backend: {name!r}.")
    return importlib.import_module(BACKENDS[name]).BACKEND


__all__ = ["get_optimizer_backend"]

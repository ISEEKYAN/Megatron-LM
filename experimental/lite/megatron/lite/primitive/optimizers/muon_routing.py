# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-native Muon parameter routing metadata for MLite models."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.protocols import ExpertClassifierFn


def tag_muon_parameter_metadata(
    model_chunks: Iterable[nn.Module],
    *,
    is_expert_param: ExpertClassifierFn | Callable[[str], bool],
) -> None:
    """Tag expert metadata before the upstream Muon buffer-routing pass.

    Modules own semantic metadata for their parameters: vocab primitives mark their exact
    embedding/output weight, and attention primitives mark their exact fused-QKV weight.
    This pass deliberately knows only the caller-provided expert classifier. Existing
    TP/EP and module-owned attributes remain untouched.
    """

    for model in model_chunks:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_expert_param(name):
                param.expert_tp = True


__all__ = ["tag_muon_parameter_metadata"]

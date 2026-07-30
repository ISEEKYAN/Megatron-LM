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
    """Tag Muon ownership and expert metadata before optimizer wrapping.

    Modules own semantic metadata for their parameters: vocab primitives mark their exact
    embedding/output weight, and attention primitives mark their exact fused-QKV weight.
    Matrix parameters other than embedding/output weights are owned by Muon; all other
    trainable parameters use the scalar-optimizer fallback. This pass deliberately knows
    only the caller-provided expert classifier. Existing TP/EP and module-owned attributes
    remain untouched.
    """

    for model in model_chunks:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.is_managed_by_layer_wise_optimizer = param.dim() == 2 and not getattr(
                param, "is_embedding_or_output_parameter", False
            )
            if is_expert_param(name):
                param.expert_tp = True


__all__ = ["tag_muon_parameter_metadata"]

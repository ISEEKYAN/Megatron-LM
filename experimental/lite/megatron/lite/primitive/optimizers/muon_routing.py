# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muon parameter-routing metadata shared by DistOpt and FSDP2."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch.nn as nn

from megatron.lite.primitive.protocols import ExpertClassifierFn

_EMBEDDING_OUTPUT_MODULES = frozenset({"VocabParallelEmbedding", "VocabParallelOutput"})


def is_managed_by_muon(param: nn.Parameter) -> bool:
    """Match Muon's matrix-only route, excluding embeddings and output weights."""
    return param.dim() == 2 and not getattr(param, "is_embedding_or_output_parameter", False)


def _tag_embedding_and_output_parameters(model: nn.Module) -> None:
    for module in model.modules():
        if type(module).__name__ not in _EMBEDDING_OUTPUT_MODULES:
            continue
        for param in module.parameters(recurse=True):
            param.is_embedding_or_output_parameter = True


def tag_muon_parameter_metadata(
    model_chunks: Iterable[nn.Module], *, is_expert_param: ExpertClassifierFn | Callable[[str], bool]
) -> None:
    """Tag routing before FSDP2 replaces parameters with DTensor shards."""
    for model in model_chunks:
        _tag_embedding_and_output_parameters(model)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_expert_param(name):
                param.expert_tp = True
            param.is_managed_by_layer_wise_optimizer = is_managed_by_muon(param)


__all__ = ["is_managed_by_muon", "tag_muon_parameter_metadata"]

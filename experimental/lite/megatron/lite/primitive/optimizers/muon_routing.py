# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-native Muon parameter routing metadata for MLite models."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.protocols import ExpertClassifierFn

_EMBEDDING_OUTPUT_MODULES = frozenset({"VocabParallelEmbedding", "VocabParallelOutput"})


def is_managed_by_muon(param: nn.Parameter) -> bool:
    """Mirror Megatron's Muon route: matrix weights except embeddings/outputs."""

    return param.dim() == 2 and not getattr(
        param, "is_embedding_or_output_parameter", False
    )


def _tag_embedding_and_output_parameters(model: nn.Module) -> None:
    # All five native model families assemble their token embedding and LM head from these
    # shared primitive types. Central type-name routing avoids model-protocol edits while
    # keeping this helper importable in CPU tests without importing Transformer Engine.
    for module in model.modules():
        if type(module).__name__ not in _EMBEDDING_OUTPUT_MODULES:
            continue
        for param in module.parameters(recurse=True):
            param.is_embedding_or_output_parameter = True


def _tag_fused_qkv_parameters(model: nn.Module) -> None:
    """Attach the same per-query-group QKV split metadata used by Megatron."""

    for module in model.modules():
        if type(module).__name__ != "GQAttention":
            continue
        num_q_heads = int(module.num_heads_local)
        num_kv_heads = int(module.num_kv_heads_local)
        head_dim = int(module.head_dim)
        if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
            continue
        q_projection_size = (num_q_heads // num_kv_heads) * head_dim
        if bool(getattr(module, "_output_gate", False)):
            split_shapes = [q_projection_size, q_projection_size, head_dim, head_dim]
        else:
            split_shapes = [q_projection_size, head_dim, head_dim]
        param = module.qkv.linear.weight
        if param.dim() == 2 and param.shape[0] % sum(split_shapes) == 0:
            param.is_qkv = True
            param.qkv_split_shapes = split_shapes


def tag_muon_parameter_metadata(
    model_chunks: Iterable[nn.Module],
    *,
    is_expert_param: ExpertClassifierFn | Callable[[str], bool],
) -> None:
    """Tag Muon routing, QKV splitting, and expert metadata before DDP wrapping.

    Existing TP/EP attributes are deliberately left untouched. The subsequent DistOpt
    metadata pass may fill missing attributes, but must continue to respect model-provided
    ``tensor_model_parallel``, ``sequence_parallel``, ``partition_*``, and ``allreduce``.
    """

    for model in model_chunks:
        _tag_embedding_and_output_parameters(model)
        _tag_fused_qkv_parameters(model)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_expert_param(name):
                param.expert_tp = True
            param.is_managed_by_layer_wise_optimizer = is_managed_by_muon(param)


__all__ = ["is_managed_by_muon", "tag_muon_parameter_metadata"]

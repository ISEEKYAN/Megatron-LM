# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reusable checkpoint tensor layout transforms without I/O dependencies."""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Generator, Mapping


def pack_grouped_query_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Pack separate HF Q/K/V weights into MCore grouped-query order."""
    q_per_group = num_attention_heads // num_key_value_heads
    query = query.view(num_key_value_heads, q_per_group * head_dim, -1)
    key = key.view(num_key_value_heads, head_dim, -1)
    value = value.view(num_key_value_heads, head_dim, -1)
    return torch.cat([query, key, value], dim=1).reshape(-1, query.shape[-1]).contiguous()


def unpack_grouped_query_qkv(
    tensor: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack an MCore grouped-query QKV weight into separate HF tensors."""
    q_per_group = num_attention_heads // num_key_value_heads
    group_width = (q_per_group + 2) * head_dim
    packed = tensor.view(num_key_value_heads, group_width, -1)
    q_end = q_per_group * head_dim
    k_end = q_end + head_dim
    query = packed[:, :q_end].reshape(num_attention_heads * head_dim, -1)
    key = packed[:, q_end:k_end].reshape(num_key_value_heads * head_dim, -1)
    value = packed[:, k_end:].reshape(num_key_value_heads * head_dim, -1)
    return query, key, value


def iter_checkpoint_tensors(
    model: nn.Module,
    weight_map: Mapping[str, list[str]],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield parameters plus mapped persistent buffers for checkpoint I/O."""
    yield from model.named_parameters()
    parameter_names = {name for name, _parameter in model.named_parameters()}
    state_names = set(model.state_dict())
    mapped_names = set(weight_map)
    for name, buffer in model.named_buffers():
        if name in state_names and name in mapped_names and name not in parameter_names:
            yield name, buffer


__all__ = [
    "iter_checkpoint_tensors",
    "pack_grouped_query_qkv",
    "unpack_grouped_query_qkv",
]

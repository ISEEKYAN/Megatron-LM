# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Policy helpers for token-wise expert-parallel chunk overlap."""

from __future__ import annotations


def validate_ep_chunk_overlap_config(
    enabled: bool,
    *,
    use_deepep: bool,
    ep_size: int,
    topk: int,
    max_token_rows_per_rank: int | None = None,
) -> bool:
    if not isinstance(enabled, bool):
        raise TypeError("enable_ep_chunk_overlap must be a bool")
    if enabled and (not use_deepep or ep_size <= 1):
        raise ValueError("ChunkedEP requires DeepEP and EP > 1")
    if enabled and topk > ep_size:
        raise ValueError("ChunkedEP router top-k must not exceed EP size")
    if enabled and (
        not isinstance(max_token_rows_per_rank, int)
        or isinstance(max_token_rows_per_rank, bool)
        or max_token_rows_per_rank < 2
    ):
        raise ValueError("ChunkedEP requires ep_chunk_max_token_rows_per_rank >= 2")
    return enabled


def ep_chunk_ranges(
    num_tokens: int,
) -> list[tuple[int, int]]:
    """Split tokens into the one supported production profile: two halves."""
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
        raise TypeError("EP chunk token count must be an int")
    if num_tokens < 2:
        raise ValueError("EP chunk overlap requires at least two tokens")
    midpoint = (num_tokens + 1) // 2
    return [(0, midpoint), (midpoint, num_tokens)]


__all__ = [
    "ep_chunk_ranges",
    "validate_ep_chunk_overlap_config",
]

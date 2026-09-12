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
    *,
    chunk_count: int = 2,
) -> list[tuple[int, int]]:
    """Split tokens into a fixed number of contiguous non-empty chunks."""
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int):
        raise TypeError("EP chunk token count must be an int")
    if isinstance(chunk_count, bool) or not isinstance(chunk_count, int):
        raise TypeError("EP chunk count must be an int")
    if chunk_count < 2:
        raise ValueError("EP chunk overlap requires at least two chunks")
    if num_tokens < chunk_count:
        if chunk_count == 2:
            raise ValueError("EP chunk overlap requires at least two tokens")
        raise ValueError(
            f"EP chunk overlap requires at least {chunk_count} tokens"
        )
    base, remainder = divmod(num_tokens, chunk_count)
    ranges = []
    start = 0
    for index in range(chunk_count):
        end = start + base + int(index < remainder)
        ranges.append((start, end))
        start = end
    return ranges


__all__ = [
    "ep_chunk_ranges",
    "validate_ep_chunk_overlap_config",
]

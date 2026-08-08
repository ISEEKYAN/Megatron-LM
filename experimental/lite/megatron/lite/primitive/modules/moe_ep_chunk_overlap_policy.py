# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Policy helpers for token-wise expert-parallel chunk overlap."""

from __future__ import annotations


def validate_ep_chunk_overlap_config(
    enabled: bool, *, use_deepep: bool, ep_size: int, topk: int
) -> bool:
    if not isinstance(enabled, bool):
        raise TypeError("enable_ep_chunk_overlap must be a bool")
    if enabled and (not use_deepep or ep_size <= 1):
        raise ValueError("ChunkedEP requires DeepEP and EP > 1")
    if enabled and topk > ep_size:
        raise ValueError("ChunkedEP router top-k must not exceed EP size")
    return enabled


def recompute_modules_for_ep_chunk_overlap(
    modules: list[str], *, enabled: bool
) -> list[str]:
    """Avoid wrapping the native ChunkedEP full recompute in another checkpoint."""
    if not enabled:
        return modules
    if "full" in modules:
        return ["attn"]
    return [name for name in modules if name != "moe"]


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
    "recompute_modules_for_ep_chunk_overlap",
    "validate_ep_chunk_overlap_config",
]

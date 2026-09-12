# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3-only choice of its bounded non-fused head-loss composition."""

from __future__ import annotations


def use_chunked_head_loss(
    *,
    has_labels: bool,
    use_fused_kernels: bool,
    calculate_entropy: bool,
    has_chunked_ep: bool,
) -> bool:
    """Keep unsupported output and non-ChunkedEP compositions on their fallback."""
    return (
        has_labels
        and has_chunked_ep
        and not use_fused_kernels
        and not calculate_entropy
    )


def balanced_head_loss_chunk_size(token_count: int, chunk_count: int) -> int:
    """Use all configured EP chunks with token windows differing by at most one."""
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or token_count <= 0
    ):
        raise ValueError(f"token_count must be a positive integer, got {token_count!r}")
    if (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count <= 0
    ):
        raise ValueError(f"chunk_count must be a positive integer, got {chunk_count!r}")
    return (token_count + chunk_count - 1) // chunk_count


__all__ = ["balanced_head_loss_chunk_size", "use_chunked_head_loss"]

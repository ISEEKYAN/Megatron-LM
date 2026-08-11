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


__all__ = ["use_chunked_head_loss"]

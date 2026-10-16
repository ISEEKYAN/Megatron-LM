# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batched stable compaction of compressed and sliding-window KV slots."""
import torch


def sparse_metadata(mask, main_indices, *, main_size, window_size, width):
    """Keep input main-slot order followed by ascending visible window slots."""
    if mask.ndim != 3 or mask.dtype != torch.bool or not mask.is_cuda:
        raise ValueError('Sparse mask must be CUDA BSN boolean')
    batch, rows, _ = mask.shape
    if not 0 <= window_size <= mask.shape[-1] or width <= 0:
        raise ValueError('Invalid sparse window or output width')
    has_main = main_indices is not None
    if has_main and (main_indices.shape[:2] != (batch, rows) or main_indices.ndim != 3):
        raise ValueError('Sparse main indices must be BST')
    from vllm.model_executor.kernels.batch_invariant import compact_sparse_slots

    return compact_sparse_slots(
        mask, main_indices, main_size=main_size, window_size=window_size, width=width
    )

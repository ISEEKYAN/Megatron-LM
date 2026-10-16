# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Probability arithmetic for the materialized deployment head."""

import torch


def log_softmax(logits):
    """Share vLLM's native FP32 probability forward and owned native VJP."""
    value = logits.float()
    if not value.is_cuda:
        return torch.log_softmax(value, dim=-1)
    from vllm.model_executor.kernels.batch_invariant import native_log_softmax

    rows = value.reshape(-1, value.shape[-1])
    return native_log_softmax(rows, -1).reshape_as(value)

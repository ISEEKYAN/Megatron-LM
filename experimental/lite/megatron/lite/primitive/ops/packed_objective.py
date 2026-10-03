# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Masked token objective; callers own target shifting and token layout."""

import math

import torch
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy


def packed_objective(
    logits,
    labels=None,
    mask=None,
    *,
    temperature=1.0,
    denominator=None,
    calculate_entropy=False,
    loss_scale=1.0,
    return_log_probs=True,
    tp_group=None,
):
    """Reduce already-aligned targets; denominator may cover multiple shards/batches.

    Without labels, logits are unscaled and retain their local vocabulary partition.
    Temperature applies only to the labeled objective. With labels,
    a missing mask means every token counts; an all-zero mask produces zero loss.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if denominator is not None and (not math.isfinite(denominator) or denominator <= 0):
        raise ValueError("Loss denominator must be finite and positive")
    if labels is not None:
        logits = logits / temperature
    result = {}
    if calculate_entropy:
        result["entropy"] = vocab_parallel_entropy(logits, tp_group)
    if labels is None:
        result["logits"] = logits
        return result
    if labels.shape != logits.shape[:-1] or (
        mask is not None and mask.shape != labels.shape
    ):
        raise ValueError("Labels and mask must match the logits token shape")
    mask = torch.ones_like(labels, dtype=torch.float32) if mask is None else mask
    denominator = mask.sum().clamp_min(1) if denominator is None else denominator
    log_probs = -vocab_parallel_cross_entropy(logits.clone(), labels, tp_group)
    result["loss"] = -(log_probs * mask).sum() / denominator * loss_scale
    if return_log_probs:
        result["log_probs"] = log_probs
    return result

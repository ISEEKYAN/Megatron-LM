# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Materialized-head log probabilities for the packed protocol consumer."""
import torch

from .deployment_math import visible_forward


def log_softmax(logits):
    """Fixed-row CUDA log probabilities with an owned FP32 logsumexp VJP.

    The input is the materialized model-owned head output. No projection or
    quantizer is repeated here. CPU uses the ordinary Torch implementation.
    """
    if not logits.is_cuda:
        return torch.log_softmax(logits.float(), dim=-1)

    def visible(x):
        from vllm.model_executor.determinism.batch_invariant import log_softmax

        return log_softmax(x.float(), dim=-1)

    def reference(x):
        value = x.float()
        return value - torch.logsumexp(value, dim=-1, keepdim=True)

    return visible_forward(visible, reference, logits)

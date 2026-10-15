# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-visible norm arithmetic and owned gradients."""

import torch

from .autograd import visible_forward
from .linear import decoded_bf16_master


def rms_norm(x, weight, shape, eps):
    def reference(values, master):
        return torch.nn.functional.rms_norm(
            values, shape, decoded_bf16_master(master), eps
        )

    def visible(values, master):
        from vllm.model_executor.kernels.batch_invariant import rms_norm_cuda

        out = rms_norm_cuda(
            values.reshape(-1, values.shape[-1]),
            master.bfloat16().to(values.dtype),
            eps,
        )
        return out.reshape_as(values)

    return visible_forward(visible, reference, x, weight)


def qkv_rms_norm(x, weight, shape, eps):
    """Native CUDA row arithmetic shared with deployment Q/KV norms.

    The FP32/STE reference owns the unchanged mathematical VJP.
    """
    from megatron.lite.primitive.kernels import deployment_rms_norm

    if tuple(shape) != (x.shape[-1],):
        raise ValueError('Deployment QKV RMS requires one normalized row dimension')

    def reference(values, master):
        return torch.nn.functional.rms_norm(
            values, shape, decoded_bf16_master(master), eps
        )

    return visible_forward(
        lambda values, master: deployment_rms_norm.row_rms_norm(values, master, eps),
        reference,
        x,
        weight,
    )

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment row RMS forward shared with the native CUDA provider."""

import torch


def row_rms_norm(x, weight, eps):
    """Decode BF16 gamma for forward; modules.deployment_math.norm owns the VJP."""
    from vllm.model_executor.kernels.batch_invariant import rms_norm_cuda

    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim < 1
        or weight.device != x.device
        or weight.dtype != torch.float32
        or weight.shape != (x.shape[-1],)
        or x.shape[-1] < 1
    ):
        raise ValueError(
            'Deployment row RMS requires CUDA BF16 rows and an FP32 gamma master'
        )
    return rms_norm_cuda(x, weight.bfloat16(), eps)

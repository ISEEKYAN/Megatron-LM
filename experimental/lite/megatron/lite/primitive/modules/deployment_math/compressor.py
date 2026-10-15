# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-visible compressor arithmetic and owned gradients."""

import torch

from .autograd import visible_forward
from .linear import bf16_fp32_linear, decoded_bf16_master


def compressor(x, wkv, wgate, gamma, ratio, eps):
    # Match the deployment's merged BF16-input/weight GEMM, FP32 output.
    weights = wkv if wgate is None else torch.cat((wkv, wgate))
    raw = bf16_fp32_linear(x, weights)

    def visible(scores, master):
        from megatron.lite.primitive.kernels import deployment_compressor

        return deployment_compressor.compress_norm(scores, master, ratio, eps)

    def reference(scores, master):
        cutoff = scores.shape[1] // ratio * ratio
        values = scores[:, :cutoff, :512]
        if ratio == 2:
            gates = scores[:, :cutoff, 512:].unflatten(1, (-1, 2))
            values = (values.unflatten(1, (-1, 2)) * gates.softmax(2)).sum(2)
        # Native pooling retains FP32 through normalization; only latent rounds.
        return torch.nn.functional.rms_norm(
            values, (512,), decoded_bf16_master(master), eps
        ).bfloat16()

    return visible_forward(visible, reference, raw, gamma)

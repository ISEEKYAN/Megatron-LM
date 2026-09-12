# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Floating linear with a persistent FP32 master and native FP32 wgrad.

The forward uses the activation dtype. The backward weight GEMM multiplies
FP32 operands and returns FP32 directly to the FP32 leaf, avoiding a BF16
weight-gradient intermediate. This is a correctness provider, not TE fusion.
"""

import torch
from torch.nn import functional as F


class _NativeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        if weight.dtype != torch.float32:
            raise ValueError('Native wgrad requires an FP32 master')
        compute_weight = weight.to(x.dtype)
        ctx.save_for_backward(x, compute_weight)
        return F.linear(x, compute_weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx = grad @ weight
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dw = (
                grad.reshape(-1, weight.shape[0]).float().T
                @ x.reshape(-1, weight.shape[1]).float()
            )
        return dx, dw


def native_fp32_linear(x, weight):
    return _NativeLinear.apply(x, weight)

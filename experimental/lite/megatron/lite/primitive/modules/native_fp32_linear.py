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


class Linear(torch.nn.Module):
    """Bias-free floating, block-FP8 or group32-FP4 projection."""

    def __init__(
        self,
        input_size,
        output_size,
        *,
        fp8=False,
        dtype=torch.bfloat16,
        fp8_operator=None,
        fake_quant=None
    ):
        super().__init__()
        self.in_features, self.out_features = input_size, output_size
        self.weight = torch.nn.Parameter(
            torch.empty(output_size, input_size, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.fp8 = fp8
        self.fp8_operator, self.fake_quant = fp8_operator, fake_quant

    def forward(self, x):
        weight = self.weight
        if getattr(self, 'quantized', False):
            x, weight = self.fake_quant(x), self.fake_quant(weight)
        if self.fp8:
            return self.fp8_operator(x, weight)
        linear = native_fp32_linear if getattr(self, 'native_fp32', False) else F.linear
        return linear(x, weight)


def FP4Linear(input_size, output_size, *, quantized, fake_quant):
    module = Linear(input_size, output_size, fake_quant=fake_quant)
    module.quantized = quantized
    return module


def configure_residual_projections(model, dtype, optimizing, row_type):
    for module in model.modules():
        if isinstance(module, (Linear, torch.nn.Embedding)):
            if not isinstance(module, Linear) or module.weight.dtype != torch.float32:
                module.to(dtype=dtype)
            if optimizing and isinstance(module, Linear):
                module.native_fp32 = True
                module.weight.data = module.weight.data.float()
        if isinstance(module, row_type):
            module.output_dtype = dtype

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Distinct SWA/Linear FP8 policies and a native FP8 GEMM correctness path."""

from dataclasses import dataclass

import torch

from .block_fp8 import dequantize_block_fp8, quantize_block_fp8
from .ds41_kv import _IdentityGradient, _validate_input


@dataclass(frozen=True)
class FP8Values:
    values: torch.Tensor
    scale: torch.Tensor
    decoded: torch.Tensor


@torch.no_grad()
def _quantize_rows(x):
    _validate_input(x, 32)
    rows = x.reshape(-1, x.shape[-1])
    values, scale = quantize_block_fp8(rows, (1,32), scale_format="e8m0")
    decoded = dequantize_block_fp8(values, scale, (1,32)).to(x.dtype)
    return FP8Values(values.reshape(x.shape), scale.reshape(*x.shape[:-1], -1), decoded.reshape(x.shape))


def quantize_swa(post_rope):
    """Quantize every channel, including the entire rotated tail."""
    return _quantize_rows(post_rope)


def quantize_linear_activation(x):
    return _quantize_rows(x)


def fake_quant_swa(post_rope):
    return _IdentityGradient.apply(post_rope, quantize_swa(post_rope).decoded)


def _fp8_gemm(a, a_scale, b, b_scale):
    """FP8 products per K block, followed by FP32 scale correction/accumulation."""
    rows, columns = a.shape[0], b.shape[0]
    output = torch.zeros(rows, columns, device=a.device, dtype=torch.float32)
    unit = torch.ones((), device=a.device, dtype=torch.float32)
    for group, start in enumerate(range(0, a.shape[1], 32)):
        # Scalar unit scales ensure native FP8 multiplication; actual block scales
        # are applied separately, as in the published blockwise accumulation.
        product = torch._scaled_mm(a[:, start:start+32].contiguous(),
                                   b[:, start:start+32].contiguous().T,
                                   unit, unit, out_dtype=torch.float32)
        output += (product * a_scale[:, group].float()[:, None]
                   * b_scale[:, group].float().repeat_interleave(32)[None, :])
    return output


class _DynamicLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        activation = quantize_linear_activation(x)
        encoded_weight, scales = quantize_block_fp8(weight, (32,32), scale_format="e8m0")
        decoded_weight = dequantize_block_fp8(encoded_weight, scales, (32,32)).to(weight.dtype)
        ctx.save_for_backward(activation.decoded.reshape(-1, x.shape[-1]), decoded_weight)
        ctx.input_shape = x.shape
        result = _fp8_gemm(activation.values.reshape(-1, x.shape[-1]),
                           activation.scale.reshape(-1, x.shape[-1] // 32), encoded_weight, scales)
        return result.reshape(*x.shape[:-1], weight.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        activation, weight = ctx.saved_tensors
        flat_grad = grad.reshape(-1, weight.shape[0]).float()
        dx = (flat_grad @ weight.float()).reshape(ctx.input_shape).to(activation.dtype)
        dw = (flat_grad.T @ activation.float()).to(weight.dtype)
        return dx, dw


def dynamic_fp8_linear(x, weight):
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("dynamic FP8 Linear requires CUDA; no CPU GEMM fallback")
    _validate_input(x, 32)
    _validate_input(weight, 32)
    if x.ndim < 2 or weight.ndim != 2 or weight.shape[0] % 32 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("Linear requires matching K and weight dimensions divisible by 32")
    if x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError("floating activation and weight must share device and dtype")
    return _DynamicLinear.apply(x, weight)

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
    values, scale = quantize_block_fp8(rows, (1, 32), scale_format="e8m0")
    decoded = dequantize_block_fp8(values, scale, (1, 32)).to(x.dtype)
    return FP8Values(
        values.reshape(x.shape),
        scale.reshape(*x.shape[:-1], -1),
        decoded.reshape(x.shape),
    )


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
        product = torch._scaled_mm(
            a[:, start : start + 32].contiguous(),
            b[:, start : start + 32].contiguous().T,
            unit,
            unit,
            out_dtype=torch.float32,
        )
        output += (
            product
            * a_scale[:, group].float()[:, None]
            * b_scale[:, group].float().repeat_interleave(32)[None, :]
        )
    return output


class _DynamicLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        activation = quantize_linear_activation(x)
        encoded_weight, scales = quantize_block_fp8(
            weight, (32, 32), scale_format="e8m0"
        )
        decoded_weight = dequantize_block_fp8(encoded_weight, scales, (32, 32)).to(
            weight.dtype
        )
        ctx.save_for_backward(
            activation.decoded.reshape(-1, x.shape[-1]), decoded_weight
        )
        ctx.input_shape = x.shape
        result = _fp8_gemm(
            activation.values.reshape(-1, x.shape[-1]),
            activation.scale.reshape(-1, x.shape[-1] // 32),
            encoded_weight,
            scales,
        )
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
    if (
        x.ndim < 2
        or weight.ndim != 2
        or weight.shape[0] % 32
        or x.shape[-1] != weight.shape[-1]
    ):
        raise ValueError(
            "Linear requires matching K and weight dimensions divisible by 32"
        )
    if x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError("floating activation and weight must share device and dtype")
    return _DynamicLinear.apply(x, weight)


class _PublishedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, scale, weight, master, output_dtype):
        encoded, weight_scale = quantize_block_fp8(weight, (32, 32), scale_format='e8m0')
        ctx.save_for_backward(values, scale, encoded, weight_scale)
        ctx.weight_dtype = weight.dtype
        ctx.has_master = master is not None
        # Pass release values/scales directly. Do not run activation quantization.
        result = _fp8_gemm(values.reshape(-1, values.shape[-1]),
                           scale.reshape(-1, scale.shape[-1]), encoded, weight_scale)
        return result.reshape(*values.shape[:-1], weight.shape[0]).to(output_dtype)

    @staticmethod
    def backward(ctx, grad):
        values, scale, encoded, weight_scale = ctx.saved_tensors
        gradient = grad.reshape(-1, grad.shape[-1]).float()
        dx = dw = None
        if ctx.has_master and ctx.needs_input_grad[3]:
            weight = dequantize_block_fp8(encoded, weight_scale, (32, 32))
            dx = (gradient @ weight.float()).reshape(values.shape)
        if ctx.needs_input_grad[2]:
            # Decoding is for the backward arithmetic, never a GEMM input rewrite.
            activation = values.float() * scale.float().repeat_interleave(32, -1)
            dw = (gradient.T @ activation.reshape(-1, values.shape[-1])).to(ctx.weight_dtype)
        return None, None, dw, dx, None


def published_fp8_linear(values, scale, weight, *, master=None, output_dtype=torch.bfloat16):
    """Native FP8 GEMM using immutable row/block32 activation publication.

    Optional FP32 master is the STE gradient carrier, not forward GEMM data.
    This is a numerical path using the existing blockwise kernel; it makes no
    throughput/overlap claim. There is no CPU or decoded activation fallback.
    """
    if not values.is_cuda or not scale.is_cuda or not weight.is_cuda:
        raise RuntimeError('Published FP8 Linear requires CUDA; no CPU GEMM fallback')
    if (values.ndim < 2 or values.shape[-1] % 32
            or values.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu
            or scale.shape != (*values.shape[:-1], values.shape[-1] // 32)
            or weight.ndim != 2 or weight.shape[0] % 32
            or weight.shape[1] != values.shape[-1]
            or values.device != scale.device or values.device != weight.device):
        raise ValueError('Require compatible FP8 values, E8M0 block32 scales and projection')
    if master is not None and (master.dtype != torch.float32 or master.shape != values.shape
                               or master.device != values.device):
        raise ValueError('Master carrier must be matching resident FP32 rows')
    if output_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('Require floating output dtype')
    return _PublishedLinear.apply(values, scale, weight, master, output_dtype)

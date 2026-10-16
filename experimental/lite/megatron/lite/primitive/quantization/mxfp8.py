# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Distinct SWA/Linear FP8 policies and a native FP8 GEMM correctness path."""

from dataclasses import dataclass

import torch

from .block_fp8 import dequantize_block_fp8, quantize_block_fp8
from .mxfp4 import _validate_input
from .qat import _FloatFakeQuantSTE


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


# SWA (including its rotated tail) and Linear activations share row/block32 E8M0.
quantize_linear_activation = _quantize_rows


def quantize_swa(post_rope):
    return _quantize_rows(post_rope)


def fake_quant_swa(post_rope):
    return _FloatFakeQuantSTE.apply(post_rope, quantize_swa(post_rope).decoded, None)


def _fp8_gemm(a, a_scale, b, b_scale):
    """FP8 products per K block, followed by FP32 scale correction/accumulation."""
    from megatron.lite.primitive.kernels import mxfp8_block32

    return mxfp8_block32.batched_block32_mm(
        a, a_scale, b, b_scale, torch.float32, weight_scale_group_rows=32
    )


class _DynamicLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        activation = quantize_linear_activation(x)
        encoded_weight, scales = quantize_block_fp8(
            weight, (32, 32), scale_format="e8m0"
        )
        decoded_weight = dequantize_block_fp8(encoded_weight, scales, (32, 32)).to(
            x.dtype
        )
        ctx.weight_dtype = weight.dtype
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
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = (
                (flat_grad @ weight.float())
                .reshape(ctx.input_shape)
                .to(activation.dtype)
            )
            dw = (flat_grad.T @ activation.float()).to(ctx.weight_dtype)
        return dx, dw


def dynamic_fp8_linear(x, weight, *, deployment_math=False):
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
    # Inverse RoPE keeps FP32 activations even in a frozen BF16 reference.
    mixed_reference = (
        deployment_math and x.dtype == torch.float32 and weight.dtype == torch.bfloat16
    )
    if x.device != weight.device or (
        x.dtype != weight.dtype
        and weight.dtype != torch.float32
        and not mixed_reference
    ):
        raise ValueError(
            "activation and weight must share device and compute dtype, use an FP32 master, or explicitly opt into deployment FP32 activations with BF16 reference weights"
        )
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("dynamic FP8 Linear requires CUDA; no CPU GEMM fallback")
    return _DynamicLinear.apply(x, weight)


class _DynamicGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, activation=None):
        from megatron.lite.primitive.kernels import mxfp8_block32

        if activation is None:
            activation = quantize_linear_activation(x)
        codes, scales = quantize_block_fp8(weight, (32, 32), scale_format="e8m0")
        decoded = dequantize_block_fp8(codes, scales, (32, 32)).to(x.dtype)
        ctx.weight_dtype = weight.dtype
        ctx.input_shape = x.shape
        ctx.save_for_backward(activation.decoded, decoded)
        groups, outputs, inner = weight.shape
        result = mxfp8_block32.batched_block32_mm(
            activation.values.reshape(-1, groups, inner),
            activation.scale.reshape(-1, groups, inner // 32),
            codes,
            scales,
            torch.float32,
            grouped=True,
            weight_scale_group_rows=32,
        )
        return result.reshape(*x.shape[:-1], outputs).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        activation, weight = ctx.saved_tensors
        groups, outputs, inner = weight.shape
        g = grad.reshape(-1, groups, outputs).transpose(0, 1).float()
        a = activation.reshape(-1, groups, inner).transpose(0, 1).float()
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = (
                torch.bmm(g, weight.float())
                .transpose(0, 1)
                .reshape(ctx.input_shape)
                .to(activation.dtype)
            )
            dw = torch.bmm(g.transpose(1, 2), a).to(ctx.weight_dtype)
        return dx, dw, None


def dynamic_fp8_grouped_linear(x, weight, *, activation=None):
    """One grouped K32 launch; each group retains its independent forward codec.

    ``activation`` supplies FP8 codes/scales produced by a native kernel for
    ``x``; ``x`` then only carries the VJP.
    """
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("dynamic grouped FP8 Linear requires CUDA")
    _validate_input(x, 32)
    _validate_input(weight, 32)
    if (
        x.ndim < 3
        or weight.ndim != 3
        or x.shape[-2] != weight.shape[0]
        or x.shape[-1] != weight.shape[-1]
        or weight.shape[1] % 32
    ):
        raise ValueError(
            "Grouped Linear requires matching groups/K and N divisible by 32"
        )
    if x.device != weight.device or weight.dtype not in (
        x.dtype,
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError("Grouped activation/master device or dtype mismatch")
    if activation is not None and (
        activation.values.shape != x.shape
        or activation.scale.shape != (*x.shape[:-1], x.shape[-1] // 32)
    ):
        raise ValueError("Native grouped activation must match the input blocks")
    return _DynamicGroupedLinear.apply(x, weight, activation)

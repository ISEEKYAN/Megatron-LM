# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Block32 E4M3/E8M0 codec and a dynamic FP8 Linear correctness path.

Activations use row blocks ``(1, 32)``; weights use ``(32, 32)`` tiles. Both
reuse :func:`quantize_block_fp8` with UE8M0 scales. The GEMM multiplies native
E4M3 operands with :func:`torch._scaled_mm` once per 32-wide K block (unit
scalar scales), applies the two block scales in FP32 and accumulates in FP32.
It is a correctness provider, not a fused production kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .block_fp8 import dequantize_block_fp8, quantize_block_fp8

BLOCK = 32
ACTIVATION_BLOCK = (1, BLOCK)
WEIGHT_BLOCK = (BLOCK, BLOCK)


@dataclass(frozen=True)
class FP8Values:
    values: torch.Tensor
    scale: torch.Tensor
    decoded: torch.Tensor


def _validate(x: torch.Tensor, name: str) -> None:
    if x.ndim < 2 or x.shape[-1] == 0 or x.shape[-1] % BLOCK:
        raise ValueError(
            f"{name} must be >=2-D with last dimension divisible by {BLOCK}"
        )
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"{name} must be F32, BF16 or F16, got {x.dtype}")


@torch.no_grad()
def quantize_block32(x: torch.Tensor, block_shape=ACTIVATION_BLOCK) -> FP8Values:
    """Encode trailing 2-D blocks; ``decoded`` is the dequantized value in ``x.dtype``."""
    _validate(x, "block32 input")
    values, scale = quantize_block_fp8(x, block_shape, scale_format="e8m0")
    decoded = dequantize_block_fp8(values, scale, block_shape).to(x.dtype)
    return FP8Values(values, scale, decoded)


def _fp8_gemm(a, a_scale, b, b_scale):
    """``a [M, K] @ b[N, K].T`` with per-K-block scale correction in FP32."""
    output = torch.zeros(a.shape[0], b.shape[0], device=a.device, dtype=torch.float32)
    unit = torch.ones((), device=a.device, dtype=torch.float32)
    for group, start in enumerate(range(0, a.shape[1], BLOCK)):
        product = torch._scaled_mm(
            a[:, start : start + BLOCK].contiguous(),
            b[:, start : start + BLOCK].contiguous().T,
            unit,
            unit,
            out_dtype=torch.float32,
        )
        row_scale = a_scale[:, group].float()[:, None]
        column_scale = b_scale[:, group].float().repeat_interleave(BLOCK)[None, :]
        output += product * row_scale * column_scale
    return output


class _DynamicFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, diagnostic):
        flat = x.reshape(-1, x.shape[-1])
        activation = quantize_block32(flat)
        encoded = quantize_block32(weight, WEIGHT_BLOCK)
        decoded_weight = encoded.decoded.float()
        ctx.save_for_backward(activation.decoded, decoded_weight)
        ctx.input_shape, ctx.weight_dtype = x.shape, weight.dtype
        if diagnostic:
            # Floating diagnostic: the same decoded operands through one FP32 GEMM.
            result = F.linear(activation.decoded.float(), decoded_weight)
        else:
            result = _fp8_gemm(
                activation.values, activation.scale, encoded.values, encoded.scale
            )
        return result.reshape(*x.shape[:-1], weight.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        activation, weight = ctx.saved_tensors
        flat_grad = grad.reshape(-1, weight.shape[0]).float()
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = (flat_grad @ weight).reshape(ctx.input_shape).to(activation.dtype)
            dw = (flat_grad.T @ activation.float()).to(ctx.weight_dtype)
        return dx, dw, None


def dynamic_fp8_linear(x, weight, *, diagnostic=False):
    """Bias-free ``x @ weight.T`` with block32 FP8 operands and FP32 wgrad.

    ``diagnostic=True`` keeps the codec but replaces the FP8 GEMM with an FP32
    GEMM on decoded operands, separating codec error from GEMM error.
    """
    _validate(x, "activation")
    _validate(weight, "weight")
    if weight.ndim != 2 or weight.shape[0] % BLOCK or x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"weight must be [N, K] with N divisible by {BLOCK}, K matching x"
        )
    if x.device != weight.device:
        raise ValueError("activation and weight must share a device")
    if x.dtype != weight.dtype and weight.dtype != torch.float32:
        raise TypeError("weight must match the activation dtype or be an FP32 master")
    if not diagnostic and not x.is_cuda:
        raise RuntimeError(
            "dynamic FP8 Linear requires CUDA; use diagnostic=True on CPU"
        )
    return _DynamicFP8Linear.apply(x, weight, diagnostic)


__all__ = ["FP8Values", "dynamic_fp8_linear", "quantize_block32"]

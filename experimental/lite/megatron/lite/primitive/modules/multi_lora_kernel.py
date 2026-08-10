# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dispatch the full Mint-derived Triton BGMV training implementation."""

from __future__ import annotations

import torch
from megatron.lite.primitive.modules import multi_lora_bgmv

_TRITON_AVAILABLE = multi_lora_bgmv._TRITON_AVAILABLE
use_fused_bgmv = multi_lora_bgmv.use_fused_bgmv


def _can_use_triton(
    x: torch.Tensor, a_bank: torch.Tensor, b_bank: torch.Tensor
) -> bool:
    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and x.is_contiguous()
        and a_bank.is_contiguous()
        and b_bank.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def dense_batched_lora_forward(
    x: torch.Tensor,
    a_bank: torch.Tensor,
    b_bank: torch.Tensor,
    slots: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return delta and saved hidden via Triton or the independent eager path."""
    if _can_use_triton(x, a_bank, b_bank):
        return multi_lora_bgmv.bgmv_fwd(
            x, a_bank, b_bank, slots, scale, max_g_size_hint=x.shape[0]
        )
    hidden = torch.bmm(a_bank.index_select(0, slots), x.unsqueeze(-1)).squeeze(-1)
    delta = torch.bmm(b_bank.index_select(0, slots), hidden.unsqueeze(-1)).squeeze(-1)
    return delta * scale, hidden


def dense_batched_lora_stage_forward(
    x, weight, slots, *, scale=1.0, output_dtype=None, max_g_size_hint=None
):
    """Selected-bank shrink/expand stage; CUDA dispatch stays in Mint Triton."""
    if _can_use_triton(x, weight, weight):
        return multi_lora_bgmv.bgmv_stage_fwd(
            x,
            weight,
            slots,
            scale=scale,
            output_dtype=output_dtype,
            max_g_size_hint=max_g_size_hint,
        )
    selected = weight.index_select(0, slots)
    compute_dtype = torch.promote_types(x.dtype, weight.dtype)
    output = torch.bmm(
        selected.to(compute_dtype), x.to(compute_dtype).unsqueeze(-1)
    ).squeeze(-1)
    return (output * scale).to(output_dtype or x.dtype)


class BatchedLoraLinearStage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, slots, scale, output_dtype, max_g_size_hint):
        if slots.ndim != 1 or slots.shape[0] != x.shape[0]:
            raise ValueError(
                "stage slots must be one-dimensional and match input rows."
            )
        if slots.numel() > 1 and bool(torch.any(slots[1:] < slots[:-1])):
            raise ValueError(
                "batched LoRA linear stage requires slots sorted ascending."
            )
        ctx.save_for_backward(x, weight, slots)
        ctx.scale, ctx.max_g_size_hint = scale, max_g_size_hint
        return dense_batched_lora_stage_forward(
            x,
            weight,
            slots,
            scale=scale,
            output_dtype=output_dtype,
            max_g_size_hint=max_g_size_hint,
        )

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, slots = ctx.saved_tensors
        if _can_use_triton(x, weight, weight):
            grad_x, grad_w = multi_lora_bgmv.bgmv_stage_bwd(
                x,
                grad_out,
                weight,
                slots,
                scale=ctx.scale,
                max_g_size_hint=ctx.max_g_size_hint,
            )
        else:
            selected = weight.index_select(0, slots)
            compute_dtype = torch.promote_types(grad_out.dtype, weight.dtype)
            grad_x = (
                torch.bmm(
                    grad_out.to(compute_dtype).unsqueeze(1), selected.to(compute_dtype)
                ).squeeze(1)
                * ctx.scale
            ).to(x.dtype)
            grad_w = torch.zeros_like(weight)
            grad_w.index_add_(
                0,
                slots,
                (
                    grad_out.to(compute_dtype).unsqueeze(-1)
                    * x.to(compute_dtype).unsqueeze(1)
                    * ctx.scale
                ).to(weight.dtype),
            )
        return grad_x, grad_w, None, None, None, None


def batched_lora_linear_stage(
    x, weight, slots, *, scale=1.0, output_dtype=None, max_g_size_hint=None
):
    return BatchedLoraLinearStage.apply(
        x, weight, slots, scale, output_dtype, max_g_size_hint
    )


def dense_batched_lora_backward(x, grad_output, a_bank, b_bank, slots, scale, hidden):
    """Return Triton gradients, or ``None`` to select the eager fallback."""
    if not _can_use_triton(x, a_bank, b_bank):
        return None
    return multi_lora_bgmv.bgmv_bwd(
        x,
        grad_output,
        a_bank,
        b_bank,
        slots,
        scale,
        hidden=hidden,
        max_g_size_hint=x.shape[0],
    )


__all__ = [
    "_TRITON_AVAILABLE",
    "use_fused_bgmv",
    "dense_batched_lora_backward",
    "dense_batched_lora_forward",
    "dense_batched_lora_stage_forward",
    "BatchedLoraLinearStage",
    "batched_lora_linear_stage",
]

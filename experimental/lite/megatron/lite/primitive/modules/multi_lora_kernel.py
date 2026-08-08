# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dispatch the full Mint-derived Triton BGMV training implementation."""

from __future__ import annotations

import torch
from megatron.lite.primitive.modules import multi_lora_bgmv

_TRITON_AVAILABLE = multi_lora_bgmv._TRITON_AVAILABLE
_FUSED_N_THRESHOLD = 256


def _use_fused_bgmv(out_features: int) -> bool:
    return out_features >= _FUSED_N_THRESHOLD


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
    "_use_fused_bgmv",
    "dense_batched_lora_backward",
    "dense_batched_lora_forward",
]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent eager reference for a row-selected dense LoRA bank."""

from __future__ import annotations

import torch


def dense_lora_delta_reference(
    x: torch.Tensor,
    a_bank: torch.Tensor,
    b_bank: torch.Tensor,
    lora_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compute ``scale * B[slot] @ A[slot] @ x`` for every input row.

    A slot of ``-1`` is an explicitly inactive adapter and produces a zero
    delta. This intentionally uses indexing and matmul directly, rather than
    the operator kernel, so it remains a parity oracle.
    """
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional [tokens, in_features].")
    if a_bank.ndim != 3 or b_bank.ndim != 3:
        raise ValueError("A_bank and B_bank must be three-dimensional dense banks.")
    if lora_indices.ndim != 1:
        raise ValueError("lora_indices must be one-dimensional.")
    if lora_indices.dtype != torch.int64:
        raise ValueError("lora_indices must use torch.int64.")
    if lora_indices.shape[0] != x.shape[0]:
        raise ValueError("lora_indices must have one entry per input row.")
    if a_bank.shape[0] != b_bank.shape[0] or a_bank.shape[1] != b_bank.shape[2]:
        raise ValueError("A_bank and B_bank must agree on slots and rank.")
    if a_bank.shape[2] != x.shape[1]:
        raise ValueError("A_bank input features must match x.")
    if min(a_bank.shape[0], a_bank.shape[1]) < 1:
        raise ValueError("dense LoRA banks must have at least one slot and rank.")
    if lora_indices.numel() and (
        lora_indices.min() < 0 or lora_indices.max() >= a_bank.shape[0]
    ):
        raise IndexError("lora_indices contains a slot out of range.")
    if lora_indices.numel() > 1 and not bool(
        torch.all(lora_indices[1:] >= lora_indices[:-1]).item()
    ):
        raise ValueError("lora_indices must be monotonically non-decreasing.")

    hidden = torch.bmm(a_bank.index_select(0, lora_indices), x.unsqueeze(-1)).squeeze(
        -1
    )
    return (
        torch.bmm(b_bank.index_select(0, lora_indices), hidden.unsqueeze(-1)).squeeze(
            -1
        )
        * scale
    )


def dense_lora_backward_reference_fp32_single_cast(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    a_bank: torch.Tensor,
    b_bank: torch.Tensor,
    lora_indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate in FP32 and quantize each returned gradient once.

    This is an explicit numerical contract for diagnosing BF16 differences; it
    is deliberately separate from PyTorch autograd's implementation details.
    """
    x32, grad32 = x.float(), grad_output.float()
    a32, b32 = a_bank.float(), b_bank.float()
    selected_a, selected_b = (
        a32.index_select(0, lora_indices),
        b32.index_select(0, lora_indices),
    )
    hidden = torch.bmm(selected_a, x32.unsqueeze(-1)).squeeze(-1)
    grad_hidden = (
        torch.bmm(selected_b.transpose(1, 2), grad32.unsqueeze(-1)).squeeze(-1) * scale
    )
    grad_x = torch.bmm(selected_a.transpose(1, 2), grad_hidden.unsqueeze(-1)).squeeze(
        -1
    )
    grad_a = torch.zeros_like(a32)
    grad_a.index_add_(
        0,
        lora_indices,
        torch.bmm(grad_hidden.unsqueeze(-1), x32.unsqueeze(1)),
    )
    grad_b = torch.zeros_like(b32)
    grad_b.index_add_(
        0,
        lora_indices,
        torch.bmm((grad32 * scale).unsqueeze(-1), hidden.unsqueeze(1)),
    )
    return tuple(gradient.to(x.dtype) for gradient in (grad_x, grad_a, grad_b))


def dense_lora_backward_reference_bf16_staged(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    a_bank: torch.Tensor,
    b_bank: torch.Tensor,
    lora_indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Model the production BF16 stores between backward stages explicitly."""
    if x.dtype is not torch.bfloat16:
        raise ValueError("the staged reference is defined only for torch.bfloat16")
    x32, grad32 = x.float(), grad_output.float()
    a32, b32 = a_bank.float(), b_bank.float()
    selected_a, selected_b = (
        a32.index_select(0, lora_indices),
        b32.index_select(0, lora_indices),
    )
    hidden = torch.bmm(selected_a, x32.unsqueeze(-1)).squeeze(-1).to(torch.bfloat16)
    grad_hidden = (
        torch.bmm(selected_b.transpose(1, 2), grad32.unsqueeze(-1)).squeeze(-1) * scale
    ).to(torch.bfloat16)
    grad_x = (
        torch.bmm(selected_a.transpose(1, 2), grad_hidden.float().unsqueeze(-1))
        .squeeze(-1)
        .to(torch.bfloat16)
    )
    grad_a = torch.zeros_like(a32)
    grad_a.index_add_(
        0,
        lora_indices,
        torch.bmm(grad_hidden.float().unsqueeze(-1), x32.unsqueeze(1)),
    )
    grad_b = torch.zeros_like(b32)
    grad_b.index_add_(
        0,
        lora_indices,
        torch.bmm((grad32 * scale).unsqueeze(-1), hidden.float().unsqueeze(1)),
    )
    return grad_x, grad_a.to(torch.bfloat16), grad_b.to(torch.bfloat16)


__all__ = [
    "dense_lora_backward_reference_bf16_staged",
    "dense_lora_backward_reference_fp32_single_cast",
    "dense_lora_delta_reference",
]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Autograd operator for a dense, row-selected multi-LoRA bank."""

from __future__ import annotations

import numbers

import megatron.lite.primitive.modules.multi_lora_kernel as multi_lora_kernel
import torch


def _validate_inputs(x, a_bank, b_bank, lora_indices, scale) -> None:
    if not isinstance(scale, numbers.Real):
        raise TypeError("scale must be a real scalar.")
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional [tokens, in_features].")
    if a_bank.ndim != 3 or b_bank.ndim != 3:
        raise ValueError("A_bank and B_bank must be three-dimensional dense banks.")
    if lora_indices.ndim != 1:
        raise ValueError("lora_indices must be one-dimensional.")
    if lora_indices.shape[0] != x.shape[0]:
        raise ValueError("lora_indices must have one entry per input row.")
    if (
        x.device != a_bank.device
        or x.device != b_bank.device
        or x.device != lora_indices.device
    ):
        raise ValueError("x, banks, and lora_indices must share a device.")
    if x.dtype != a_bank.dtype or x.dtype != b_bank.dtype:
        raise ValueError("x, A_bank, and B_bank must share a dtype.")
    if lora_indices.dtype != torch.int64:
        raise ValueError("lora_indices must use torch.int64.")
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


class BatchedLoraDelta(torch.autograd.Function):
    """Apply per-row LoRA slots with the prototype-compatible ``apply`` API.

    ``A_bank`` is ``[slots, rank, in_features]`` and ``B_bank`` is
    ``[slots, out_features, rank]``. ``lora_indices`` has one int64,
    monotonically non-decreasing resident slot per row.
    """

    @staticmethod
    def forward(ctx, x, a_bank, b_bank, lora_indices, scale):
        _validate_inputs(x, a_bank, b_bank, lora_indices, scale)
        output, hidden = multi_lora_kernel.dense_batched_lora_forward(
            x, a_bank, b_bank, lora_indices, scale
        )
        ctx.save_for_backward(x, a_bank, b_bank, lora_indices, hidden)
        ctx.scale = float(scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, a_bank, b_bank, lora_indices, hidden = ctx.saved_tensors
        triton_grads = multi_lora_kernel.dense_batched_lora_backward(
            x, grad_output, a_bank, b_bank, lora_indices, ctx.scale, hidden
        )
        if triton_grads is not None:
            grad_x, grad_a, grad_b = triton_grads
            return grad_x, grad_a, grad_b, None, None
        grad_x = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        grad_a = torch.zeros_like(a_bank) if ctx.needs_input_grad[1] else None
        grad_b = torch.zeros_like(b_bank) if ctx.needs_input_grad[2] else None
        grad_delta = grad_output * ctx.scale
        selected_a = a_bank.index_select(0, lora_indices)
        selected_b = b_bank.index_select(0, lora_indices)
        projected_grad = torch.bmm(
            selected_b.transpose(1, 2), grad_delta.unsqueeze(-1)
        ).squeeze(-1)
        if grad_x is not None:
            grad_x.copy_(
                torch.bmm(
                    selected_a.transpose(1, 2), projected_grad.unsqueeze(-1)
                ).squeeze(-1)
            )
        if grad_a is not None:
            per_row_a = projected_grad.unsqueeze(-1) * x.unsqueeze(1)
            grad_a.index_add_(0, lora_indices, per_row_a)
        if grad_b is not None:
            per_row_b = grad_delta.unsqueeze(-1) * hidden.unsqueeze(1)
            grad_b.index_add_(0, lora_indices, per_row_b)
        return grad_x, grad_a, grad_b, None, None


__all__ = ["BatchedLoraDelta"]

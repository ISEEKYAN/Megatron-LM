# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3-MoE composition helpers for dense multi-LoRA sidecars."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from megatron.lite.primitive.modules.multi_lora_bank import DenseLoraBank


class _AllReduceGradient(torch.autograd.Function):
    """Identity forward whose replicated-bank gradient is summed across EP."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad = grad_output.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


def apply_dense_lora_delta(
    bank: DenseLoraBank,
    x: torch.Tensor,
    lora_indices: torch.Tensor,
    *,
    scale: float,
    gradient_sync_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Apply a dense bank while preserving the operator's sorted-slot contract."""
    if x.ndim != 2:
        raise ValueError("MoE multi-LoRA inputs must be two-dimensional.")
    if lora_indices.ndim != 1 or lora_indices.shape[0] != x.shape[0]:
        raise ValueError("MoE multi-LoRA indices must have one entry per input row.")
    synced_bank = bank
    if gradient_sync_group is not None and dist.get_world_size(gradient_sync_group) > 1:
        synced_bank = DenseLoraBank(
            _AllReduceGradient.apply(bank.a_bank, gradient_sync_group),
            _AllReduceGradient.apply(bank.b_bank, gradient_sync_group),
        )
    # Empty EP receives still traverse both autograd wrappers: every rank must
    # execute the same shared-bank all-reduce in backward.  The zero factors
    # retain those dependencies without sending TE's unallocated empty tensor
    # through the dense LoRA operator.
    if x.shape[0] == 0:
        return (
            x.new_zeros((0, bank.b_bank.shape[1]))
            + (synced_bank.a_bank.sum() + synced_bank.b_bank.sum()) * 0
        )

    if lora_indices.numel() < 2 or bool(
        torch.all(lora_indices[1:] >= lora_indices[:-1])
    ):
        return synced_bank.delta(x, lora_indices, scale=scale)

    order = torch.argsort(lora_indices, stable=True)
    restore = torch.empty_like(order)
    restore[order] = torch.arange(order.numel(), device=order.device)
    sorted_delta = synced_bank.delta(
        x.index_select(0, order), lora_indices.index_select(0, order), scale=scale
    )
    return sorted_delta.index_select(0, restore)


@dataclass(frozen=True)
class MoELoraSidecar:
    """Per-batch LoRA banks and token slots for one Qwen3-MoE layer."""

    fc1: DenseLoraBank
    fc2: DenseLoraBank
    lora_indices: torch.Tensor
    scale: float
    # External callers provide replicated banks outside the model/optimizer
    # lifecycle, so they retain the legacy explicit EP reduction.  The model
    # builder sets this False for model-owned banks; dist-opt then owns their
    # single dense-DP/finalize reduction.
    requires_explicit_ep_sync: bool = True
    qkv: DenseLoraBank | None = None
    proj: DenseLoraBank | None = None


__all__ = ["MoELoraSidecar", "apply_dense_lora_delta"]

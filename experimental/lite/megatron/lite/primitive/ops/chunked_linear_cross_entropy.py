# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Chunked vocab-parallel LM-head projection plus cross entropy.

This is deliberately a non-fused primitive: it preserves the default Qwen3
BF16 LM-head/CE numerics while bounding vocab-shaped temporaries to one token
window.  The caller decides whether its model composition may use it.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.lite.primitive.ops.cross_entropy import (
    _DEFAULT_CHUNK_SIZE,
    _chunk_loss_and_softmax,
    _tp_info,
    _vocab_range,
)


class _ChunkedVocabParallelLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        tp_group,
        sequence_parallel: bool,
        temperature: float,
        chunk_size: int,
    ) -> torch.Tensor:
        rank, world_size = _tp_info(tp_group)
        if sequence_parallel and world_size > 1:
            total_hidden = torch.empty(
                (hidden.shape[0] * world_size, *hidden.shape[1:]),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            dist.all_gather_into_tensor(
                total_hidden, hidden.contiguous(), group=tp_group
            )
        else:
            total_hidden = hidden

        if total_hidden.shape[:-1] != target.shape:
            raise ValueError(
                "hidden leading shape must match target shape after sequence gathering, "
                f"got {total_hidden.shape[:-1]} and {target.shape}"
            )
        local_vocab = weight.shape[0]
        vocab_start, vocab_end = _vocab_range(local_vocab, rank, world_size)
        hidden_2d = total_hidden.reshape(-1, total_hidden.shape[-1])
        target_1d = target.reshape(-1)
        loss = torch.empty(target_1d.shape, dtype=torch.float32, device=hidden.device)
        for start in range(0, target_1d.numel(), chunk_size):
            end = min(start + chunk_size, target_1d.numel())
            logits = hidden_2d[start:end].matmul(weight.t())
            if temperature != 1.0:
                logits = logits / temperature
            loss[start:end], _, _, _ = _chunk_loss_and_softmax(
                logits,
                target_1d[start:end],
                tp_group,
                vocab_start,
                vocab_end,
                return_softmax=False,
            )

        ctx.save_for_backward(total_hidden, weight, target)
        ctx.tp_group = tp_group
        ctx.sequence_parallel = sequence_parallel
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.vocab_start = vocab_start
        ctx.vocab_end = vocab_end
        ctx.world_size = world_size
        ctx.local_sequence = hidden.shape[0]
        return loss.reshape_as(target)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        total_hidden, weight, target = ctx.saved_tensors
        hidden_2d = total_hidden.reshape(-1, total_hidden.shape[-1])
        target_1d = target.reshape(-1)
        grad_output_1d = grad_output.reshape(-1)
        grad_hidden_full = torch.empty_like(total_hidden).reshape_as(hidden_2d)
        grad_weight = torch.zeros_like(weight)

        for start in range(0, target_1d.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, target_1d.numel())
            logits = hidden_2d[start:end].matmul(weight.t())
            if ctx.temperature != 1.0:
                logits = logits / ctx.temperature
            _, softmax, target_mask, masked_target = _chunk_loss_and_softmax(
                logits,
                target_1d[start:end],
                ctx.tp_group,
                ctx.vocab_start,
                ctx.vocab_end,
                return_softmax=True,
            )
            row_indices = torch.arange(end - start, device=softmax.device)
            softmax[row_indices, masked_target] -= 1.0 - target_mask.float()
            softmax.mul_(grad_output_1d[start:end].unsqueeze(-1))
            if ctx.temperature != 1.0:
                softmax.div_(ctx.temperature)
            # CE owns FP32 probability math; the vanilla BF16 head backward
            # consumes a BF16 dlogits, exactly as the non-chunked path does.
            grad_logits = softmax.to(dtype=weight.dtype)
            grad_hidden_full[start:end].copy_(grad_logits.matmul(weight))
            grad_weight.add_(grad_logits.t().matmul(hidden_2d[start:end]))

        if ctx.sequence_parallel and ctx.world_size > 1:
            grad_hidden = torch.empty(
                (ctx.local_sequence, *total_hidden.shape[1:]),
                dtype=total_hidden.dtype,
                device=total_hidden.device,
            )
            # This one collective is both the TP sum and SP sequence split.
            dist.reduce_scatter_tensor(
                grad_hidden,
                grad_hidden_full.reshape_as(total_hidden).contiguous(),
                group=ctx.tp_group,
            )
        elif ctx.world_size > 1:
            grad_hidden = grad_hidden_full.reshape_as(total_hidden)
            dist.all_reduce(grad_hidden, group=ctx.tp_group)
        else:
            grad_hidden = grad_hidden_full.reshape_as(total_hidden)
        return grad_hidden, grad_weight, None, None, None, None, None


def chunked_vocab_parallel_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    *,
    tp_group=None,
    sequence_parallel: bool = False,
    temperature: float = 1.0,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Return per-token CE without materializing a full vocab-shaped tensor."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError(f"chunk_size must be an integer, got {chunk_size!r}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if hidden.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"hidden and weight feature dimensions differ: {hidden.shape[-1]} and {weight.shape[-1]}"
        )
    return _ChunkedVocabParallelLinearCrossEntropy.apply(
        hidden, weight, target, tp_group, sequence_parallel, temperature, chunk_size
    )


__all__ = ["chunked_vocab_parallel_linear_cross_entropy"]

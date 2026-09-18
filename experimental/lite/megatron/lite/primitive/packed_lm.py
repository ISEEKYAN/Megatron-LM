# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed next-token loss normalization, CP targets, and output reconstruction."""

import math
from dataclasses import replace

import torch
from megatron.lite.primitive.parallel.thd import roll_packed_thd_left
from torch.nn import functional as F


def prepare_microbatches(data_iter, count, *, dp_group=None):
    """Use one valid-token denominator for all SFT microbatches."""
    from megatron.lite.runtime.contracts.loss import LossContext, split_loss_context

    if count < 1:
        raise ValueError('Microbatch count must be positive')
    items = [split_loss_context(next(data_iter)) for _ in range(count)]
    total = 0.0
    for batch, _ in items:
        if batch.labels is None:
            raise ValueError('SFT normalization requires labels')
        mask = (
            torch.ones_like(batch.labels, dtype=torch.float32)
            if batch.loss_mask is None
            else batch.loss_mask
        )
        if (
            mask.shape != batch.input_ids.shape
            or not torch.isfinite(mask).all()
            or (mask < 0).any()
        ):
            raise ValueError('Expected finite nonnegative token loss weights')
        # Count exactly the weights consumed by _text_output's next-token CE.
        # The packed shift zeros each sequence tail (there is no next label),
        # so the original first-token weight does not contribute.
        shifted_mask, _ = roll_packed_thd_left(mask, cu_seqlens_padded=batch.cu_seqlens)
        total += float(shifted_mask.sum())
    # The generic runtime divides every microbatch by count after this loss.
    dp_size = 1
    if dp_group is not None:
        tokens = torch.tensor(
            total, dtype=torch.float64, device=items[0][0].input_ids.device
        )
        torch.distributed.all_reduce(tokens, group=dp_group)
        total = float(tokens)
        dp_size = torch.distributed.get_world_size(dp_group)
    denominator = max(total, 1.0) / (count * dp_size)
    return [
        (
            batch,
            replace(context or LossContext(), normalization_denominator=denominator),
        )
        for batch, context in items
    ]


def _cp_targets(batch, cp_context):
    """Shift full documents once, then optionally select this CP rank's tokens."""
    mask = (
        torch.ones_like(batch.labels, dtype=torch.float32)
        if batch.loss_mask is None
        else batch.loss_mask
    )
    labels, mask = (
        roll_packed_thd_left(value, cu_seqlens_padded=batch.cu_seqlens)[0]
        for value in (batch.labels, mask)
    )
    denominator = mask.sum().clamp_min(1)
    if cp_context is None:
        return labels, mask, denominator
    return (
        cp_context.slice(labels, seq_dim=0),
        cp_context.slice(mask, seq_dim=0),
        denominator,
    )


def text_output(logits, batch, *, cp_context=None):
    from megatron.lite.runtime.contracts.loss import get_loss_context

    context = get_loss_context()
    temperature = 1.0 if context is None else context.temperature
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    logits = logits / temperature
    result = {'logits': logits}
    if batch.labels is not None:
        if batch.labels.shape != batch.input_ids.shape:
            raise ValueError('Labels must match packed input shape')
        if batch.loss_mask is not None and batch.loss_mask.shape != batch.labels.shape:
            raise ValueError('Loss mask must match packed input shape')
        labels, mask, denominator = _cp_targets(batch, cp_context)
        token_loss = F.cross_entropy(logits, labels, reduction='none')
        if context is not None and context.normalization_denominator is not None:
            denominator = context.normalization_denominator
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError('Loss denominator must be finite and positive')
        result['loss'] = (token_loss * mask).sum() / denominator
        if cp_context is not None:
            # DDP averages the disjoint CP token contributions.
            result['loss'] = result['loss'] * cp_context.size
        if context is not None:
            result['loss'] = result['loss'] * context.loss_scale
        if context is None or context.return_log_probs:
            result['log_probs'] = -token_loss
    if context is not None and context.calculate_entropy:
        log_probs = logits.log_softmax(-1)
        result['entropy'] = -(log_probs.exp() * log_probs).sum(-1)
    return result


def unpack_forward_output(model, batch, output):
    if isinstance(output, dict):
        return {
            key: unpack_forward_output(model, batch, value)
            for key, value in output.items()
        }
    if model.ps.cp_size > 1 and isinstance(output, torch.Tensor) and output.ndim > 0:
        from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

        cp_context = ContiguousCPSequence(
            batch.total_tokens, model.ps.cp_rank, model.ps.cp_size, model.ps.cp_group
        )
        output = cp_context.gather(output, seq_dim=0)
    if (
        isinstance(output, torch.Tensor)
        and output.ndim > 0
        and output.shape[0] == batch.total_tokens
    ):
        return torch.nested.as_nested_tensor(
            list(output.split(batch.seq_lens.tolist()))
        )
    return output

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SFT normalization against independent, unpacked next-token CE."""

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.runtime.contracts import PackedBatch
from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context
from torch.nn import functional as F


@pytest.mark.parametrize('mask_kind', ['none', 'ones', 'weighted', 'zero'])
def test_packed_sft_matches_unpacked_weighted_mean(mask_kind):
    lengths = [1, 3, 5]
    ids = torch.arange(sum(lengths)) % 4
    weights = {
        'none': None,
        'ones': torch.ones(9, dtype=torch.float64),
        'weighted': torch.tensor(
            [8, 4, 0.5, 1.5, 2, 0, 1, 0.5, 0.5], dtype=torch.float64
        ),
        'zero': torch.zeros(9, dtype=torch.float64),
    }[mask_kind]
    batch = PackedBatch(ids, ids, torch.tensor(lengths), weights)
    logits = (
        torch.arange(36, dtype=torch.float64).reshape(9, 4) % 7 / 8
    ).requires_grad_()
    serial_logits = logits.detach().clone().requires_grad_()
    serial_mask = torch.ones(9, dtype=torch.float64) if weights is None else weights
    # Dense per-sequence baseline: predict labels[1:] from logits[:-1].
    numerators, counts, token_losses = [], [], []
    for scores, labels, mask in zip(
        serial_logits.split(lengths), ids.split(lengths), serial_mask.split(lengths)
    ):
        ce = F.cross_entropy(scores[:-1], labels[1:], reduction='none')
        numerators.append((ce * mask[1:]).sum())
        counts.append(mask[1:].sum())
        token_losses.append(ce)
    total = torch.stack(counts).sum().clamp_min(1)
    reference = torch.stack(numerators).sum() / total
    local = protocol._text_output(logits, batch)
    offset = 0
    for length, ce in zip(lengths, token_losses):
        torch.testing.assert_close(
            -local['log_probs'][offset : offset + length - 1],
            ce,
            rtol=0,
            atol=0,
            msg='packed versus unpacked token CE',
        )
        offset += length
    # Grouped sums may differ by float64 reduction rounding; token CE is exact above.
    torch.testing.assert_close(
        local['loss'], reference, rtol=0, atol=1e-15, msg='packed local weighted mean'
    )
    # Uneven microbatches, each itself packed; preserve the caller's loss policy.
    batches = [
        PackedBatch(
            ids[:4],
            ids[:4],
            torch.tensor([1, 3]),
            None if weights is None else weights[:4],
        ),
        PackedBatch(
            ids[4:],
            ids[4:],
            torch.tensor([5]),
            None if weights is None else weights[4:],
        ),
    ]
    context = LossContext(source_batch='source', return_log_probs=False)
    prepared = protocol.prepare_microbatches(iter((b, context) for b in batches), 2)
    actual = []
    for (microbatch, ctx), scores in zip(prepared, logits.split([4, 5])):
        assert ctx.source_batch == 'source' and not ctx.return_log_probs
        assert ctx.normalization_denominator == float(total) / 2, 'CE token denominator'
        with use_loss_context(ctx):
            actual.append(protocol._text_output(scores, microbatch)['loss'] / 2)
    loss = sum(actual)
    torch.testing.assert_close(
        loss,
        reference,
        rtol=0,
        atol=1e-15,
        msg='prepared packed versus unpacked weighted mean',
    )
    loss.backward()
    reference.backward()
    torch.testing.assert_close(
        logits.grad,
        serial_logits.grad,
        rtol=0,
        atol=0,
        msg='packed versus unpacked per-token gradient',
    )

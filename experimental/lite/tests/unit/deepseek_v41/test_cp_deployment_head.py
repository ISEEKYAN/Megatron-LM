# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment head under contiguous CP: each rank's rows equal the unsplit rows.

The head is row-wise (batch-invariant GEMM and log-softmax per token), so a CP
rank's logits for its slice give exactly the CP1 log-probs of those tokens; the
labels/mask are shifted on the full document first and then sliced.
"""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('cp', [2, 3])
def test_cp_rank_log_probs_equal_unsplit_rows(v41_core_te, cp):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence
    from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context

    generator = torch.Generator().manual_seed(4)
    tokens, vocab = 13, 11
    logits = torch.randn(tokens, vocab, generator=generator, dtype=torch.float64)
    ids = torch.randint(0, vocab, (tokens,), generator=generator)
    mask = (torch.rand(tokens, generator=generator) > 0.2).double()
    batch = SimpleNamespace(
        input_ids=ids,
        labels=ids,
        loss_mask=mask,
        cu_seqlens=torch.tensor([0, 6, tokens], dtype=torch.int32),
    )
    denominator = float(protocol._cp_targets(batch, None)[2])
    context = LossContext(normalization_denominator=denominator)
    with use_loss_context(context):
        full = protocol.text_output(None, None, batch, logits=logits)
    total = torch.zeros((), dtype=torch.float64)
    for rank in range(cp):
        partition = ContiguousCPSequence(tokens, rank, cp)
        local_logits = partition.slice(logits[None], seq_dim=1)[0]
        with use_loss_context(context):
            local = protocol.text_output(
                None, None, batch, cp_context=partition, logits=local_logits
            )
        expected = partition.slice(full['log_probs'][None], seq_dim=1)[0]
        assert torch.equal(local['log_probs'], expected)
        total = total + local['loss']
    # Per-rank losses use the global denominator, so they sum to the CP1 loss.
    torch.testing.assert_close(total, full['loss'], rtol=1e-12, atol=1e-12)

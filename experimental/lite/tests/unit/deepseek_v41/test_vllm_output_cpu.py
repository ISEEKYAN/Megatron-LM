# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The deployment tail owns only the LP arithmetic over the chunked BF16-logit head."""
import functools
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import model as deployment
from megatron.lite.model.deepseek_v41.vllm.head import _HeadLogProbs
from megatron.lite.model.deepseek_v41.vllm.model import DeepseekV41Model as Deployment


def _cpu_head(hidden, weight, labels, temperature, *, entropy, rows=1):
    return _HeadLogProbs.apply(hidden, weight, labels, temperature, entropy, rows)


def test_masked_lp_entropy_and_logits_over_the_deployment_hidden():
    torch.manual_seed(5)
    model = object.__new__(Deployment)
    torch.nn.Module.__init__(model)
    model.layout = NS(has_embed=True, has_head=True)
    model.ps = NS(tp_size=1, cp_size=1, cp_rank=0)
    model.engrams, model.release = {}, NS()
    model.head = NS(weight=torch.randn(6, 4, requires_grad=True))
    hidden = torch.randn(3, 4, requires_grad=True)
    labels, mask = torch.tensor([1, 0, 5]), torch.tensor([1.0, 0.0, 1.0])
    packed = NS(cu_seqlens_q_padded=torch.tensor([0, 2, 3]), cu_seqlens_q=None)
    seen = {}

    def model_forward(ids, positions, *args, **kwargs):
        seen['positions'] = positions
        return hidden

    run = functools.partial(model, torch.tensor([0, 1, 2]), packed_seq_params=packed)
    with (
        patch.object(Deployment, '_params', lambda self: None),
        patch.object(Deployment, '_rope', lambda self: None),
        patch.object(Deployment, '_core', lambda self: (None, 'mega')),
        patch.object(deployment.F, 'model_forward', model_forward),
        patch.object(deployment, 'head_log_probs', _cpu_head),
    ):
        output = run(labels=labels, loss_mask=mask, temperature=0.5, calculate_entropy=True)
        logits = run()['logits']
        model.layout.has_head = False
        with pytest.raises(NotImplementedError, match='PP=1'):
            run(labels=labels)
    assert seen['positions'].tolist() == [0, 1, 0]
    reference = hidden @ model.head.weight.T
    assert torch.equal(logits, reference.bfloat16())
    expected = (reference.bfloat16().float() / 0.5).log_softmax(-1)
    torch.testing.assert_close(output['log_probs'], expected[torch.arange(3), labels])
    torch.testing.assert_close(output['loss'], -(expected[0, 1] + expected[2, 5]) / 2)
    torch.testing.assert_close(output['entropy'], -(expected.exp() * expected).sum(-1))
    assert torch.equal(output['valid_mask'], mask.bool())
    assert torch.equal(output['rows'], torch.arange(3))
    output['loss'].backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad[1].count_nonzero() == 0

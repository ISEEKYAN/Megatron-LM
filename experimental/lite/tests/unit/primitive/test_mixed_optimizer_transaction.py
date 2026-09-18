# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Candidate checks must precede publication of weights, state and table bytes."""

from types import SimpleNamespace

import pytest
import torch
from megatron.lite.primitive.optimizers import headwise_muon as mixed


@pytest.mark.parametrize('fault', ['adam_state', 'parameter', 'weight', 'scale'])
def test_nonfinite_candidate_does_not_publish(fault, monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(2, 32))
    table = SimpleNamespace(
        master=parameter, weight=torch.ones(2, 32), scale=torch.ones(2, 1)
    )
    config = SimpleNamespace(lr=0.01, clip_grad=0)
    optimizer = mixed.MixedOptimizer(
        [{'params': [parameter], 'algorithm': 'adamw'}], config, [table]
    )
    gradient = torch.ones_like(parameter)
    parameter.grad = parameter.main_grad = gradient
    original_step = torch.optim.AdamW.step
    original_quantize = mixed.quantize_block_fp8

    def step(candidate):
        original_step(candidate)
        owner = candidate.param_groups[0]['params'][0]
        if fault == 'adam_state':
            candidate.state[owner]['exp_avg'].fill_(float('inf'))
        elif fault == 'parameter':
            owner.data.fill_(float('inf'))

    def quantize(*args, **kwargs):
        values = list(original_quantize(*args, **kwargs))
        if fault in ('weight', 'scale'):
            index = 0 if fault == 'weight' else 1
            values[index] = values[index].float().fill_(float('nan'))
        return values

    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    monkeypatch.setattr(mixed, 'quantize_block_fp8', quantize)
    assert not optimizer.step()[0], 'NONFINITE_CANDIDATE_MUST_ABORT'
    assert torch.equal(parameter, torch.ones_like(parameter)), 'WEIGHT_NOT_PUBLISHED'
    assert not optimizer.optimizers[0].state, 'ADAM_STATE_NOT_PUBLISHED'
    assert torch.equal(table.weight, torch.ones_like(table.weight))
    assert torch.equal(table.scale, torch.ones_like(table.scale))
    assert parameter.grad is gradient and parameter.main_grad is gradient

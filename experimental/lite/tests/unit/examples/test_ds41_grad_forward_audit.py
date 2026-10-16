# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""C5 accepts the real differentiable forward and rejects one-ULP drift."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def compare():
    source = (
        Path(__file__).resolve().parents[3]
        / 'examples/verl/scripts/ds41_w4a8_proxy/audit/grad_forward.py'
    )
    spec = importlib.util.spec_from_file_location('ds41_grad_forward_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare


def test_grad_forward_exact_valid_tokens_preserve_autograd(compare):
    actor = torch.tensor([[-2.0, -3.0]], requires_grad=True)
    rollout = torch.tensor([[-2.0, -99.0]])
    receipt = compare(
        actor, rollout, rollout, torch.tensor([[1, 0]]), grad_enabled=True
    )
    assert torch.equal(receipt['actor_log_probs'], torch.tensor([-2.0]))
    assert not receipt['actor_log_probs'].requires_grad
    actor.sum().backward()
    assert torch.equal(actor.grad, torch.ones_like(actor))


@pytest.mark.parametrize('changed', ['rollout', 'old'])
def test_grad_forward_rejects_single_ulp_in_either_paired_path(compare, changed):
    actor = torch.tensor([[-2.0]], requires_grad=True)
    rollout, old = actor.detach().clone(), actor.detach().clone()
    target = rollout if changed == 'rollout' else old
    target[0, 0] = torch.nextafter(target[0, 0], torch.tensor(0.0))
    with pytest.raises(AssertionError, match='logprob mismatch'):
        compare(actor, rollout, old, torch.ones_like(actor), grad_enabled=True)


@pytest.mark.parametrize('grad_enabled,requires_grad', [(False, True), (True, False)])
def test_grad_forward_rejects_inference_only_substitute(
    compare, grad_enabled, requires_grad
):
    actor = torch.tensor([[-2.0]], requires_grad=requires_grad)
    with pytest.raises(AssertionError, match='grad-enabled'):
        compare(
            actor,
            actor.detach(),
            actor.detach(),
            torch.ones_like(actor),
            grad_enabled=grad_enabled,
        )


@pytest.mark.parametrize('invalid', ['empty', 'nonfinite', 'shape'])
def test_grad_forward_fails_closed_on_invalid_evidence(compare, invalid):
    actor = torch.tensor([[-2.0]], requires_grad=True)
    rollout = actor.detach().clone()
    mask = torch.ones_like(actor)
    if invalid == 'empty':
        mask.zero_()
    elif invalid == 'nonfinite':
        rollout.fill_(float('nan'))
    else:
        rollout = rollout.flatten()
    with pytest.raises(AssertionError):
        compare(actor, rollout, actor.detach(), mask, grad_enabled=True)

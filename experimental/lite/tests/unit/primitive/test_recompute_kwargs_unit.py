# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint parity against ordinary autograd, without CUDA/TE dependencies."""
import copy
import inspect

import pytest
import torch
from megatron.lite.primitive.recompute import wrap_checkpoint
from torch import nn

pytestmark = pytest.mark.mlite


class KeywordLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1.0, 5.0))
        self.calls = 0

    def forward(self, *, hidden, config=None):
        self.calls += 1
        scale = 1.0 if config is None else config['scale']
        return hidden * self.weight * scale


def test_keyword_only_keeps_input_and_parameter_gradients():
    layer = KeywordLayer()
    original_signature = inspect.signature(layer.forward)
    wrap_checkpoint(layer, preserve_rng_state=False)
    x = torch.ones(4, requires_grad=True)
    out = layer(hidden=x, config={'scale': 2.0})
    assert out.requires_grad
    out.sum().backward()
    assert torch.equal(x.grad, torch.arange(1.0, 5.0) * 2)
    assert torch.equal(layer.weight.grad, torch.full((4,), 2.0))
    assert layer.calls == 2
    assert inspect.signature(layer.forward) == original_signature


class AliasLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def forward(self, x, *, duplicate, nested, scale, label):
        assert duplicate is x
        assert nested['same'][0] is x
        assert nested['frozen'].requires_grad is False
        assert (
            nested['view'].untyped_storage().data_ptr()
            == x.untyped_storage().data_ptr()
        )
        assert label == 'alias'
        return (
            (x + duplicate + nested['same'][0] + nested['view'] + nested['frozen'])
            * self.weight
            * scale
        )


def _alias_run(layer):
    x = torch.arange(4.0, requires_grad=True)
    frozen = torch.ones(4)
    out = layer(
        x,
        duplicate=x,
        nested={'same': [x], 'view': x.view(4), 'frozen': frozen},
        scale=0.5,
        label='alias',
    )
    out.sum().backward()
    assert frozen.grad is None
    return out.detach(), x.grad, layer.weight.grad


def test_alias_duplicate_view_nested_kwargs_and_non_tensors_match_reference():
    reference = AliasLayer()
    wrapped = copy.deepcopy(reference)
    wrap_checkpoint(wrapped, preserve_rng_state=False)
    assert all(
        torch.equal(a, b) for a, b in zip(_alias_run(reference), _alias_run(wrapped))
    )


@pytest.mark.parametrize('input_requires_grad', [False, True])
def test_frozen_input_flags_and_parameter_gradient_parity(input_requires_grad):
    reference = KeywordLayer()
    wrapped = copy.deepcopy(reference)
    wrap_checkpoint(wrapped, preserve_rng_state=False)
    results = []
    for layer in (reference, wrapped):
        x = torch.ones(4, requires_grad=input_requires_grad)
        y = layer(hidden=x)
        y.sum().backward()
        results.append((y.detach(), x.grad, layer.weight.grad))
    for a, b in zip(*results):
        assert (a is None and b is None) or torch.equal(a, b)


class RandomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(7))
        self.draws = []

    def forward(self, x=None, *, hidden=None):
        x = hidden if hidden is not None else x
        noise = torch.rand_like(x)
        self.draws.append(noise.clone())
        return x * self.weight * noise


@pytest.mark.parametrize('keyword', [False, True])
@pytest.mark.parametrize('preserve_rng_state', [False, True])
def test_rng_output_input_and_parameter_gradient_parity(keyword, preserve_rng_state):
    reference = RandomLayer()
    wrapped = copy.deepcopy(reference)
    wrap_checkpoint(wrapped, preserve_rng_state=preserve_rng_state)
    results = []
    for layer in (reference, wrapped):
        torch.manual_seed(123)
        x = torch.ones(7, requires_grad=True)
        y = layer(hidden=x) if keyword else layer(x)
        after_forward = torch.get_rng_state().clone()
        y.sum().backward()
        results.append((y.detach(), x.grad, layer.weight.grad, torch.get_rng_state()))
        if preserve_rng_state:
            assert torch.equal(after_forward, torch.get_rng_state())
    assert torch.equal(results[0][0], results[1][0])
    assert len(wrapped.draws) == 2
    if preserve_rng_state:
        assert all(torch.equal(a, b) for a, b in zip(*results))
        assert torch.equal(*wrapped.draws)
    else:
        assert not torch.equal(*wrapped.draws)
        assert not torch.equal(results[0][1], results[1][1])


def test_no_grad_call_remains_no_grad():
    layer = KeywordLayer()
    wrap_checkpoint(layer, preserve_rng_state=False)
    with torch.no_grad():
        out = layer(hidden=torch.ones(4, requires_grad=True))
    assert not out.requires_grad


def test_rng_restored_when_replay_raises():
    class FailsDuringReplay(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *, hidden):
            self.calls += 1
            noise = torch.rand_like(hidden)
            if self.calls == 2:
                raise RuntimeError('injected replay error')
            return hidden * noise

    layer = FailsDuringReplay()
    wrap_checkpoint(layer)
    out = layer(hidden=torch.ones(4, requires_grad=True))
    after_forward = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='injected replay error'):
        out.sum().backward()
    assert torch.equal(after_forward, torch.get_rng_state())

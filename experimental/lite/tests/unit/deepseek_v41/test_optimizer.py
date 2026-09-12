# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import math
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite.optimizer_groups import (
    VisionOptimizerConfig,
    parameter_groups,
)
from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn, sinkhorn_direction


@pytest.mark.parametrize(
    'case', ['valid', 'unknown', 'unknown_visual', 'alias', 'indexer', 'head']
)
def test_v41_optimizer_rejects_unknown_alias_and_live_indexer(moe, model_config, case):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    model = DeepseekV41Model(model_config, token_map=list(range(256)), quantized=False)
    if case == 'unknown':
        model.extra = torch.nn.Parameter(torch.ones(2, 2))
    elif case == 'unknown_visual':
        model.vision.extra = torch.nn.Parameter(torch.ones(3))
        model._bind('vision.extra', model.vision, 'extra', 'vision')
    elif case == 'alias':
        model.extra = model.layers[0].attn.wq_a.weight
    elif case == 'indexer':
        model.layers[2].attn.indexer.requires_grad_(True)
    elif case == 'head':
        key = 'layers.0.attn.wq_b.weight'
        model.tensor_bindings[key] = replace(
            model.tensor_bindings[key], head_count=None
        )
    policy = VisionOptimizerConfig(0.5, 1.0, 0.0)
    if case != 'valid':
        message = dict(
            unknown='Every parameter must have exactly one binding',
            unknown_visual='Unknown parameter owner',
            alias='alias',
            indexer='indexer must remain frozen',
            head='head count',
        )[case]
        with pytest.raises(ValueError, match=message):
            parameter_groups(model, lr=0.001, vision_policy=policy)
    else:
        groups = parameter_groups(model, lr=0.001, vision_policy=policy)
        routed = {id(p): g['algorithm'] for g in groups for p in g['params']}
        assert set(routed) == {id(p) for p in model.parameters() if p.requires_grad}
        assert routed[id(model.aligner.w1.weight)] == 'muon'
        assert (
            routed[id(model.vision.norm.weight)]
            == routed[id(model.image_start)]
            == 'adamw'
        )
        assert all(
            id(p) not in routed
            for layer in model.layers
            if layer.attn.indexer is not None
            for p in layer.attn.indexer.parameters()
        )


def scalar_direction(matrix):
    # Independent Python arithmetic: constants never imported from production.
    u = [[float(x) for x in row] for row in matrix]
    rho = [math.sqrt(math.fsum(x * x for x in row)) for row in u]
    for i, value in enumerate(rho):
        if value <= 0.001 * math.fsum(rho) / len(rho):
            u[i] = [0.0] * len(u[i])
    trace = []
    for iteration in range(11):
        if iteration % 2:
            u = list(map(list, zip(*u)))
        u = [
            [x / (math.sqrt(math.fsum(y * y for y in row)) + 1e-20) for x in row]
            for row in u
        ]
        if iteration % 2:
            u = list(map(list, zip(*u)))
        trace.append(torch.tensor(u, dtype=torch.float64))
    return trace[-1] * math.sqrt(len(u[0])), trace


@pytest.mark.parametrize(
    'matrix',
    [
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        [[1.0], [1999.0]],
        [[1.0], [1999.0], [0.0], [0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[1e-20]],
        [[1e30, -2e30], [0.0, 1e30]],
    ],
)
def test_algorithm1_constants_mask_and_normalization(matrix):
    expected, reference_trace = scalar_direction(matrix)
    trace = []
    actual = sinkhorn_direction(
        torch.tensor(matrix), trace=lambda _, value: trace.append(value.double())
    )
    assert len(trace) == 11, 'Algorithm1 K'
    for a, b in zip([actual.double(), *trace], [expected, *reference_trace]):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('restore', [False, True])
def test_algorithm1_momentum_fresh_n_and_resume(restore):
    p = torch.nn.Parameter(torch.arange(12).reshape(4, 3).float() / 8)
    opt = Sinkhorn([{'params': [p], 'multiplier': 5.0}], lr=0.02)
    expected, momentum = p.detach().double(), torch.zeros_like(p, dtype=torch.float64)
    for i, gradient in enumerate(
        (
            [[1, 2, -1], [0, 0, 0], [4, -3, 0.5], [1e-7, 0, 0]],
            [[0, 0, 0]] * 4,
            [[-2, 1, 3], [4, 3, -2], [0, 0, 0], [5, 1, -3]],
        )
    ):
        p.grad = torch.tensor(gradient, dtype=torch.float32)
        momentum = 0.95 * momentum + 0.05 * p.grad.double()
        n = 0.95 * momentum + 0.05 * p.grad.double()
        direction, _ = scalar_direction(n.tolist())
        expected = expected - 0.18 * 0.02 * 5 * direction
        assert opt.step()
        torch.testing.assert_close(p.double(), expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            opt.state[p]['momentum'].double(), momentum, atol=1e-7, rtol=2e-6
        )
        assert set(opt.state[p]) == {'momentum'}, 'No warm-start state'
        if restore and i == 0:
            saved = deepcopy(opt.state_dict())
            opt = Sinkhorn([{'params': [p], 'multiplier': 5.0}], lr=0.02)
            opt.load_state_dict(saved)
    before = p.detach().clone()
    p.grad.fill_(float('nan'))
    assert not opt.step() and torch.equal(p, before)


@pytest.mark.parametrize(
    'action,prepared,message',
    [
        ('candidates', False, 'No prepared Sinkhorn step'),
        ('commit_step', False, 'No prepared Sinkhorn step'),
        ('state_dict', True, 'Cannot checkpoint a prepared Sinkhorn step'),
        ('load_state_dict', True, 'Cannot restore a prepared Sinkhorn step'),
        ('step', False, 'Sinkhorn requires explicit accumulated gradients'),
    ],
)
def test_staged_optimizer_guards(action, prepared, message):
    opt = Sinkhorn([torch.nn.Parameter(torch.ones(2, 2))], lr=0.1)
    saved = opt.state_dict()
    if prepared:
        assert opt.prepare_step()
    args = (
        (saved,)
        if action == 'load_state_dict'
        else ((lambda: None,) if action == 'step' else ())
    )
    with pytest.raises((ValueError, RuntimeError), match=message):
        getattr(opt, action)(*args)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import math
from copy import deepcopy

import pytest
import torch
from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn, sinkhorn_direction


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
def test_staged_optimizer_guards(action, prepared, message, monkeypatch):
    opt = Sinkhorn([torch.nn.Parameter(torch.ones(2, 2))], lr=0.1)
    saved = opt.state_dict()
    if prepared:
        assert opt.prepare_step()
    args = (
        (saved,)
        if action == 'load_state_dict'
        else ((lambda: None,) if action == 'step' else ())
    )
    reads = []
    if action in ('candidates', 'commit_step'):
        original_get = type(opt).__getattribute__

        def read_prepared(self, name):
            value = original_get(self, name)
            if self is opt and name == '_prepared':
                reads.append(value)
                # The first read checks the guard; a second read consumes candidates.
                assert len(reads) == 1, 'STAGED_GUARD_PRECEDES_CANDIDATE_READ'
            return value

        monkeypatch.setattr(type(opt), '__getattribute__', read_prepared)
    with pytest.raises((ValueError, RuntimeError), match=message):
        try:
            getattr(opt, action)(*args)
        except TypeError as error:
            pytest.fail(f'STAGED_OPTIMIZER_GUARD_BEFORE_CONSUME: {error}')
    if action in ('candidates', 'commit_step'):
        assert reads == [None], 'STAGED_UNPREPARED_GUARD_EXECUTED'

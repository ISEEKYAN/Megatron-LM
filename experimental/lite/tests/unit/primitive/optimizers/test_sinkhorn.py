# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A4 Algorithm 1: independent scalar arithmetic and optimizer state transitions."""

import math

import pytest
import torch


def scalar_direction(matrix):
    """Float64 Python scalars; no production normalization or distributed helpers."""
    u = [[float(value) for value in row] for row in matrix]
    rows, columns = len(u), len(u[0])
    rho = [math.sqrt(math.fsum(value * value for value in row)) for row in u]
    threshold = 0.001 * math.fsum(rho) / rows
    for i in range(rows):
        if rho[i] <= threshold:
            u[i] = [0.0] * columns
    for iteration in range(11):
        if iteration % 2 == 0:
            for i in range(rows):
                denominator = math.sqrt(math.fsum(x * x for x in u[i])) + 1e-20
                u[i] = [x / denominator for x in u[i]]
        else:
            for j in range(columns):
                denominator = (
                    math.sqrt(math.fsum(u[i][j] ** 2 for i in range(rows))) + 1e-20
                )
                for i in range(rows):
                    u[i][j] /= denominator
    return torch.tensor(u, dtype=torch.float64) * math.sqrt(columns)


def scalar_step(weight, momentum, gradient, lr, multiplier=1):
    m = [
        [0.95 * old + 0.05 * g for old, g in zip(mr, gr)]
        for mr, gr in zip(momentum, gradient)
    ]
    n = [
        [0.95 * old + 0.05 * g for old, g in zip(mr, gr)] for mr, gr in zip(m, gradient)
    ]
    delta = scalar_direction(n)
    return torch.tensor(
        weight, dtype=torch.float64
    ) - 0.18 * lr * multiplier * delta, torch.tensor(m, dtype=torch.float64)


@pytest.mark.parametrize(
    'matrix',
    [
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        [[1.0], [1999.0], [0.0], [0.0]],
        [[1.0], [1999.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[1e-20]],
    ],
)
def test_a4_fresh_direction(matrix):
    from megatron.lite.primitive.optimizers.sinkhorn import sinkhorn_direction

    actual = sinkhorn_direction(torch.tensor(matrix, dtype=torch.float32))
    torch.testing.assert_close(
        actual.double(), scalar_direction(matrix), atol=2e-6, rtol=2e-6
    )


@pytest.mark.parametrize('restore', [False, True])
def test_sinkhorn_optimizer_trajectory_and_native_gradient(restore):
    from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn

    parameter = torch.nn.Parameter(
        torch.arange(12, dtype=torch.float32).reshape(4, 3) / 8
    )
    optimizer = Sinkhorn([{'params': [parameter], 'multiplier': 5.0}], lr=0.02)
    expected, momentum = parameter.detach().double(), torch.zeros_like(
        parameter, dtype=torch.float64
    )
    gradients = [
        [[1, 2, -1], [0, 0, 0], [4, -3, 0.5], [1e-7, 0, 0]],
        [[0, 0, 0]] * 4,
        [[-2, 1, 3], [4, 3, -2], [0, 0, 0], [5, 1, -3]],
    ]
    for step, gradient in enumerate(gradients):
        parameter.grad = torch.tensor(gradient, dtype=torch.float32)
        expected, momentum = scalar_step(
            expected.tolist(), momentum.tolist(), gradient, 0.02, 5
        )
        assert optimizer.step()
        torch.testing.assert_close(parameter.double(), expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            optimizer.state[parameter]['momentum'].double(),
            momentum,
            atol=1e-7,
            rtol=2e-6,
        )
        assert set(optimizer.state[parameter]) == {'momentum'}
        if restore and step == 0:
            saved = optimizer.state_dict()
            parameter = torch.nn.Parameter(parameter.detach().clone())
            optimizer = Sinkhorn([{'params': [parameter], 'multiplier': 5.0}], lr=0.02)
            optimizer.load_state_dict(saved)
    parameter.main_grad = torch.zeros_like(parameter, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='FP32'):
        optimizer.step()


def test_sinkhorn_stages_before_atomic_commit():
    from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn

    p = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = Sinkhorn([p], lr=1.0)
    p.grad = torch.ones_like(p)
    assert optimizer.prepare_step()
    assert torch.equal(p, torch.ones_like(p))
    assert not optimizer.state[p]
    optimizer.discard_step()
    assert optimizer.step()
    old = p.detach().clone()
    old_m = optimizer.state[p]['momentum'].clone()
    p.grad[0, 0] = float('nan')
    assert not optimizer.step()
    assert torch.equal(p, old)
    assert torch.equal(optimizer.state[p]['momentum'], old_m)

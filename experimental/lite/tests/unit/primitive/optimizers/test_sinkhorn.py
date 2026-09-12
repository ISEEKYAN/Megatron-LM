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
        [[1e30, -2e30], [0.0, 1e30]],
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


def test_engram_sinkhorn_publication_restore_and_retry(monkeypatch):
    from megatron.lite.model.deepseek_v41.lite import table_state
    from megatron.lite.primitive.modules import engram_lookup

    def construct():
        table = engram_lookup.ShardedEngramTable(
            torch.full((4, 32), 256.0).to(torch.float8_e4m3fn),
            torch.full((4, 1), 1 / 256).to(torch.float8_e8m0fnu),
            engram_lookup.RowLookup((0, 4)),
            trainable=True,
        )
        return table_state.EngramSinkhornState(table)

    state = construct()
    expected = state.table.master.detach().double()
    momentum = torch.zeros_like(expected)
    for step in range(3):
        gradient = torch.zeros(4, 32)
        gradient[step % 2] = torch.arange(1.0, 33.0)
        state.begin(step)
        state.accept_gradient(step, gradient)
        if step == 1:
            previous = state.table.master.detach().clone()
            with monkeypatch.context() as patch:

                def fail(*args, **kwargs):
                    raise RuntimeError('publication fault')

                patch.setattr(table_state, 'quantize_block_fp8', fail)
                with pytest.raises(RuntimeError, match='publication'):
                    state.step(lr=0.001)
            assert torch.equal(state.table.master, previous)
            assert state.active_step == step
        expected, momentum = scalar_step(
            expected.tolist(), momentum.tolist(), gradient.tolist(), 0.001, 5
        )
        assert state.step(lr=0.001)
        torch.testing.assert_close(
            state.table.master.double(), expected, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            state.momentum.double(), momentum, atol=2e-6, rtol=2e-6
        )
        assert state.main_grad.dtype == torch.float32
        saved = state.state_dict()
        state = construct()
        state.load_state_dict(saved)
        assert state.version == step + 1


def test_sinkhorn_rejects_damaged_checkpoint_without_changing_state():
    from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn

    p = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = Sinkhorn([p], lr=1.0)
    p.grad = torch.ones_like(p)
    assert optimizer.step()
    old = optimizer.state[p]['momentum'].clone()
    saved = optimizer.state_dict()
    saved['state'] = {0: {'momentum': torch.zeros(1, 2)}}
    with pytest.raises(ValueError, match='momentum'):
        optimizer.load_state_dict(saved)
    assert torch.equal(optimizer.state[p]['momentum'], old)


@pytest.mark.parametrize('matrix', [[[1e-20]], [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
def test_a4_normalization_intermediates(matrix):
    from megatron.lite.primitive.optimizers.sinkhorn import sinkhorn_direction

    observed = []
    sinkhorn_direction(
        torch.tensor(matrix, dtype=torch.float32),
        trace=lambda iteration, value: observed.append(value.double()),
    )
    assert len(observed) == 11
    if len(matrix) == 1:
        torch.testing.assert_close(
            observed[0],
            torch.tensor([[0.5]], dtype=torch.float64),
            atol=2e-6,
            rtol=2e-6,
        )
    else:
        u = [[float(v) for v in row] for row in matrix]
        for iteration in range(11):
            if iteration % 2 == 0:
                for i, row in enumerate(u):
                    norm = math.sqrt(math.fsum(v * v for v in row)) + 1e-20
                    u[i] = [v / norm for v in row]
            else:
                for j in range(2):
                    norm = math.sqrt(math.fsum(row[j] ** 2 for row in u)) + 1e-20
                    for row in u:
                        row[j] /= norm
            torch.testing.assert_close(
                observed[iteration],
                torch.tensor(u, dtype=torch.float64),
                atol=2e-6,
                rtol=2e-6,
            )


def test_engram_replica_order_cannot_silently_change_optimizer_owner():
    from megatron.lite.model.deepseek_v41.lite.parallel import EngramLayout

    layout = EngramLayout(5, ((2, 3), (0, 1)), world_size=4)
    with pytest.raises(ValueError, match='ascending'):
        layout.create_groups()

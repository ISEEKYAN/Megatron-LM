# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Four-rank NCCL Algorithm 1 against the independent A4 scalar reference."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_sinkhorn import scalar_step

pytestmark = pytest.mark.gpus(4)


def span(size, parts, rank):
    cuts = [len(range(i, size, parts)) for i in range(parts)]
    return sum(cuts[:rank]), sum(cuts[: rank + 1])


def _run(rank, rendezvous, configuration):
    from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn

    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    try:
        # Create groups collectively and deterministically; stats contain each
        # optimizer-owned row once, including intervals split between replicas.
        vertical = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        horizontal = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        row_group = column_group = replica_group = None
        replica = 0
        rows = 3 if configuration == 'empty_rows' else 5
        columns = 3 if configuration == 'empty_columns' else 4
        row_start, row_end, col_start, col_end = 0, rows, 0, columns
        if configuration in ('rows', 'empty_rows'):
            row_start, row_end = span(rows, 4, rank)
            row_group = dist.group.WORLD
        elif configuration in ('columns', 'empty_columns'):
            col_start, col_end = span(columns, 4, rank)
            column_group = dist.group.WORLD
        elif configuration == 'row_columns':
            row_start, row_end = span(5, 2, rank // 2)
            col_start, col_end = span(4, 2, rank % 2)
            row_group, column_group = vertical[rank % 2], horizontal[rank // 2]
        elif configuration == 'row_replicas':
            replica = rank // 2
            row_start, row_end = span(5, 2, rank % 2)
            row_group, replica_group = dist.group.WORLD, vertical[rank % 2]
        elif configuration == 'column_replicas':
            replica = rank // 2
            col_start, col_end = span(4, 2, rank % 2)
            row_group = replica_group = vertical[rank % 2]
            column_group = horizontal[rank // 2]
        else:
            raise AssertionError(configuration)
        cut = (slice(row_start, row_end), slice(col_start, col_end))
        expected = (
            torch.arange(rows * columns, dtype=torch.float64).reshape(rows, columns)
            / 16
        )
        momentum = torch.zeros_like(expected)
        parameter = torch.nn.Parameter(expected[cut].float().cuda())
        options = dict(
            lr=0.03125,
            row_group=row_group,
            column_group=column_group,
            replica_group=replica_group,
        )
        optimizer = Sinkhorn([{'params': [parameter], 'multiplier': 5.0}], **options)
        coefficients = (0.25, 0.75)
        gradients = [
            [[1, 0, 0, 0], [1999, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0]] * 5,
            [
                [1, 2, -3, 4],
                [-2, 5, 1, 0],
                [0, 0, 0, 0],
                [2, -4, 5, 1],
                [1e-8, 0, 0, 0],
            ],
            [[0, 0, 0, 0]] * 5,
        ]
        for step, gradient in enumerate(gradients):
            gradient = [row[:columns] for row in gradient[:rows]]
            full = torch.tensor(gradient, dtype=torch.float32)
            # Real backward creates native FP32 gradients on each replica.
            parameter.grad = None
            contribution = full[cut].cuda()
            if replica_group is not None:
                contribution *= coefficients[replica]
            (parameter * contribution).sum().backward()
            assert parameter.grad.dtype == torch.float32
            expected, momentum = scalar_step(
                expected.tolist(), momentum.tolist(), gradient, options['lr'], 5
            )
            assert optimizer.step()
            torch.testing.assert_close(
                parameter.cpu().double(), expected[cut], atol=3e-6, rtol=3e-6
            )
            local_m = momentum[cut]
            if replica_group is not None:
                begin, end = span(row_end - row_start, 2, replica)
                local_m = local_m[begin:end]
            actual_m = optimizer.state[parameter]['momentum']
            assert actual_m.is_cuda and actual_m.dtype == torch.float32
            assert actual_m.numel() == local_m.numel()
            torch.testing.assert_close(
                actual_m.cpu().double(), local_m, atol=2e-5, rtol=3e-6
            )
            if step == 1:
                saved = optimizer.state_dict()
                parameter = torch.nn.Parameter(parameter.detach().clone())
                optimizer = Sinkhorn(
                    [{'params': [parameter], 'multiplier': 5.0}], **options
                )
                optimizer.load_state_dict(saved)
        before = parameter.detach().clone()
        old_m = optimizer.state[parameter]['momentum'].clone()
        parameter.grad = torch.zeros_like(parameter)
        if rank == 0:
            parameter.grad[0, 0] = float('inf')
        assert not optimizer.step()
        assert torch.equal(parameter, before)
        assert torch.equal(optimizer.state[parameter]['momentum'], old_m)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    'configuration',
    [
        'rows',
        'columns',
        'row_columns',
        'row_replicas',
        'column_replicas',
        'empty_rows',
        'empty_columns',
    ],
)
def test_sinkhorn_shards(configuration, tmp_path):
    assert os.getenv('SLURM_JOB_ID'), 'Run distributed GPU qualification through Slurm'
    if torch.cuda.device_count() < 4:
        pytest.skip('Requires the declared four-GPU allocation')
    mp.spawn(
        _run, args=(f'file://{tmp_path}/rendezvous', configuration), nprocs=4, join=True
    )


@pytest.mark.gpus(1)
def test_sinkhorn_epsilon_first_division_cuda():
    from megatron.lite.primitive.optimizers.sinkhorn import sinkhorn_direction

    assert os.getenv('SLURM_JOB_ID') and torch.cuda.is_available()
    observed = []
    output = sinkhorn_direction(
        torch.tensor([[1e-20]], device='cuda', dtype=torch.float32),
        trace=lambda iteration, value: observed.append(value),
    )
    assert len(observed) == 11
    torch.testing.assert_close(
        observed[0], torch.tensor([[0.5]], device='cuda'), atol=0, rtol=0
    )
    torch.testing.assert_close(output, torch.ones_like(output), atol=1e-6, rtol=0)


def _row_lookup(rank, rendezvous, trainable):
    from megatron.lite.primitive.modules.engram_lookup import (
        RowLookup,
        ShardedEngramTable,
    )
    from megatron.lite.primitive.optimizers.sinkhorn import Sinkhorn

    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    try:
        cuts = (0, 2, 3, 4, 5)
        begin, end = cuts[rank : rank + 2]
        full = (
            (torch.arange(160).reshape(5, 32).float() / 32)
            .to(torch.float8_e4m3fn)
            .cuda()
        )
        scales = torch.ones(5, 1, device='cuda').to(torch.float8_e8m0fnu)
        lookup = RowLookup(cuts, dist.group.WORLD)
        table = ShardedEngramTable(
            full[begin:end],
            scales[begin:end],
            lookup,
            trainable=trainable,
            output_dtype=torch.float32,
        )
        ids = torch.tensor(
            [] if rank == 0 else [rank, (rank + 1) % 5, rank],
            dtype=torch.int64,
            device='cuda',
        )
        values, row_scales, _ = table.lookup_fp8(ids)
        assert torch.equal(values.view(torch.uint8), full.view(torch.uint8)[ids])
        assert torch.equal(row_scales.view(torch.uint8), scales.view(torch.uint8)[ids])
        result = table(ids)
        torch.testing.assert_close(result, full.float()[ids], atol=0, rtol=0)
        if trainable:
            (result.sum() * ((rank + 1) / 1024)).backward()
            gradient = torch.zeros(5, 32)
            for peer in range(1, 4):
                for row in [peer, (peer + 1) % 5, peer]:
                    gradient[row] += (peer + 1) / 1024
            torch.testing.assert_close(
                table.master.grad.cpu(), gradient[begin:end], atol=0, rtol=0
            )
            expected, momentum = scalar_step(
                full.float().cpu().tolist(),
                torch.zeros(5, 32).tolist(),
                gradient.tolist(),
                0.001,
                5,
            )
            opt = Sinkhorn(
                [{'params': [table.master], 'multiplier': 5}],
                lr=0.001,
                row_group=dist.group.WORLD,
            )
            assert opt.step()
            torch.testing.assert_close(
                table.master.cpu().double(), expected[begin:end], atol=2e-6, rtol=2e-6
            )
            torch.testing.assert_close(
                opt.state[table.master]['momentum'].cpu().double(),
                momentum[begin:end],
                atol=2e-6,
                rtol=2e-6,
            )
            table.refresh_storage()
            assert table.weight.is_cuda and table.scale.is_cuda and table.master.is_cuda
        else:
            assert table.master is None and not result.requires_grad
        with pytest.raises(ValueError, match='Row ID outside logical table'):
            lookup.route(torch.tensor([-1 if rank == 0 else 0], device='cuda'))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('trainable', [False, True])
def test_resident_row_lookup_backward_and_update(trainable, tmp_path):
    assert os.getenv('SLURM_JOB_ID'), 'Run GPU qualification through Slurm'
    if torch.cuda.device_count() < 4:
        pytest.skip('Requires the declared four-GPU allocation')
    mp.spawn(
        _row_lookup,
        args=(f'file://{tmp_path}/row-lookup', trainable),
        nprocs=4,
        join=True,
    )

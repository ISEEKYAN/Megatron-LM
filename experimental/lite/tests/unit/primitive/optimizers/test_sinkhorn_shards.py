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


def _run_engram(rank, rendezvous, checkpoint_root, trainable):
    from megatron.lite.model.deepseek_v41.lite import prefetch, table_state
    from megatron.lite.primitive.modules import engram_lookup
    from megatron.lite.primitive.quantization import block_fp8

    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    try:
        row_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        replicas = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        begin, end = span(5, 2, rank % 2)

        def construct():
            table = engram_lookup.ShardedEngramTable(
                torch.full((end - begin, 32), 256.0, device='cuda').to(
                    torch.float8_e4m3fn
                ),
                torch.full((end - begin, 1), 1 / 256, device='cuda').to(
                    torch.float8_e8m0fnu
                ),
                engram_lookup.RowLookup((0, 3, 5), row_groups[rank // 2]),
                trainable=trainable,
                output_dtype=torch.float32,
            )
            return table_state.EngramSinkhornState(
                table, row_group=dist.group.WORLD, replica_group=replicas[rank % 2]
            )

        state = construct()
        expected = torch.ones(5, 32, dtype=torch.float64)
        momentum = torch.zeros_like(expected)
        features = torch.arange(1.0, 33.0, dtype=torch.float64)
        for step in range(3):
            schedules = [
                {
                    'a': [] if worker == 3 else [step, step, 4],
                    'b': [1, 4] if worker % 2 == 0 else [2],
                }
                for worker in range(4)
            ]
            ids = {
                name: torch.tensor(rows, device='cuda', dtype=torch.int64)
                for name, rows in schedules[rank].items()
            }
            batch = prefetch.EngramPrefetch(state).start(
                step, ids, stream=torch.cuda.Stream()
            )
            for name in ('b', 'a'):
                output = batch.view(name)(ids[name])
                if trainable:
                    coefficient = (rank + 1) * (1 if name == 'a' else 2**-10)
                    (output * features.float().cuda() * coefficient).sum().backward()
            if trainable:
                assert state.table.master.grad is None
                assert not torch.count_nonzero(state.main_grad)
            batch.flush()
            global_grad = torch.zeros(5, 32, dtype=torch.float64)
            local_grad = torch.zeros_like(global_grad)
            for worker, schedule in enumerate(schedules):
                for name, rows in schedule.items():
                    coefficient = (worker + 1) * (1 if name == 'a' else 2**-10)
                    for row in rows:
                        global_grad[row] += coefficient * features
                        if worker // 2 == rank // 2:
                            local_grad[row] += coefficient * features
            if trainable:
                torch.testing.assert_close(
                    state.main_grad.cpu().double(),
                    local_grad[begin:end],
                    atol=0,
                    rtol=0,
                )
                expected, momentum = scalar_step(
                    expected.tolist(), momentum.tolist(), global_grad.tolist(), 0.001, 5
                )
                assert state.step(lr=0.001)
                torch.testing.assert_close(
                    state.table.master.cpu().double(),
                    expected[begin:end],
                    atol=3e-6,
                    rtol=3e-6,
                )
                owned_start, owned_end = span(end - begin, 2, rank // 2)
                torch.testing.assert_close(
                    state.momentum.cpu().double(),
                    momentum[begin:end][owned_start:owned_end],
                    atol=2e-5,
                    rtol=3e-6,
                )
                values, scales = block_fp8.quantize_block_fp8(
                    state.table.master.detach(), (1, 32), scale_format='e8m0'
                )
                assert torch.equal(
                    values.view(torch.uint8), state.table.weight.view(torch.uint8)
                )
                assert torch.equal(
                    scales.view(torch.uint8), state.table.scale.view(torch.uint8)
                )
                assert state.momentum.is_cuda and state.main_grad.is_cuda
            else:
                assert not state.step(lr=0.001)
                assert state.table.master is state.main_grad is state.momentum is None
                assert torch.all(state.table.weight.float() == 256)
                assert torch.all(state.table.scale.float() == 1 / 256)
            path = f'{checkpoint_root}/state-{rank}.pt'
            torch.save(state.state_dict(), path)
            state = construct()
            state.load_state_dict(
                torch.load(path, map_location=f'cuda:{rank}', weights_only=True)
            )
            assert state.last_step == step
            assert state.table.weight.is_cuda and state.table.scale.is_cuda
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('trainable', [False, True])
def test_engram_sinkhorn_lookup_prefetch_restart(trainable, tmp_path):
    assert os.getenv('SLURM_JOB_ID'), 'Run distributed GPU qualification through Slurm'
    if torch.cuda.device_count() < 4:
        pytest.skip('Requires the declared four-GPU allocation')
    mp.spawn(
        _run_engram,
        args=(f'file://{tmp_path}/rendezvous', str(tmp_path), trainable),
        nprocs=4,
        join=True,
    )

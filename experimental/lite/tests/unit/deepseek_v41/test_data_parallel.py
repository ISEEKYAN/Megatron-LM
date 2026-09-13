# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DP must preserve the global token objective and complete optimizer updates."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig
from megatron.lite.primitive.train_step import run_microbatch_loop
from megatron.lite.runtime.contracts import PackedBatch


def _dp_worker(rank, config, trainable, directory):
    serial, impl = _init_parallel_worker(rank, config, trainable, directory)
    try:
        parallel = protocol.build_model(config, impl_cfg=impl)
        parallel.chunks[0].load_state_dict(serial.chunks[0].state_dict())
        assert parallel.parallel_state.dp_size == 2
        errors = {'gradient_max_abs': 0.0, 'parameter_max_abs': 0.0}
        for step in range(2):
            _train_parallel_step(serial, parallel, rank, step)
            for (name, p), (_, q) in zip(
                parallel.chunks[0].named_parameters(),
                serial.chunks[0].named_parameters(),
                strict=True,
            ):
                assert (p.grad is None) == (q.grad is None), name
                if p.grad is not None:
                    errors['gradient_max_abs'] = max(
                        errors['gradient_max_abs'], float((p.grad - q.grad).abs().max())
                    )
                    # This fixture has an exact serial baseline, including unequal token counts.
                    torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
            assert serial.optimizer.step()[0]
            assert parallel.optimizer.step()[0]
            for (name, p), (_, q) in zip(
                parallel.chunks[0].named_parameters(),
                serial.chunks[0].named_parameters(),
                strict=True,
            ):
                torch.testing.assert_close(p, q, atol=0, rtol=0, msg=name)
                errors['parameter_max_abs'] = max(
                    errors['parameter_max_abs'], float((p - q).detach().abs().max())
                )
                replicas = [torch.empty_like(p) for _ in range(2)]
                dist.all_gather(replicas, p)
                torch.testing.assert_close(*replicas, atol=0, rtol=0, msg=name)
            _check_parallel_buffers(parallel, serial, rank, trainable)
        Path(directory, f'rank-{rank}.json').write_text(json.dumps(errors))
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
@pytest.mark.parametrize('trainable', [False, True])
def test_dp_matches_global_batch(model_config, trainable, tmp_path):
    assert torch.cuda.device_count() >= 2, 'DP comparison requires two allocated GPUs'
    mp.spawn(
        _dp_worker, args=(model_config, trainable, str(tmp_path)), nprocs=2, join=True
    )
    for rank in range(2):
        print(
            f'DP parity trainable_engram={trainable} rank={rank}: '
            f'{(tmp_path / f"rank-{rank}.json").read_text()}'
        )


def _init_parallel_worker(rank, config, trainable, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(19)
    impl = protocol.ImplConfig(
        device=f'cuda:{rank}',
        dtype=torch.float32,
        quantized=False,
        token_map=list(range(256)),
        trainable_engram=trainable,
        optimizer='muon',
        optimizer_config=OptimizerConfig(0.0001, 5, 'quintic'),
    )
    # Construct the serial baseline before joining the distributed world.
    serial = protocol.build_model(config, impl_cfg=impl)
    dist.init_process_group(
        'nccl',
        init_method=(Path(directory) / 'rendezvous').as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    return serial, impl


def _train_parallel_step(serial, parallel, rank, step):
    batches = []
    for other_rank in range(2):
        ids = torch.arange(1 + step, 5 + step + other_rank * 2, device=rank)
        batches.append(PackedBatch(ids, ids, torch.tensor([len(ids)], device=rank)))
    if step == 0:
        with torch.no_grad():
            local_logits = parallel.forward_step(parallel.chunks[0], batches[rank])[
                'logits'
            ]
            serial_logits = serial.forward_step(serial.chunks[0], batches[rank])[
                'logits'
            ]
        torch.testing.assert_close(local_logits, serial_logits, atol=0, rtol=0)
    for bundle, inputs in ((serial, batches), (parallel, [batches[rank]])):
        bundle.optimizer.zero_grad()
        run_microbatch_loop(
            bundle.chunks[0],
            iter(inputs),
            len(inputs),
            bundle.forward_step,
            prepare_microbatches=bundle.extras['prepare_microbatches'],
        )
        if bundle.finalize_grads is not None:
            bundle.finalize_grads()


def _check_parallel_buffers(parallel, serial, rank, trainable):
    for block, reference in zip(parallel.chunks[0].layers, serial.chunks[0].layers):
        for name in ('bias', 'bias_vl'):
            torch.testing.assert_close(
                getattr(block.ffn.gate, name),
                getattr(reference.ffn.gate, name),
                atol=0,
                rtol=0,
            )
        if block.engram is not None:
            table = block.engram.embed
            assert table.weight.device == torch.device('cuda', rank)
            assert table.scale.device == table.weight.device
            if trainable:
                assert table.master.device == table.weight.device

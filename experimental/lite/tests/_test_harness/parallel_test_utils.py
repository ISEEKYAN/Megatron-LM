# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Distributed test lifecycle, exact comparisons, and model fixtures."""

import json
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch

assert_exact = partial(torch.testing.assert_close, atol=0, rtol=0)


def seed_engram(model):
    with torch.no_grad():
        for block in model.layers:
            if block is not None and block.engram is not None:
                table = block.engram.embed
                rows = torch.linspace(
                    -0.25, 0.25, table.weight.numel(), device=table.weight.device
                ).reshape(table.weight.shape)
                table.weight.copy_(rows.to(table.weight.dtype))
                if table.master is not None:
                    table.master.copy_(table.weight.float())


def init_world(rank, directory, *, world, timeout, rendezvous):
    torch.distributed.init_process_group(
        'nccl',
        init_method=(Path(directory) / rendezvous).as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=timeout),
    )


@contextmanager
def world(rank, directory, *, world=2, timeout=120, rendezvous='rendezvous'):
    init_world(rank, directory, world=world, timeout=timeout, rendezvous=rendezvous)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


def prepare_worker(rank, seed=19):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(seed)


def run_workers(worker, args, directory, *, world=2, report=False, tail=()):
    torch.multiprocessing.spawn(
        worker, args=(*args, str(directory), *tail), nprocs=world, join=True
    )
    if report:
        for rank in range(world):
            print(
                f'{worker.__name__} args={args[1:]} rank={rank}: '
                f'{(Path(directory) / f"rank-{rank}.json").read_text()}'
            )


def report_rank(directory, rank, values):
    Path(directory, f'rank-{rank}.json').write_text(json.dumps(values))


def parameter_pairs(model, *references, strict=False):
    parameters = [dict(reference.named_parameters()) for reference in references]
    owned = dict(model.named_parameters())
    if strict:
        for reference in parameters:
            if owned.keys() != reference.keys():
                raise ValueError("parameter ownership differs from the full reference")
    for name, parameter in owned.items():
        yield name, parameter, *(reference[name] for reference in parameters)


def load_local_state(model, reference):
    state = reference.state_dict()
    model.load_state_dict({name: state[name] for name in model.state_dict()})


def backward(bundle, batches, forward_step=None, *, finalize=False):
    from megatron.lite.primitive.train_step import run_microbatch_loop

    result = run_microbatch_loop(
        bundle.chunks[0],
        iter(batches),
        len(batches),
        bundle.forward_step if forward_step is None else forward_step,
        prepare_microbatches=bundle.extras['prepare_microbatches'],
    )
    if finalize and bundle.finalize_grads is not None:
        bundle.finalize_grads()
    return result


@contextmanager
def forward_hooks(model, eligible, callback):
    handles = [
        module.register_forward_hook(partial(callback, name))
        for name, module in model.named_modules()
        if eligible(name, module)
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _init_parallel_worker(rank, config, trainable):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig

    prepare_worker(rank)
    impl = protocol.ImplConfig(
        device=f'cuda:{rank}',
        dtype=torch.float32,
        quantized=False,
        token_map=list(range(256)),
        trainable_engram=trainable,
        shard_engram=False,
        optimizer='muon',
        optimizer_config=OptimizerConfig(0.0001, 5, 'quintic'),
    )
    # Construct the serial baseline before joining the distributed world.
    serial = protocol.build_model(config, impl_cfg=impl)
    return serial, impl


def _train_parallel_step(serial, parallel, rank, step):
    from megatron.lite.runtime.contracts import PackedBatch

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
        assert_exact(local_logits, serial_logits)
    for bundle, inputs in ((serial, batches), (parallel, [batches[rank]])):
        bundle.optimizer.zero_grad()
        backward(bundle, inputs, finalize=True)


def _check_parallel_buffers(parallel, serial, rank, trainable):
    for block, reference in zip(parallel.chunks[0].layers, serial.chunks[0].layers):
        for name in ('bias', 'bias_vl'):
            assert_exact(
                getattr(block.ffn.gate, name), getattr(reference.ffn.gate, name)
            )
        if block.engram is not None:
            table = block.engram.embed
            assert table.weight.device == torch.device('cuda', rank)
            assert table.scale.device == table.weight.device
            if trainable:
                assert table.master.device == table.weight.device


def record_error(errors, key, actual, expected):
    error = float((actual - expected).detach().abs().max())
    errors[key] = max(errors[key], error)

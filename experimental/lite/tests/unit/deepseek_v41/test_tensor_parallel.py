# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TP ownership, full logical optimizer scope, and serial training oracles."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_tp_layout_rejects_double_slice_and_nondivisible():
    from megatron.lite.primitive.parallel.matrix import TensorShard

    layout = TensorShard((8, 4), 0, 1, 2)
    full = torch.arange(32).reshape(8, 4)
    torch.testing.assert_close(layout.slice(full), full[4:], atol=0, rtol=0)
    with pytest.raises(ValueError, match='full logical shape'):
        layout.slice(full[4:])
    with pytest.raises(ValueError, match='divisible'):
        TensorShard((7, 4), 0, 0, 2)


def test_tp_requires_initialized_world(moe, model_config):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(ValueError, match='TP requires an initialized'):
        protocol.build_model(
            model_config,
            impl_cfg=protocol.ImplConfig(
                device='meta', quantized=False, parallel=ParallelConfig(tp=2)
            ),
        )


def _worker(rank, config, directory, case, trainable=False):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig
    from megatron.lite.primitive.parallel.matrix import slice_parameter
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.manual_seed(41)
    impl = protocol.ImplConfig(
        device=f'cuda:{rank}',
        dtype=torch.float32,
        quantized=False,
        token_map=list(range(256)),
        trainable_engram=trainable,
        optimizer='muon',
        optimizer_config=OptimizerConfig(0.001, 5, 'quintic'),
    )
    # Independent full-matrix baseline exists before any process group.
    serial = protocol.build_model(config, impl_cfg=impl)
    restored = (
        protocol.build_model(config, impl_cfg=impl) if case == 'training' else None
    )
    dist.init_process_group(
        'nccl',
        init_method=(Path(directory) / 'rendezvous').as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        parallel = protocol.build_model(
            config, impl_cfg=replace(impl, parallel=ParallelConfig(tp=2))
        )
        model, reference = parallel.chunks[0], serial.chunks[0]
        full_state = reference.state_dict()
        parameters = dict(model.named_parameters())
        model.load_state_dict(
            {
                name: (
                    slice_parameter(parameters[name], value)
                    if name in parameters
                    else value
                )
                for name, value in full_state.items()
            }
        )
        local_count = sum(p.numel() for p in model.parameters())
        serial_count = sum(p.numel() for p in reference.parameters())
        assert local_count < serial_count, 'TP_LOCAL_PARAMETER_STORAGE_MUST_SHRINK'
        assert (
            model.ps.tp_size == 2 and model.ps.dp_size == 1 and model.ps.etp_size == 1
        )
        print(
            f'TP_STORAGE rank={rank} local={local_count} serial={serial_count}',
            flush=True,
        )
        if case == 'scope':
            _optimizer_scope(serial, parallel)
        else:
            _training(serial, parallel, restored, directory, trainable)
    finally:
        dist.destroy_process_group()


def _optimizer_scope(serial, parallel):
    from megatron.lite.primitive.parallel.matrix import slice_parameter

    model, reference = parallel.chunks[0], serial.chunks[0]
    names = ['layers.0.attn.wq_a.weight', 'layers.0.attn.wq_b.weight', 'head.weight']
    full, local = dict(reference.named_parameters()), dict(model.named_parameters())
    routes = {g['owner_key']: g for g in parallel.optimizer.param_groups}
    assert routes[names[0]]['matrix_shape'] == (32, 32), 'TP_SHARED_MATRIX_SCOPE'
    assert routes[names[1]]['matrix_shape'] == (2, 32, 32), 'TP_HEADWISE_MATRIX_SCOPE'
    for step in range(2):
        serial.optimizer.zero_grad()
        parallel.optimizer.zero_grad()
        for name in names:
            # The oracle's full gradient is created independently of TP values.
            p = full[name]
            values = torch.arange(
                p.numel(), device=p.device, dtype=torch.float32
            ).reshape_as(p)
            gradient = (values * (0.017 + step * 0.003)).sin() + (values * 0.031).cos()
            p.grad = p.main_grad = gradient
            q = local[name]
            q.grad = q.main_grad = slice_parameter(q, gradient).clone()
        assert serial.optimizer.step()[0]
        assert parallel.optimizer.step()[0]
        for name in names:
            torch.testing.assert_close(
                local[name],
                slice_parameter(local[name], full[name]),
                atol=0,
                rtol=0,
                msg='TP_OPTIMIZER_LOGICAL_SCOPE:' + name,
            )
    print('TP_OPTIMIZER_LOGICAL_SCOPE two_steps max_abs=0', flush=True)


@pytest.mark.gpus(2)
def test_tp_optimizer_uses_full_logical_matrices(model_config, tmp_path):
    assert torch.cuda.device_count() >= 2, 'TP optimizer requires two allocated GPUs'
    mp.spawn(_worker, args=(model_config, str(tmp_path), 'scope'), nprocs=2, join=True)


@pytest.mark.gpus(2)
@pytest.mark.parametrize('trainable', [False, True])
def test_tp_training_and_checkpoint_roundtrip(model_config, tmp_path, trainable):
    assert torch.cuda.device_count() >= 2, 'TP training requires two allocated GPUs'
    mp.spawn(
        _worker,
        args=(model_config, str(tmp_path), 'training', trainable),
        nprocs=2,
        join=True,
    )


def _training(serial, parallel, restored, directory, trainable):
    from megatron.lite.primitive.parallel.matrix import slice_parameter
    from megatron.lite.primitive.train_step import run_microbatch_loop
    from megatron.lite.runtime.contracts import PackedBatch

    model, reference = parallel.chunks[0], serial.chunks[0]
    reference_parameters = dict(reference.named_parameters())
    snapshots = {}
    handles = []

    def capture(name, module, args, output):
        snapshots.setdefault(name, []).append((args[0].detach(), output.detach()))

    def diagnose(name, module, args, output):
        if name not in snapshots:
            return
        x, y = snapshots[name].pop(0)
        delta = float((output - y).abs().max())
        if delta:
            q = reference.get_submodule(name)
            full64 = torch.nn.functional.linear(x.double(), q.weight.double()).float()
            print(
                'TP_FORWARD_DIAG '
                + json.dumps(
                    dict(
                        name=name,
                        input_max_abs=float((args[0] - x).abs().max()),
                        output_max_abs=delta,
                        serial_fp64_max_abs=float((y - full64).abs().max()),
                        tp_fp64_max_abs=float((output - full64).abs().max()),
                    )
                ),
                flush=True,
            )

    from functools import partial

    for name, module in model.named_modules():
        if hasattr(getattr(module, 'weight', None), 'tp_shard'):
            handles.append(
                reference.get_submodule(name).register_forward_hook(
                    partial(capture, name)
                )
            )
            handles.append(module.register_forward_hook(partial(diagnose, name)))
    collectives = {'all_reduce': 0, 'all_gather': 0}
    originals = {name: getattr(dist, name) for name in collectives}

    def count(name):
        def call(*args, **kwargs):
            collectives[name] += 1
            return originals[name](*args, **kwargs)

        return call

    for name in collectives:
        setattr(dist, name, count(name))
    for step in range(2):
        ids = torch.arange(1 + step, 9 + step, device=next(model.parameters()).device)
        batch = PackedBatch(ids, ids, torch.tensor([4, 4], device=ids.device))
        with torch.no_grad():
            expected = serial.forward_step(reference, batch)['logits']
            actual = parallel.forward_step(model, batch)['logits']
        torch.testing.assert_close(
            actual, expected, atol=0, rtol=0, msg='TP_LOGITS_PARITY'
        )
        for bundle in (serial, parallel):
            bundle.optimizer.zero_grad()
            run_microbatch_loop(
                bundle.chunks[0],
                iter([batch]),
                1,
                bundle.forward_step,
                prepare_microbatches=bundle.extras['prepare_microbatches'],
            )
        for name, p in model.named_parameters():
            q = reference_parameters[name]
            assert (p.grad is None) == (q.grad is None), 'TP_GRADIENT_PRESENCE:' + name
            if p.grad is not None:
                torch.testing.assert_close(
                    p.grad,
                    slice_parameter(p, q.grad),
                    atol=0,
                    rtol=0,
                    msg='TP_GRADIENT_PARITY:' + name,
                )
        assert serial.optimizer.step()[0]
        assert parallel.optimizer.step()[0]
        for name, p in model.named_parameters():
            torch.testing.assert_close(
                p,
                slice_parameter(p, reference_parameters[name]),
                atol=0,
                rtol=0,
                msg='TP_STEP_PARAMETER_PARITY:' + name,
            )
    print(
        f'TP_TRAINING trainable_engram={trainable} two_steps logits=0 gradient=0 parameter=0',
        flush=True,
    )
    for name, original in originals.items():
        setattr(dist, name, original)
    assert all(collectives.values()), 'TP_REAL_COLLECTIVES_REQUIRED'
    print('TP_COLLECTIVES ' + json.dumps(collectives), flush=True)
    _roundtrip(serial, parallel, restored, directory)


def _roundtrip(serial, parallel, restored, directory):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import (
        CheckpointTensorStore,
        load_model,
        save_model,
    )
    from megatron.lite.primitive.parallel.matrix import slice_parameter
    from safetensors.torch import save_file

    model, reference, loaded = parallel.chunks[0], serial.chunks[0], restored.chunks[0]
    archive = Path(directory) / 'archive.safetensors'
    if dist.get_rank() == 0:
        # Inactive archival bytes are inert but complete, including DSpark keys.
        save_file(
            {name: torch.zeros(1) for name in reference.archival_bindings}, archive
        )
    dist.barrier()
    model.archival_store = CheckpointTensorStore.load(
        [archive], expected_keys=reference.archival_bindings
    )
    path = Path(directory) / 'tp-checkpoint'
    save_model(model, path)
    dist.barrier()
    load_model(loaded, path)
    for name, value in reference.state_dict().items():
        actual = loaded.state_dict()[name]
        assert torch.equal(
            value.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
        ), ('TP_SAVE_TO_SERIAL_ROUNDTRIP:' + name)
    # Loading the same full checkpoint into TP applies precisely one slice.
    load_model(model, path)
    expected = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter,
            slice_parameter(parameter, expected[name]),
            atol=0,
            rtol=0,
            msg='TP_CHECKPOINT_SINGLE_SLICE:' + name,
        )
    restored.optimizer.load_state_dict(parallel.optimizer.state_dict())
    print('TP_SAVE_TO_SERIAL_ROUNDTRIP parameters_and_buffers_bitwise=true', flush=True)

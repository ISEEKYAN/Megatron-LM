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
    from megatron.lite.primitive.parallel.matrix import TensorShard, gather_parameter

    layout = TensorShard((8, 4), 0, 1, 2)
    full = torch.arange(32).reshape(8, 4)
    torch.testing.assert_close(layout.slice(full), full[4:], atol=0, rtol=0)
    with pytest.raises(ValueError, match='full logical shape'):
        layout.slice(full[4:])
    with pytest.raises(ValueError, match='divisible'):
        TensorShard((7, 4), 0, 0, 2)
    shard = torch.nn.Parameter(torch.zeros(4, 4))
    shard.tp_shard = layout
    with pytest.raises(ValueError, match='explicit TP group'):
        gather_parameter(shard, None)


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


def test_fp64_diagnostic_preserves_accumulation_before_fp32_publication():
    from megatron.lite.primitive.modules.native_fp32_linear import native_fp32_linear

    # Casting operands to FP32 before the reduction loses the middle term.
    x = torch.tensor([[1e8], [1.0], [-1e8]], dtype=torch.float64)
    weight = torch.ones(1, 1, dtype=torch.float32, requires_grad=True)
    native_fp32_linear(x, weight).sum().backward()
    assert weight.grad.dtype == torch.float32
    assert weight.grad.item() == 1.0, 'FP64_ACCUMULATION_BEFORE_FP32_PUBLICATION'


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
    # This sensitivity probe is separate from the untouched parity reference.
    probe = protocol.build_model(config, impl_cfg=impl) if case == 'native' else None
    if probe is not None:
        probe.chunks[0].load_state_dict(serial.chunks[0].state_dict())
    if case == 'training':
        _full_precision_reference(serial.chunks[0])
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
        assert len(model.layers) == 40, 'TP_FULL_LAYER_CHAIN'
        assert {block.attn.ratio for block in model.layers} == {
            0,
            1,
            2,
        }, 'TP_CSA2_THREE_MODES'
        assert all(
            not p.requires_grad
            for block in model.layers
            if block.attn.indexer is not None
            for p in block.attn.indexer.parameters()
        ), 'TP_INDEXER_MUST_REMAIN_FROZEN'
        print(
            f'TP_STORAGE rank={rank} local={local_count} serial={serial_count}',
            flush=True,
        )
        if case == 'scope':
            _optimizer_scope(serial, parallel)
        else:
            if case == 'training':
                _full_precision_parallel(model)
            _training(
                serial,
                parallel,
                restored,
                directory,
                trainable,
                exact=case == 'training',
                probe=probe,
            )
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
        differences = {
            name: float(
                (local[name] - slice_parameter(local[name], full[name])).abs().max()
            )
            for name in names
            if not torch.equal(local[name], slice_parameter(local[name], full[name]))
        }
        assert not differences, 'TP_OPTIMIZER_LOGICAL_SCOPE:' + json.dumps(differences)
    print('TP_OPTIMIZER_LOGICAL_SCOPE two_steps max_abs=0', flush=True)


@pytest.mark.gpus(2)
def test_tp_optimizer_uses_full_logical_matrices(model_config, tmp_path):
    assert torch.cuda.device_count() >= 2, 'TP optimizer requires two allocated GPUs'
    mp.spawn(_worker, args=(model_config, str(tmp_path), 'scope'), nprocs=2, join=True)


@pytest.mark.gpus(2)
@pytest.mark.parametrize('trainable', [False, True])
def test_tp_training_and_checkpoint_roundtrip(model_config, tmp_path, trainable):
    assert torch.cuda.device_count() >= 2, 'TP training requires two allocated GPUs'
    # Native FP32 trajectories are reported without a relaxed tolerance. The
    # independent full-matrix FP64 oracle must then pass exact FP32 publication.
    for case in ('native', 'training'):
        directory = tmp_path / case
        directory.mkdir()
        mp.spawn(
            _worker,
            args=(model_config, str(directory), case, trainable),
            nprocs=2,
            join=True,
        )


def _training(serial, parallel, restored, directory, trainable, *, exact, probe=None):
    from megatron.lite.primitive.parallel.matrix import slice_parameter
    from megatron.lite.primitive.train_step import run_microbatch_loop
    from megatron.lite.runtime.contracts import PackedBatch

    model, reference = parallel.chunks[0], serial.chunks[0]
    reference_parameters = dict(reference.named_parameters())
    collectives = {'all_reduce': 0, 'all_gather': 0}
    originals = {name: getattr(dist, name) for name in collectives}

    def count(name):
        def call(*args, **kwargs):
            collectives[name] += 1
            return originals[name](*args, **kwargs)

        return call

    for name in collectives:
        setattr(dist, name, count(name))
    errors, amplification = [], []
    for step in range(2):
        error = dict(logits=0.0, gradient=0.0, parameter=0.0)
        ids = torch.arange(1 + step, 9 + step, device=next(model.parameters()).device)
        batch = PackedBatch(ids, ids, torch.tensor([4, 4], device=ids.device))
        with torch.no_grad():
            expected = serial.forward_step(reference, batch)['logits']
            actual = parallel.forward_step(model, batch)['logits']
            if probe is not None and step:
                predicted = probe.forward_step(probe.chunks[0], batch)['logits']
                amplification[-1].update(
                    predicted_next_logits_delta=float(
                        (predicted - expected).abs().max()
                    ),
                    measured_next_logits_delta=float((actual - expected).abs().max()),
                    prediction_logits_residual=float((predicted - actual).abs().max()),
                )
                torch.testing.assert_close(
                    predicted,
                    actual,
                    atol=0,
                    rtol=0,
                    msg='TP_UPDATE_AMPLIFICATION_LOGITS',
                )
        assert actual.shape == expected.shape, 'TP_HEAD_MUST_GATHER_FULL_VOCAB'
        error['logits'] = float((actual - expected).abs().max())
        if exact:
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
        has_fp32_low_bits = False
        for name, p in model.named_parameters():
            q = reference_parameters[name]
            assert (p.grad is None) == (q.grad is None), 'TP_GRADIENT_PRESENCE:' + name
            if p.grad is not None:
                has_fp32_low_bits |= not torch.equal(p.grad, p.grad.bfloat16().float())
                assert p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all(), (
                    'TP_NATIVE_FP32_MAIN_GRAD:' + name
                )
                error['gradient'] = max(
                    error['gradient'],
                    float((p.grad - slice_parameter(p, q.grad)).abs().max()),
                )
                if exact:
                    torch.testing.assert_close(
                        p.grad,
                        slice_parameter(p, q.grad),
                        atol=0,
                        rtol=0,
                        msg=lambda detail: 'TP_GRADIENT_PARITY:' + name + '\n' + detail,
                    )
        assert has_fp32_low_bits, 'TP_MAIN_GRAD_MUST_NOT_BE_UPCAST_BF16'
        if probe is not None:
            amplification.append(
                _replay_gradient_perturbation(serial, parallel, probe, batch)
            )
        assert serial.optimizer.step()[0]
        assert parallel.optimizer.step()[0]
        if probe is not None:
            assert probe.optimizer.step()[0]
            predicted_parameters = dict(probe.chunks[0].named_parameters())
            amplification[-1]['predicted_parameter_delta'] = max(
                float((p - reference_parameters[name]).detach().abs().max())
                for name, p in predicted_parameters.items()
            )
            residual = 0.0
            for name, p in model.named_parameters():
                predicted = slice_parameter(p, predicted_parameters[name])
                residual = max(residual, float((p - predicted).detach().abs().max()))
                torch.testing.assert_close(
                    p,
                    predicted,
                    atol=0,
                    rtol=0,
                    msg='TP_UPDATE_AMPLIFICATION_PARAMETERS:' + name,
                )
            amplification[-1]['prediction_parameter_residual'] = residual
        for name, p in model.named_parameters():
            error['parameter'] = max(
                error['parameter'],
                float(
                    (p - slice_parameter(p, reference_parameters[name]))
                    .detach()
                    .abs()
                    .max()
                ),
            )
            if exact:
                torch.testing.assert_close(
                    p,
                    slice_parameter(p, reference_parameters[name]),
                    atol=0,
                    rtol=0,
                    msg='TP_STEP_PARAMETER_PARITY:' + name,
                )
        errors.append(error)
        if probe is not None:
            measured = torch.tensor(
                error['parameter'], device=ids.device, dtype=torch.float64
            )
            dist.all_reduce(measured, op=dist.ReduceOp.MAX)
            amplification[-1]['measured_parameter_delta'] = float(measured)
    print(
        f'TP_TRAINING trainable_engram={trainable} fp64_rounded_to_fp32={exact} errors={json.dumps(errors)}',
        flush=True,
    )
    for name, original in originals.items():
        setattr(dist, name, original)
    assert all(collectives.values()), 'TP_REAL_COLLECTIVES_REQUIRED'
    print('TP_COLLECTIVES ' + json.dumps(collectives), flush=True)
    if probe is not None:
        print('TP_UPDATE_AMPLIFICATION ' + json.dumps(amplification), flush=True)
    if exact:
        _roundtrip(serial, parallel, restored, directory)


@torch.no_grad()
def _replay_gradient_perturbation(serial, parallel, probe, batch):
    """Measured-perturbation experiment, NOT the independent parity oracle.

    Start from identical full matrices/state. Apply the measured gradient
    perturbation to a separate serial optimizer; never copy TP updated weights.
    Exact prediction residuals detect additional errors in clipping, momentum,
    parameter publication or the next forward.
    """
    from megatron.lite.primitive.parallel.matrix import gather_parameter

    probe.optimizer.zero_grad()
    reference = dict(serial.chunks[0].named_parameters())
    predicted = dict(probe.chunks[0].named_parameters())
    gradient_delta = 0.0
    for name, p in parallel.chunks[0].named_parameters():
        if p.grad is None:
            continue
        measured = gather_parameter(p, parallel.parallel_state.tp_group, p.grad)
        baseline = reference[name].grad
        delta = measured.double() - baseline.double()
        gradient_delta = max(gradient_delta, float(delta.abs().max()))
        q = predicted[name]
        q.grad = q.main_grad = (baseline.double() + delta).float()
    # Routing statistics come from this full forward, never TP buffers.
    output = probe.chunks[0](batch.input_ids[None], cu_seqlens=batch.cu_seqlens)
    probe.optimizer.accumulate_modality_loads(output['modality_loads'])
    return dict(gradient_input_delta=gradient_delta)


def _projection_names(model):
    # Independent object inventory: no TP layout or parallel values are read.
    owners = [model.head]
    for block in model.layers:
        owners.extend(
            getattr(block.attn, name) for name in ('wq_a', 'wq_b', 'wkv', 'wo_b')
        )
        compressor = block.attn.compressor
        if compressor is not None:
            owners.append(compressor.wkv)
            if hasattr(compressor, 'wgate'):
                owners.append(compressor.wgate)
    ids = {id(module) for module in owners}
    return [name for name, module in model.named_modules() if id(module) in ids]


class _FullMatrix64(torch.autograd.Function):
    """Independent serial oracle: every GEMM uses the original complete matrix."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x.double(), weight.double())
        return (x.double() @ weight.double().T).float()

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy = dy.double()
        dx = (dy @ weight).float()
        dw = (
            dy.reshape(-1, weight.shape[0]).T @ x.reshape(-1, weight.shape[1])
        ).float()
        return dx, dw


def _full_precision_reference(model):
    from functools import partial

    def forward(weight, x):
        return _FullMatrix64.apply(x, weight)

    assert not dist.is_initialized(), 'TP_SERIAL_ORACLE_MUST_PRECEDE_DISTRIBUTED_INIT'
    for name in _projection_names(model):
        module = model.get_submodule(name)
        module.forward = partial(forward, module.weight)


def _full_precision_parallel(model):
    # Actual native linear/TP collectives execute with double activation
    # accumulators. Local dgrad remains double until the real TP all-reduce,
    # then casts back through the input hook; masters/main_grad remain FP32.
    for name in _projection_names(model):
        module = model.get_submodule(name)
        module.register_forward_pre_hook(lambda module, args: (args[0].double(),))
        module.register_forward_hook(lambda module, args, output: output.float())


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
    try:
        load_model(loaded, path)
    except (ValueError, RuntimeError) as error:
        raise AssertionError('TP_SAVE_TO_SERIAL_ROUNDTRIP') from error
    for name, value in reference.state_dict().items():
        actual = loaded.state_dict()[name]
        assert torch.equal(
            value.contiguous().reshape(-1).view(torch.uint8),
            actual.contiguous().reshape(-1).view(torch.uint8),
        ), ('TP_SAVE_TO_SERIAL_ROUNDTRIP:' + name)
    # Loading the same full checkpoint into TP applies precisely one slice.
    try:
        load_model(model, path)
    except (ValueError, RuntimeError) as error:
        raise AssertionError('TP_CHECKPOINT_SINGLE_SLICE') from error
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

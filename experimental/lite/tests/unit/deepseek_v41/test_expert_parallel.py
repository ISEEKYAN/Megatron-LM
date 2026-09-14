# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""EP must preserve the global token objective and complete optimizer updates."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig
from test_data_parallel import (
    _check_parallel_buffers,
    _init_parallel_worker,
    _train_parallel_step,
)


def _ep_worker(rank, config, trainable, directory, optimizer_failure=None):
    serial, impl = _init_parallel_worker(rank, config, trainable, directory)
    try:
        parallel = protocol.build_model(
            config, impl_cfg=replace(impl, parallel=ParallelConfig(ep=2))
        )
        assert parallel.parallel_state.ep_size == 2, 'EP topology must be active'
        ps = parallel.parallel_state
        execution = parallel.forward_step.keywords['execution_model']
        assert execution.process_group is ps.dp_group, 'EP_DENSE_USES_DP_GROUP'
        assert ps.dp_size == 2 and ps.expert_dp_size == 1, 'EP_WORLD_DECOMPOSITION'
        assert 'embed.weight' not in execution.parameters_to_ignore, 'EP_EMBED_IS_DENSE'

        model = parallel.chunks[0]
        reference = serial.chunks[0]
        local_state = model.state_dict()
        model.load_state_dict(
            {k: v for k, v in reference.state_dict().items() if k in local_state}
        )
        for bundle in (serial, parallel):
            for layer in bundle.chunks[0].layers:
                # Both sources route to experts 0/1: rank 1 receives zero tokens.
                layer.ffn.gate.bias[:2] = 100
        for layer in model.layers:
            owned = [i for i, e in enumerate(layer.ffn.experts) if e is not None]
            assert owned == list(
                range(rank * 2, rank * 2 + 2)
            ), 'global expert ownership'
        if optimizer_failure is not None:
            _check_optimizer_contract(parallel, rank, optimizer_failure)
            return
        records = {}
        _capture_expert_linear_inputs(reference, records)
        errors = {'gradient_max_abs': 0.0, 'parameter_max_abs': 0.0}
        for step in range(2):
            records.clear()
            # Step 0 leaves one receiving rank empty; step 1 exercises both
            # owners while leaving one expert empty on each rank.
            for bundle in (serial, parallel):
                for layer in bundle.chunks[0].layers:
                    layer.ffn.gate.bias.fill_(-100)
                    layer.ffn.gate.bias[[0, 1 if step == 0 else 2]] = 100
            _train_parallel_step(serial, parallel, rank, step)
            _finalize_reference_expert_wgrad(records, reference, rank)
            diagnostics = {}
            serial_parameters = dict(reference.named_parameters())
            for name, parameter in model.named_parameters():
                reference_gradient = serial_parameters[name].grad
                if (parameter.grad is None) != (reference_gradient is None):
                    diagnostics[name] = {'gradient_presence_mismatch': True}
                if parameter.grad is not None and reference_gradient is not None:
                    delta = float((parameter.grad - reference_gradient).abs().max())
                    if delta:
                        diagnostics[name] = {
                            'max_abs': delta,
                            'reference_max': float(reference_gradient.abs().max()),
                        }
            Path(directory, f'gradient-rank-{rank}.json').write_text(
                json.dumps(diagnostics)
            )
            print(
                f'EP gradient differences rank={rank}: {json.dumps(diagnostics)}',
                flush=True,
            )
            failures = [None, None]
            dist.all_gather_object(failures, next(iter(diagnostics), None))
            assert not any(failures), f'EP_GRADIENT_PARITY: {failures}'
            for name, p in model.named_parameters():
                q = serial_parameters[name]
                assert (p.grad is None) == (q.grad is None), name
                if p.grad is not None:
                    errors['gradient_max_abs'] = max(
                        errors['gradient_max_abs'], float((p.grad - q.grad).abs().max())
                    )
                    # This fixture has an exact serial baseline, including unequal token counts.
                    torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
            assert serial.optimizer.step()[0]
            assert parallel.optimizer.step()[0]
            changed = next(
                (
                    name
                    for name, parameter in model.named_parameters()
                    if not torch.equal(parameter, serial_parameters[name])
                ),
                None,
            )
            failures = [None, None]
            dist.all_gather_object(failures, changed)
            assert not any(failures), f'EP_PARAMETER_PARITY: {failures}'
            for name, p in model.named_parameters():
                q = serial_parameters[name]
                torch.testing.assert_close(p, q, atol=0, rtol=0, msg=name)
                errors['parameter_max_abs'] = max(
                    errors['parameter_max_abs'], float((p - q).detach().abs().max())
                )
                if '.ffn.experts.' not in name:
                    replicas = [torch.empty_like(p) for _ in range(2)]
                    dist.all_gather(replicas, p)
                    torch.testing.assert_close(*replicas, atol=0, rtol=0, msg=name)
            _check_parallel_buffers(parallel, serial, rank, trainable)
        Path(directory, f'rank-{rank}.json').write_text(json.dumps(errors))
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
@pytest.mark.parametrize('trainable', [False, True])
def test_ep_matches_global_batch(model_config, trainable, tmp_path):
    assert torch.cuda.device_count() >= 2, 'EP comparison requires two allocated GPUs'
    mp.spawn(
        _ep_worker, args=(model_config, trainable, str(tmp_path)), nprocs=2, join=True
    )
    for rank in range(2):
        print(
            f'EP parity trainable_engram={trainable} rank={rank}: '
            f'{(tmp_path / f"rank-{rank}.json").read_text()}'
        )


def test_ep_requires_distributed_world(moe, model_config):
    with pytest.raises((ValueError, RuntimeError), match='EP.*initialized|EP.*world'):
        protocol.build_model(
            model_config,
            impl_cfg=protocol.ImplConfig(
                device='meta', quantized=False, parallel=ParallelConfig(ep=2)
            ),
        )


@pytest.mark.parametrize('dimension', ['tp', 'pp', 'cp', 'vpp', 'etp'])
def test_ep_does_not_unlock_other_dimensions(moe, model_config, dimension):
    message = {
        'cp': 'CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED',
        'pp': 'V4.1_PP_COMBINATION_UNSUPPORTED',
    }.get(dimension, f'V4.1_UNSUPPORTED_PARALLELISM: {dimension};')
    with pytest.raises(NotImplementedError, match=message):
        protocol.build_model(
            model_config,
            impl_cfg=protocol.ImplConfig(
                device='meta',
                quantized=False,
                parallel=ParallelConfig(ep=2, **{dimension: 2}),
            ),
        )


def _check_optimizer_contract(bundle, rank, failure):
    model, optimizer = bundle.chunks[0], bundle.optimizer
    dense = model.embed.weight
    expert = optimizer.expert_parameters[0]
    dense.grad = dense.main_grad = torch.zeros_like(dense)
    expert.grad = expert.main_grad = torch.zeros_like(expert)
    dense.grad.flatten()[0] = 3
    expert.grad.flatten()[0] = 4 if rank == 0 else 12
    norm = optimizer._grad_norm([dense, expert], [dense.grad, expert.grad])
    assert float(norm) == 13.0, 'EP_NORM_COUNTS_DENSE_ONCE_AND_ALL_EXPERTS'
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffers = {
        name: b.detach().contiguous().reshape(-1).view(torch.uint8).clone()
        for name, b in model.named_buffers()
    }
    if failure == 'gradient':
        if rank == 1:
            expert.grad.flatten()[0] = float('inf')
    elif rank == 1:
        optimizer.optimizers[0].prepare_step = lambda: False
    assert not optimizer.step()[0], 'EP_SKIP_MUST_REACH_EVERY_RANK'
    for name, p in model.named_parameters():
        torch.testing.assert_close(
            p, parameters[name], atol=0, rtol=0, msg='EP_SKIP_PARAMETER:' + name
        )
    for name, b in model.named_buffers():
        assert torch.equal(
            b.contiguous().reshape(-1).view(torch.uint8), buffers[name]
        ), ('EP_SKIP_BUFFER:' + name)
    assert all(not backend.state for backend in optimizer.optimizers), 'EP_SKIP_STATE'


@pytest.mark.gpus(2)
@pytest.mark.parametrize('failure', ['gradient', 'candidate'])
def test_ep_global_norm_and_atomic_skip(model_config, failure, tmp_path):
    assert torch.cuda.device_count() >= 2, 'EP optimizer comparison requires two GPUs'
    mp.spawn(
        _ep_worker,
        args=(model_config, True, str(tmp_path), failure),
        nprocs=2,
        join=True,
    )


def _capture_expert_linear_inputs(model, records):
    from megatron.lite.model.deepseek_v41.lite.attention import Linear

    def capture(name, module, args, output):
        if not torch.is_grad_enabled():
            return
        entry = [args[0].detach(), None]
        records.setdefault(name + '.weight', []).append(entry)

        def gradient(value):
            entry[1] = value.detach()

        output.register_hook(gradient)

    from functools import partial

    for name, module in model.named_modules():
        if '.ffn.experts.' in name and isinstance(module, Linear):
            module.register_forward_hook(partial(capture, name))


def _finalize_reference_expert_wgrad(records, reference, rank):
    """Independent serial autograd supplies expert inputs and output gradients.

    EP concatenates source tokens before the weight GEMM. Match that reduction
    order in this oracle, while retaining the serial global-batch objective.
    Never use the parallel model's tensors or gradients as reference values.
    """
    parameters = dict(reference.named_parameters())
    summary = {
        'fp32_partition_difference': 0.0,
        'fp64_partition_difference': 0.0,
        'fp64_rounded_fp32_difference': 0.0,
    }
    for name, entries in records.items():
        xs, dys = zip(*entries)
        xs = [x.reshape(-1, x.shape[-1]) for x in xs]
        dys = [dy.reshape(-1, dy.shape[-1]) for dy in dys]
        for dtype in (torch.float32, torch.float64):
            x = torch.cat(xs).to(dtype)
            dy = torch.cat(dys).to(dtype)
            joined = dy.T @ x
            split = sum(d.to(dtype).T @ a.to(dtype) for a, d in zip(xs, dys))
            key = (
                'fp32_partition_difference'
                if dtype == torch.float32
                else 'fp64_partition_difference'
            )
            summary[key] = max(summary[key], float((joined - split).abs().max()))
            if dtype == torch.float32:
                torch.testing.assert_close(
                    parameters[name].grad,
                    split,
                    atol=0,
                    rtol=0,
                    msg='SERIAL_EXPERT_AUTOGRAD_RECONSTRUCTION:' + name,
                )
                parameters[name].grad = parameters[name].main_grad = joined
            if dtype == torch.float64:
                torch.testing.assert_close(
                    joined.float(),
                    split.float(),
                    atol=0,
                    rtol=0,
                    msg='EXPERT_FP64_REDUCTION_ROUNDED_TO_MASTER:' + name,
                )
                summary['fp64_rounded_fp32_difference'] = max(
                    summary['fp64_rounded_fp32_difference'],
                    float((joined.float() - split.float()).abs().max()),
                )
    print(f'EP_WGRAD_PRECISION rank={rank}: {json.dumps(summary)}', flush=True)


def _missing_ep_peer_worker(rank, directory):
    from megatron.lite.primitive.modules import dispatcher as dispatch_module
    from megatron.lite.primitive.modules import ep_participation
    from megatron.lite.primitive.parallel.state import init_parallel

    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        init_method=(Path(directory) / 'rendezvous').as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        ps = init_parallel(ParallelConfig(ep=2))
        dispatcher = dispatch_module.TokenDispatcher(4, 2, ps, use_deepep=False)
        ep_participation._TIMEOUT = 0.5
        dist.barrier()
        failure = None
        if rank == 0:

            def transport(*args, **kwargs):
                raise AssertionError('EP_MISSING_PEER_REACHED_TRANSPORT')

            dispatch_module.dist.all_gather_into_tensor = transport
            try:
                dispatcher.dispatch(
                    torch.ones(1, 2, device=rank),
                    torch.ones(1, 1, device=rank),
                    torch.zeros(1, 1, device=rank, dtype=torch.long),
                )
            except RuntimeError as error:
                if not (
                    'EP participation failed before dispatch.metadata' in str(error)
                    and 'expected participants=2 actual=1' in str(error)
                ):
                    failure = str(error)
            except AssertionError as error:
                failure = str(error)
            else:
                failure = 'EP_MISSING_PEER_DID_NOT_RAISE'
        failures = [None, None]
        dist.all_gather_object(failures, failure)
        assert not any(failures), f'EP_MISSING_PEER_CONTRACT: {failures}'
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(2)
def test_native_ep_missing_peer_fails_before_transport(tmp_path):
    assert (
        torch.cuda.device_count() >= 2
    ), 'EP participation requires two allocated GPUs'
    mp.spawn(_missing_ep_peer_worker, args=(str(tmp_path),), nprocs=2, join=True)


def _expert_replica_group_worker(rank, config, directory):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        'nccl',
        init_method=(Path(directory) / 'rendezvous').as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        bundle = protocol.build_model(
            config,
            impl_cfg=protocol.ImplConfig(
                parallel=ParallelConfig(ep=2),
                device=f'cuda:{rank}',
                dtype=torch.float32,
                quantized=False,
                token_map=list(range(256)),
                optimizer='muon',
                optimizer_config=OptimizerConfig(0.0001, 5, 'quintic'),
            ),
        )
        ps, optimizer = bundle.parallel_state, bundle.optimizer
        assert ps.dp_size == 4 and ps.ep_size == 2 and ps.expert_dp_size == 2
        assert dist.get_process_group_ranks(ps.dp_group) == [0, 1, 2, 3]
        assert dist.get_process_group_ranks(ps.ep_dp_group) == [rank % 2, rank % 2 + 2]
        dense = bundle.chunks[0].embed.weight
        expert = optimizer.expert_parameters[0]
        dense.grad = dense.main_grad = torch.zeros_like(dense)
        expert.grad = expert.main_grad = torch.zeros_like(expert)
        dense.grad.flatten()[0] = 3
        expert.grad.flatten()[0] = (1 if rank % 2 == 0 else 3) * (4 if rank < 2 else 12)
        bundle.finalize_grads()
        expected = torch.zeros_like(expert)
        expected.flatten()[0] = 4 if rank % 2 == 0 else 12
        correct = torch.tensor(int(torch.equal(expert.grad, expected)), device=rank)
        dist.all_reduce(correct, op=dist.ReduceOp.MIN)
        assert (
            correct.item()
        ), 'EP_REPLICA_REDUCTION_USES_MATCHING_EXPERTS_AND_DENSE_SCALE'
        assert float(dense.grad.flatten()[0]) == 3, 'EP_FINALIZE_MUST_NOT_RESCALE_DENSE'
        norm = optimizer._grad_norm([dense, expert], [dense.grad, expert.grad])
        assert float(norm) == 13.0, 'EP_REPLICA_NORM_COUNTS_UNIQUE_OWNERS'
    finally:
        dist.destroy_process_group()


@pytest.mark.gpus(4)
def test_ep_matching_expert_replica_groups(model_config, tmp_path):
    assert (
        torch.cuda.device_count() >= 4
    ), 'EP replica groups require four allocated GPUs'
    mp.spawn(
        _expert_replica_group_worker,
        args=(model_config, str(tmp_path)),
        nprocs=4,
        join=True,
    )


@pytest.mark.parametrize('ep', [0, -1, 1.5, True])
def test_ep_size_requires_a_positive_integer(moe, model_config, ep):
    with pytest.raises(ValueError, match='EP size must be a positive integer'):
        protocol.build_model(
            model_config,
            impl_cfg=protocol.ImplConfig(
                device='meta', quantized=False, parallel=ParallelConfig(ep=ep)
            ),
        )


def test_empty_expert_preserves_residual_dtype_and_backward(moe, model_config):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    model = DeepseekV41Model(model_config, quantized=False, token_map=list(range(256)))
    ffn = model.layers[0].ffn
    ffn.gate.bias.fill_(-100)
    ffn.gate.bias[:2] = 100
    x = torch.randn(4, 32, dtype=torch.bfloat16, requires_grad=True)
    output = ffn(x)
    assert output.dtype == x.dtype, 'EP_EMPTY_EXPERT_MUST_PRESERVE_RESIDUAL_DTYPE'
    output.float().square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        p.grad is None for expert in ffn.experts[2:] for p in expert.parameters()
    )

"""Slurm TP projection probe; reference is a separate world-one full model.

All comparisons are bitwise. No EP tolerance is applicable to this TP scope.
Sharding the serial initial weights is only an input to the tested TP arm.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'unit' / 'model'))
from test_qwen38_training import tiny_training_config
from qwen38_dp_probe import reduced_gradients
from megatron.lite.model.qwen3_8_flash_next.tp import (
    projection_shard,
    serial_parameter_name,
)
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts import OptimizerConfig, PackedBatch, ParallelConfig


def compare(actual, expected):
    if isinstance(actual, torch.Tensor):
        return (
            actual.dtype == expected.dtype
            and actual.shape == expected.shape
            and torch.equal(actual, expected)
        )
    return actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    assert world in (1, 2), 'TP_WORLD'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(1234)
    run_id = os.environ['QWEN_TP_RUN_ID']
    assert run_id.startswith(os.environ['SLURM_JOB_ID'] + ':'), 'TP_FRESH_REFERENCE'
    args.output.mkdir(parents=True, exist_ok=True)
    config_dir = args.output / f'config-rank{rank}'
    config_dir.mkdir(exist_ok=True)
    config = tiny_training_config()
    (config_dir / 'config.json').write_text(json.dumps(config))
    cfg = MegatronLiteConfig(
        model_name='qwen3_8_flash_next',
        hf_path=str(config_dir),
        load_hf_weights=False,
        optimizer=OptimizerConfig(lr=0.003),
        parallel=ParallelConfig(tp=world, etp=1),
        impl_cfg={
            'ngram_primes': (
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
            )
        },
    )
    runtime = MegatronLiteRuntime(str(config_dir), cfg)
    handle = runtime.build_model()
    ps = handle._parallel_state
    assert (
        ps.tp_size,
        ps.dp_size,
        ps.ep_size,
        ps.etp_size,
        ps.cp_size,
        ps.pp_size,
        ps.expert_dp_size,
    ) == (world, 1, 1, 1, 1, 1, world), 'TP_PARALLEL_SCOPE'
    model = unwrap_model(handle._model)
    reference = torch.load(args.reference, weights_only=True) if world > 1 else None
    if reference is not None:
        assert (
            reference['run_id'] == run_id and reference['world'] == 1
        ), 'TP_REFERENCE_SINGLE_PROCESS'
        assert reference['config'] == config, 'TP_REFERENCE_CONFIG'

    def expected_local(name, value):
        return (
            value.chunk(world, dim=0)[rank].contiguous()
            if world > 1 and projection_shard(name)
            else value
        )

    if reference is not None:
        model.load_state_dict(
            {
                name: expected_local(
                    name, reference['initial'][serial_parameter_name(name)]
                )
                for name in model.state_dict()
            }
        )
        handle._optimizer.reload_model_params()

    def state():
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    initial = state()
    storage = {
        name: dict(
            shape=list(p.shape),
            numel=p.numel(),
            sharded=world > 1 and projection_shard(name),
        )
        for name, p in model.named_parameters()
    }
    local_numel = sum(v['numel'] for v in storage.values())
    if reference is not None:
        assert local_numel < reference['local_numel'], (
            'TP_STORAGE',
            local_numel,
            reference['local_numel'],
        )
        for name, value in initial.items():
            assert compare(
                value,
                expected_local(name, reference['initial'][serial_parameter_name(name)]),
            ), ('TP_INITIAL_WEIGHTS', name)
        for name, p in model.named_parameters():
            assert p.tensor_model_parallel == projection_shard(name), (
                'TP_OPTIMIZER_SCOPE',
                name,
            )
            assert not p.sequence_parallel, ('TP_REPLICA_GRADIENT_SCOPE', name)
        print('TP_STORAGE', rank, local_numel, reference['local_numel'], flush=True)

    # Count actual collectives on the TP group only, only during fwd/bwd.
    calls, shapes, active = {}, {}, [False]
    head_pending = []
    for operation in (
        'all_reduce',
        'all_gather',
        'all_gather_into_tensor',
        'reduce_scatter_tensor',
    ):
        original = getattr(dist, operation)

        def counted(*a, _fn=original, _op=operation, **kw):
            group = kw.get('group')
            if active[0] and group is ps.tp_group and world > 1:
                calls[_op] = calls.get(_op, 0) + 1
                tensors = [x for x in a if isinstance(x, torch.Tensor)]
                if tensors:
                    shapes.setdefault(_op, set()).add(tuple(tensors[-1].shape))
            if _op == 'all_reduce' and head_pending:
                row = head_pending.pop()
                assert group is ps.tp_group and not kw.get(
                    'async_op', False
                ), 'TP_HEAD_REDUCE_SCOPE'
                row['pre_reduce'] = a[0].detach().cpu().clone()
                result = _fn(*a, **kw)
                row['post_reduce'] = a[0].detach().cpu().clone()
                return result
            return _fn(*a, **kw)

        setattr(dist, operation, counted)
    documents = [
        torch.arange(64, device='cuda') % 32,
        (torch.arange(64, device='cuda') * 3 + 47) % 127,
    ]
    batches = [
        PackedBatch(ids, ids.clone(), torch.tensor([64], device='cuda'))
        for ids in documents
    ]
    records = []
    traces = {}
    trace_active = [True]
    for name, module in model.named_modules():
        if name.endswith('.linear'):
            continue
        weight_name = name + (
            '.linear.weight' if hasattr(module, 'linear') else '.weight'
        )
        canonical = serial_parameter_name(weight_name)
        # Same canonical projection set on the complete serial and TP arms.
        probe_name = (
            canonical
            if canonical.endswith('.linear.weight')
            else canonical.replace('.weight', '.linear.weight')
        )
        if not projection_shard(probe_name):
            continue

        def before(module, inputs, key=canonical):
            if not trace_active[0]:
                return
            x = inputs[0].view_as(inputs[0])
            weight = (
                module.linear.weight if hasattr(module, 'linear') else module.weight
            )
            row = {
                'x': x.detach().cpu().clone(),
                'weight': weight.detach().cpu().clone(),
            }
            traces.setdefault(key, []).append(row)
            if x.requires_grad:
                x.register_hook(
                    lambda grad, row=row: row.update(dx=grad.detach().cpu().clone())
                )
            return (x, *inputs[1:])

        def after(module, inputs, output, key=canonical):
            if not trace_active[0]:
                return
            row = traces[key][-1]
            row['y'] = output.detach().cpu().clone()

            def output_gradient(grad):
                row['dy'] = grad.detach().cpu().clone()
                if key == 'lm_head.weight':
                    if world == 1:
                        # Diagnostic slices of the independent full-matrix run.
                        # They are not used by its forward/backward or optimizer.
                        weight = module.weight.detach()
                        row['serial_partials'] = [
                            part.contiguous()
                            .matmul(weight.chunk(2, 0)[owner])
                            .cpu()
                            .clone()
                            for owner, part in enumerate(grad.detach().chunk(2, -1))
                        ]
                    else:
                        assert not head_pending, 'TP_HEAD_REDUCE_PENDING'
                        head_pending.append(row)

            output.register_hook(output_gradient)

        module.register_forward_pre_hook(before)
        module.register_forward_hook(after)

    def train_step():
        outputs = []

        def loss_fn(output, batch):
            outputs.append({k: v.detach().cpu().clone() for k, v in output.items()})
            return output['loss'], {}

        runtime.zero_grad(handle)
        active[0] = True
        runtime.forward_backward(handle, batches, loss_fn, num_microbatches=2)
        active[0] = False
        gradients = reduced_gradients(handle._model, model)
        success, norm, _ = runtime.optimizer_step(handle)
        assert success and 0 < norm < float('inf'), 'TP_OPTIMIZER_STEP'
        return dict(
            outputs=outputs, gradients=gradients, grad_norm=float(norm), state=state()
        )

    for step in range(3):
        record = train_step()
        records.append(record)
        if step == 0:
            torch.save(traces, args.output / f'projection-traces-rank{rank}.pt')
            trace_active[0] = False
            if reference is not None:
                for microbatch, row in enumerate(traces['lm_head.weight']):
                    target = reference['head_partials'][microbatch]
                    assert torch.equal(row['dy'], target['dy']), 'TP_HEAD_PARTIAL_DY'
                    assert torch.equal(
                        row['weight'], target['weight'].chunk(2, 0)[rank]
                    ), 'TP_HEAD_PARTIAL_WEIGHT'
                    assert torch.equal(
                        row['pre_reduce'], target['serial_partials'][rank]
                    ), ('TP_HEAD_PRE_REDUCE_PARTIAL', rank, microbatch)
                    assert torch.equal(
                        row['post_reduce'], row['dx']
                    ), 'TP_HEAD_POST_REDUCE_DX'
                print('TP_HEAD_PRE_REDUCE_PARTIALS_BITWISE_OK', rank, flush=True)
        torch.save(record, args.output / f'step{step}-rank{rank}.pt')
        if step == 1:
            runtime.save_checkpoint(
                handle, str(args.output / 'checkpoint'), step=2, save_rng=False
            )
    restored = runtime.load_checkpoint(
        handle, str(args.output / 'checkpoint'), load_rng=False
    )
    assert restored == 2, 'TP_CHECKPOINT_STEP'
    for name, value in state().items():
        assert compare(value, records[1]['state'][name]), (
            'TP_CHECKPOINT_RESTORE',
            name,
        )
    continued = train_step()
    for field in ('state', 'gradients'):
        for name, value in continued[field].items():
            assert compare(value, records[2][field][name]), (
                'TP_CHECKPOINT_CONTINUATION',
                field,
                name,
            )
    assert continued['grad_norm'] == records[2]['grad_norm'], 'TP_CHECKPOINT_NORM'
    print('TP_SAME_LAYOUT_CONTINUATION_OK', rank, flush=True)
    result = dict(
        run_id=run_id,
        world=world,
        config=config,
        head_partials=traces['lm_head.weight'],
        initial=initial,
        steps=records,
        storage=storage,
        local_numel=local_numel,
        calls=calls,
        communication_shapes={k: sorted(v) for k, v in shapes.items()},
    )
    torch.save(result, args.output / f'result-rank{rank}.pt')
    if world == 1:
        assert not args.reference.exists(), 'TP_REFERENCE_OVERWRITE'
        torch.save(result, args.reference)
    else:
        assert calls.get('all_gather', 0) > 0 and calls.get('all_reduce', 0) > 0, (
            'TP_COMMUNICATION',
            calls,
        )
        print('TP_COMMUNICATION', rank, calls, flush=True)
        mismatches = []
        for step, (record, target) in enumerate(
            zip(records, reference['steps'], strict=True)
        ):
            if record['grad_norm'] != target['grad_norm']:
                mismatches.append(
                    (
                        'TP_OPTIMIZER_LOGICAL_SCOPE',
                        step,
                        record['grad_norm'],
                        target['grad_norm'],
                    )
                )
            for i, (output, expected) in enumerate(
                zip(record['outputs'], target['outputs'], strict=True)
            ):
                for name, value in output.items():
                    if not compare(value, expected[name]):
                        mismatches.append(
                            (
                                'TP_OUTPUT_BITWISE',
                                step,
                                i,
                                name,
                                float(
                                    (value.double() - expected[name].double())
                                    .abs()
                                    .max()
                                ),
                            )
                        )
            for field in ('gradients', 'state'):
                for name, value in record[field].items():
                    expected = expected_local(
                        name, target[field][serial_parameter_name(name)]
                    )
                    if not compare(value, expected):
                        mismatches.append(
                            (
                                'TP_BITWISE',
                                step,
                                field,
                                name,
                                float((value.double() - expected.double()).abs().max()),
                            )
                        )
        (args.output / f'comparison-rank{rank}.json').write_text(
            json.dumps(mismatches, indent=2)
        )
        assert not mismatches, ('TP_BITWISE_PARITY', mismatches[:12])
    print('QWEN38_TP_OK', world, rank, flush=True)
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

"""Slurm-only EP probe: torchrun this once with one rank, then with two ranks.

The one-rank run writes --reference; two-rank runs compare against that file.
Expert owners initialize from the one-rank weights using global expert indices.
Both execute the same two distinct documents per optimizer step, through the
native runtime and MCore distributed optimizer. No checkpoint weights are loaded.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'unit' / 'model'))
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts import OptimizerConfig, PackedBatch, ParallelConfig
from qwen38_dp_probe import reduced_gradients
from test_qwen38_training import tiny_training_config

# Approval applies only to the independently frozen proxy at 2040f053e.
_MODEL_SHA = '59fdd1d953c0c460c4099e77d32d0b52a6699c54c3cd2abc1feb7a9e48e38a5d'
_RUNTIME_SHA = {
    1: '2a105cd62b8f89623e79e30383a9885a0a28d96e743c30c6b1ae08505bb89d0e',
    2: '51a6f736d6b6dd55d23b2e9cdc39d7ebdeca81902927a27b3fee69745d13da40',
}
_DOCS_SHA = '7a2157cd4a3f71166165364bab3bb43193c46f4679f2d7dc0323f6b95dc0d514'
# Fixed existing container/overlay, observed by Slurm 18493124 (2026-09-13).
_ENV = {
    'torch': '2.12.0a0+5aff3928d8.nv26.05',
    'cuda': '13.2',
    'te': '2.15.0+42b84005',
    'cudnn': 92200,
    'gpu': 'NVIDIA H100 80GB HBM3',
    'capability': [9, 0],
    'fla': False,
    'tf32_matmul': True,
    'tf32_cudnn': True,
    'deterministic': False,
}


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def environment():
    import transformer_engine
    from megatron.lite.primitive.modules import gated_delta_net

    result = dict(
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        te=transformer_engine.__version__,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        fla=gated_delta_net._HAS_FLA,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )
    overrides = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(('NVTE_', 'MEGATRON_LITE_', 'MLITE_', 'FLA_', 'CUBLAS_'))
        and k not in ('MLITE_TESTS', 'MLITE_TEST_HARNESS')
    }
    assert overrides == {
        'CUBLAS_VERSION': '13.4.1.1',
        'NVTE_FLASH_ATTN': '1',
        'NVTE_FUSED_ATTN': '0',
        'NVTE_UNFUSED_ATTN': '0',
    }, ('EP_SCOPE_ENV_OVERRIDES', overrides)
    return result


def validate_scope(
    model_config, runtime_config, documents, *, seed, steps, world, dimensions, env
):
    assert world in (1, 2), 'EP_SCOPE_WORLD'
    assert fingerprint(model_config) == _MODEL_SHA, 'EP_SCOPE_MODEL'
    runtime_config = dict(runtime_config, hf_path='')
    assert fingerprint(runtime_config) == _RUNTIME_SHA[world], 'EP_SCOPE_RUNTIME'
    assert fingerprint(documents) == _DOCS_SHA, 'EP_SCOPE_DOCUMENTS'
    assert seed == 1234 and steps == 3, 'EP_SCOPE_SEED_STEPS'
    assert dimensions == {
        'tp': 1,
        'etp': 1,
        'cp': 1,
        'pp': 1,
        'ep': world,
        'dp': world,
        'expert_dp': 1,
    }, 'EP_SCOPE_PARALLEL'
    assert env == _ENV, ('EP_SCOPE_ENVIRONMENT', env)
    return {
        'model': model_config,
        'runtime': runtime_config,
        'documents': documents,
        'seed': seed,
        'steps': steps,
        'world': world,
        'dimensions': dimensions,
        'environment': env,
    }


def compare_tensor(name, actual, expected, kind):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, (
        'EP_TENSOR_LAYOUT',
        name,
    )
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), (
        'EP_FINITE',
        name,
    )
    if torch.equal(actual, expected):
        return {'max_abs': 0.0, 'relative_l2': 0.0, 'changed': 0}
    assert re.fullmatch(r'layers\.[01]\.mlp\.experts\.fc[12]\.weight[0-3]', name), (
        'EP_NONEXPERT_BITWISE',
        name,
        kind,
    )
    diff = actual.double() - expected.double()
    maximum = float(diff.abs().max())
    denominator = float(expected.double().norm())
    relative = float(diff.norm()) / denominator if denominator else float('inf')
    changed = int(torch.count_nonzero(diff))
    if kind == 'gradient':
        assert maximum <= 2e-8 and relative <= 3e-5, (
            'EP_EXPERT_GRADIENT_BOUND',
            name,
            maximum,
            relative,
        )
    else:
        assert kind == 'parameter' and maximum <= 2**-14 and changed <= 3, (
            'EP_EXPERT_PARAMETER_BOUND',
            name,
            maximum,
            changed,
        )
    return {'max_abs': maximum, 'relative_l2': relative, 'changed': changed}


def check_wgrad_oracle(calls, gradients, reference, rank, world):
    # Recompute from THIS run's actual inputs and backward signals. No saved oracle.
    reports = {}
    assert set(calls) == {'0.fc1', '0.fc2', '1.fc1', '1.fc2'}, 'EP_ORACLE_BRANCHES'
    for key, local_calls in calls.items():
        layer, fc = key.split('.')
        assert len(local_calls) == (2 if world == 1 else 1), 'EP_ORACLE_MICROBATCHES'
        for index in range(4 // world):
            name = f'layers.{layer}.mlp.experts.{fc}.weight{index}'
            xs = [c['x'].split(c['splits'])[index].double() for c in local_calls]
            dys = [
                c['dy'].split(c['splits'])[index].double() / world for c in local_calls
            ]
            oracle = sum(
                (dy.T @ x for x, dy in zip(xs, dys)),
                torch.zeros_like(gradients[name], dtype=torch.float64),
            )
            if world == 2:
                global_index = index + rank * 2
                ref_calls = reference['wgrad_calls'][key]
                rx = [
                    c['x'].split(c['splits'])[global_index].double() for c in ref_calls
                ]
                rd = [
                    c['dy'].split(c['splits'])[global_index].double() for c in ref_calls
                ]
                assert torch.equal(xs[0], torch.cat(rx)), ('EP_ORACLE_INPUT', name)
                assert torch.equal(dys[0], torch.cat(rd)), ('EP_ORACLE_DY', name)
                ref_oracle = sum(
                    (dy.T @ x for x, dy in zip(rx, rd)), torch.zeros_like(oracle)
                )
                assert torch.equal(oracle, ref_oracle), ('EP_COMMON_FP64_ORACLE', name)
                ref_grad = reference['gradients'][
                    f'layers.{layer}.mlp.experts.{fc}.weight{global_index}'
                ]
                assert float((ref_grad.double() - oracle).abs().max()) <= 2e-11, (
                    'EP_REFERENCE_ORACLE_ERROR',
                    name,
                )
            error = float((gradients[name].double() - oracle).abs().max())
            assert error <= 2e-11, ('EP_FP32_ORACLE_ERROR', name, error)
            reports[name] = error
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    world = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    assert world in (1, 2), world
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    seed, steps = 1234, 3
    torch.manual_seed(seed)
    args.output.mkdir(parents=True, exist_ok=True)
    config_dir = args.output / f'config-rank{rank}'
    config_dir.mkdir(exist_ok=True)
    (config_dir / 'config.json').write_text(json.dumps(tiny_training_config()))
    cfg = MegatronLiteConfig(
        model_name='qwen3_8_flash_next',
        hf_path=str(config_dir),
        load_hf_weights=False,
        optimizer=OptimizerConfig(lr=0.003),
        parallel=ParallelConfig(ep=world),
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
    assert handle._parallel_state.dp_size == world
    assert handle._parallel_state.expert_dp_size == 1
    model = unwrap_model(handle._model)
    reference = torch.load(args.reference, weights_only=True) if world == 2 else None
    run_id = os.environ['QWEN_EP_RUN_ID']
    assert run_id.startswith(os.environ['SLURM_JOB_ID'] + ':'), 'EP_FRESH_RUN_ID'
    assert args.checkpoint, 'EP_CHECKPOINT_REQUIRED'
    if reference is not None:
        assert reference['run_id'] == run_id, 'EP_STALE_REFERENCE'

    ps = handle._parallel_state
    assert ps.ep_size == world
    local_experts = 4 // world

    def global_name(name):
        if '.experts.' not in name or name.endswith('._extra_state'):
            return name
        match = re.fullmatch(r'(.*\.fc[12]\.weight)([0-9]+)', name)
        assert match, ('EP_UNKNOWN_EXPERT_PARAMETER', name)
        index = int(match[2])
        assert index < local_experts, ('EP_GLOBAL_EXPERT_MATERIALIZED', name)
        return match[1] + str(index + ps.ep_rank * local_experts)

    if reference is not None:
        model.load_state_dict(
            {
                name: reference['initial'][global_name(name)]
                for name in model.state_dict()
            }
        )
        handle._optimizer.reload_model_params()
    calls = {'all_to_all': 0, 'aux': 0}
    original_alltoall = dist.all_to_all_single

    def tracked_alltoall(*args, **kwargs):
        calls['all_to_all'] += 1
        return original_alltoall(*args, **kwargs)

    dist.all_to_all_single = tracked_alltoall
    from megatron.lite.primitive.modules import router as router_module

    original_aux = router_module.switch_load_balancing_loss_func

    def tracked_aux(*args, **kwargs):
        calls['aux'] += 1
        return original_aux(*args, **kwargs)

    router_module.switch_load_balancing_loss_func = tracked_aux
    traces = {}
    for layer_index, layer in enumerate(model.layers):
        assert layer.mlp.experts.num_local_experts == local_experts
        assert layer.mlp.router.aux_loss_coeff == 0.001
        for label, module in [
            ('router', layer.mlp.router),
            ('moe', layer.mlp),
            ('layer', layer),
        ]:
            key = f'{layer_index}.{label}'

            def capture(module, inputs, output, key=key):
                values = output if isinstance(output, tuple) else (output,)
                traces.setdefault(key, []).append(
                    tuple(v.detach().cpu().clone() for v in values)
                )

            module.register_forward_hook(capture)

    wgrad_calls = {}
    for layer_index, layer in enumerate(model.layers):
        for label, fc in [
            ('fc1', layer.mlp.experts.fc1),
            ('fc2', layer.mlp.experts.fc2),
        ]:
            key = f'{layer_index}.{label}'

            def capture_wgrad(module, inputs, output, key=key):
                if records:
                    return
                call = {
                    'x': inputs[0].detach().cpu().clone(),
                    'splits': list(inputs[1]),
                }
                wgrad_calls.setdefault(key, []).append(call)

                def capture_dy(gradient):
                    call['dy'] = gradient.detach().cpu().clone()
                    return gradient

                output.register_hook(capture_dy)

            fc.register_forward_hook(capture_wgrad)

    def state():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    initial = state()
    if reference is not None:
        for name, value in initial.items():
            assert torch.equal(value, reference['initial'][global_name(name)]), (
                'EP_INITIAL_WEIGHTS',
                name,
            )
    documents = [
        torch.arange(64, device='cuda') % 32,
        (torch.arange(64, device='cuda') * 3 + 47) % 127,
    ]
    scope = validate_scope(
        asdict(model.config),
        asdict(handle.config),
        [d.tolist() for d in documents],
        seed=torch.initial_seed(),
        steps=steps,
        world=world,
        dimensions={
            k: getattr(ps, k + '_size')
            for k in ('tp', 'etp', 'cp', 'pp', 'ep', 'dp', 'expert_dp')
        },
        env=environment(),
    )
    if reference is not None:
        assert (
            reference['scope']['model'] == scope['model']
            and reference['scope']['documents'] == scope['documents']
        ), 'EP_REFERENCE_SCOPE'
        ref_scope = reference['scope']
        validate_scope(
            ref_scope['model'],
            ref_scope['runtime'],
            ref_scope['documents'],
            seed=ref_scope['seed'],
            steps=ref_scope['steps'],
            world=1,
            dimensions=ref_scope['dimensions'],
            env=ref_scope['environment'],
        )
    assert not torch.equal(*documents)
    selected = documents if world == 1 else [documents[rank]]
    batches = [
        PackedBatch(ids, ids.clone(), torch.tensor([64], device='cuda'))
        for ids in selected
    ]
    records = []
    for step in range(steps):
        losses = []

        def loss_fn(output, batch):
            losses.append(float(output['loss'].detach()))
            return output['loss'], {}

        runtime.zero_grad(handle)
        runtime.forward_backward(
            handle, batches, loss_fn, num_microbatches=len(batches)
        )
        gradients = reduced_gradients(handle._model, model)
        for name, gradient in gradients.items():
            if '.experts.' in name:
                assert gradient.dtype == torch.float32 and not torch.equal(
                    gradient, gradient.bfloat16().float()
                ), ('EXPERT_WGRAD_BF16_TRUNCATED', name)
        oracle_report = (
            check_wgrad_oracle(
                wgrad_calls,
                gradients,
                reference['steps'][0] if reference else None,
                rank,
                world,
            )
            if step == 0
            else {}
        )
        success, norm, _ = runtime.optimizer_step(handle)
        assert success and 0 < norm < float('inf'), ('EP_OPTIMIZER_STEP', step, norm)
        if world == 2:
            gathered = [None] * world
            dist.all_gather_object(gathered, losses[0])
            losses = gathered
        current = state()
        if world == 2:
            for name, param in model.named_parameters():
                if '.experts.' in name:
                    continue
                replica = param.detach().clone()
                dist.broadcast(replica, src=0)
                assert torch.equal(param, replica), ('EP_REPLICA_MISMATCH', step, name)
        record = {
            'losses': losses,
            'grad_norm': float(norm),
            'state': current,
            'gradients': gradients,
            'traces': traces,
            'calls': dict(calls),
            'wgrad_calls': wgrad_calls if step == 0 else {},
            'oracle_errors': oracle_report,
        }
        records.append(record)
        if args.checkpoint and step == 1:
            runtime.save_checkpoint(
                handle, str(args.output / 'checkpoint'), step=2, save_rng=False
            )
        traces = {}
        if reference is not None:
            torch.save(record, args.output / f'step{step}-rank{rank}.pt')
            expected = reference['steps'][step]
            assert losses == expected['losses'], ('EP_LOSS_BITWISE', step)
            assert float(norm) == expected['grad_norm'], ('EP_NORM_BITWISE', step)
            if step == 0:
                for key, values in record['traces'].items():
                    for actual, target in zip(
                        values[0], expected['traces'][key][rank], strict=True
                    ):
                        assert torch.equal(actual, target), ('EP_TRACE_BITWISE', key)
            report = {}
            for field, kind in [('gradients', 'gradient'), ('state', 'parameter')]:
                report[field] = {
                    name: compare_tensor(
                        name, value, expected[field][global_name(name)], kind
                    )
                    for name, value in record[field].items()
                }
            print(
                'EP_APPROVED_PROXY_COMPARISON',
                rank,
                step,
                json.dumps(report),
                flush=True,
            )

    assert sum(records[-1]['losses']) < sum(
        records[0]['losses']
    ), 'EP_LOSS_NOT_DECREASING'
    assert any(
        not torch.equal(initial[k], records[-1]['state'][k]) for k in initial
    ), 'EP_NO_UPDATE'
    assert calls['aux'] == 2 * 3 * len(batches), ('EP_AUX_BRANCH_MISSING', calls)
    assert world == 1 or calls['all_to_all'] >= 2 * 3 * 6, (
        'EP_ALLTOALL_NOT_EXECUTED',
        calls,
    )
    if args.checkpoint:
        restored_step = runtime.load_checkpoint(
            handle, str(args.output / 'checkpoint'), load_rng=False
        )
        assert restored_step == 2, ('EP_CHECKPOINT_STEP', restored_step)
        for name, value in state().items():
            assert torch.equal(value, records[1]['state'][name]), (
                'EP_CHECKPOINT_RESTORE',
                name,
            )
        losses = []
        runtime.zero_grad(handle)
        runtime.forward_backward(
            handle, batches, loss_fn, num_microbatches=len(batches)
        )
        restored_gradients = reduced_gradients(handle._model, model)
        success, norm, _ = runtime.optimizer_step(handle)
        assert success
        for name, value in state().items():
            assert torch.equal(value, records[2]['state'][name]), (
                'EP_CHECKPOINT_CONTINUATION',
                name,
            )
        for name, value in restored_gradients.items():
            assert torch.equal(value, records[2]['gradients'][name]), (
                'EP_CHECKPOINT_GRADIENT',
                name,
            )
        print('QWEN38_SAME_EP_CHECKPOINT_CONTINUITY_OK', world, rank, flush=True)
    result = {'run_id': run_id, 'scope': scope, 'initial': initial, 'steps': records}
    torch.save(result, args.output / f'result-rank{rank}.pt')
    if rank == 0:
        torch.save(result, args.output / 'result.pt')
        if world == 1:
            assert not args.reference.exists(), 'REFUSE_OVERWRITE_REFERENCE'
            torch.save(result, args.reference)
        print('QWEN38_EP_OK', world, [r['losses'] for r in records], flush=True)
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

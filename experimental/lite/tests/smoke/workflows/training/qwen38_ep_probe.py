"""Slurm-only EP probe: torchrun this once with one rank, then with two ranks.

The one-rank run writes --reference; two-rank runs compare against that file.
Expert owners initialize from the one-rank weights using global expert indices.
Both execute the same two distinct documents per optimizer step, through the
native runtime and MCore distributed optimizer. No checkpoint weights are loaded.
"""

import argparse
import json
import os
import re
import sys
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    world = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    assert world in (1, 2), world
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(1234)
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
    assert not torch.equal(*documents)
    selected = documents if world == 1 else [documents[rank]]
    batches = [
        PackedBatch(ids, ids.clone(), torch.tensor([64], device='cuda'))
        for ids in selected
    ]
    records = []
    mismatches = []
    for step in range(3):
        losses = []

        def loss_fn(output, batch):
            losses.append(float(output['loss'].detach()))
            return output['loss'], {}

        runtime.zero_grad(handle)
        runtime.forward_backward(
            handle, batches, loss_fn, num_microbatches=len(batches)
        )
        gradients = reduced_gradients(handle._model, model)
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
        }
        records.append(record)
        traces = {}
        if reference is not None:
            torch.save(record, args.output / f'step{step}-rank{rank}.pt')
            expected = reference['steps'][step]
            grad_diffs = {
                name: float(
                    (value - expected['gradients'][global_name(name)]).abs().max()
                )
                for name, value in gradients.items()
                if not torch.equal(value, expected['gradients'][global_name(name)])
            }
            diffs = {
                name: float(
                    (value.float() - expected['state'][global_name(name)].float())
                    .abs()
                    .max()
                )
                for name, value in current.items()
                if not torch.equal(value, expected['state'][global_name(name)])
            }
            print(
                'EP_COMPARISON',
                rank,
                step,
                'losses',
                losses,
                expected['losses'],
                'norm',
                float(norm),
                expected['grad_norm'],
                'gradient_max_abs',
                grad_diffs,
                'parameter_max_abs',
                diffs,
                flush=True,
            )
            mismatches.append((step, losses != expected['losses'], grad_diffs, diffs))
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
    result = {'initial': initial, 'steps': records}
    torch.save(result, args.output / f'result-rank{rank}.pt')
    assert not any(loss or grads or params for _, loss, grads, params in mismatches), (
        'EP_BITWISE_PARITY',
        [
            (step, loss, len(grads), len(params))
            for step, loss, grads, params in mismatches
        ],
    )
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

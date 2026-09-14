"""Slurm-only DP probe: torchrun this once with one rank, then with two ranks.

The one-rank run writes --reference; two-rank runs compare against that file.
Both execute the same two distinct documents per optimizer step, through the
native runtime and MCore distributed optimizer. No checkpoint weights are loaded.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'unit' / 'model'))
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts import OptimizerConfig, PackedBatch
from test_qwen38_training import tiny_training_config


def reduced_gradients(wrapped, model):
    # MCore fd1121b8, param_and_grad_buffer.py:657-667 stores the only valid
    # reduce-scatter output in each bucket's rank-local contiguous shard.
    names = {param: name for name, param in model.named_parameters()}
    result = {}
    for buffer in [*wrapped.buffers, *wrapped.expert_parallel_buffers]:
        group = buffer.data_parallel_group
        for bucket in buffer.buckets:
            local = bucket.grad_data.chunk(dist.get_world_size(group))[
                dist.get_rank(group)
            ]
            gathered = [
                torch.empty_like(local) for _ in range(dist.get_world_size(group))
            ]
            dist.all_gather(gathered, local.contiguous(), group=group)
            full = torch.cat(gathered)
            assert full.dtype == torch.float32, 'DP_REDUCE_NOT_FP32'
            for param, (start, end) in bucket.param_to_index.items():
                result[names[param]] = (
                    full[start:end].reshape(param.shape).cpu().clone()
                )
    assert set(result) == {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    return result


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
    assert handle._parallel_state.expert_dp_size == world
    model = unwrap_model(handle._model)
    reference = torch.load(args.reference, weights_only=True) if world == 2 else None

    def state():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    initial = state()
    if reference is not None:
        for name, value in initial.items():
            assert torch.equal(value, reference['initial'][name]), (
                'DP_INITIAL_WEIGHTS',
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
        assert success and 0 < norm < float('inf'), ('DP_OPTIMIZER_STEP', step, norm)
        if world == 2:
            gathered = [None] * world
            dist.all_gather_object(gathered, losses[0])
            losses = gathered
        current = state()
        if world == 2:
            for name, param in model.named_parameters():
                replica = param.detach().clone()
                dist.broadcast(replica, src=0)
                assert torch.equal(param, replica), ('DP_REPLICA_MISMATCH', step, name)
        record = {
            'losses': losses,
            'grad_norm': float(norm),
            'state': current,
            'gradients': gradients,
        }
        records.append(record)
        if reference is not None:
            torch.save(record, args.output / f'step{step}-rank{rank}.pt')
            expected = reference['steps'][step]
            grad_diffs = {
                name: float((value - expected['gradients'][name]).abs().max())
                for name, value in gradients.items()
                if not torch.equal(value, expected['gradients'][name])
            }
            diffs = {
                name: float(
                    (value.float() - expected['state'][name].float()).abs().max()
                )
                for name, value in current.items()
                if not torch.equal(value, expected['state'][name])
            }
            print(
                'DP_COMPARISON',
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
            assert losses == expected['losses'], ('DP_LOSS_PARITY', step)
            assert not grad_diffs, ('DP_REDUCED_GRADIENT_PARITY', step, grad_diffs)
            assert not diffs, ('DP_PARAMETER_UPDATE_PARITY', step, diffs)
    assert sum(records[-1]['losses']) < sum(
        records[0]['losses']
    ), 'DP_LOSS_NOT_DECREASING'
    assert any(
        not torch.equal(initial[k], records[-1]['state'][k]) for k in initial
    ), 'DP_NO_UPDATE'
    result = {'initial': initial, 'steps': records}
    if rank == 0:
        torch.save(result, args.output / 'result.pt')
        if world == 1:
            assert not args.reference.exists(), 'REFUSE_OVERWRITE_REFERENCE'
            torch.save(result, args.reference)
        print('QWEN38_DP_OK', world, [r['losses'] for r in records], flush=True)
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

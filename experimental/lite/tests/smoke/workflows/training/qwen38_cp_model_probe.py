"""Native CP model execution; preserve raw boundaries before asserting parity."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'unit' / 'model'))
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts import OptimizerConfig, PackedBatch, ParallelConfig
from qwen38_dp_probe import reduced_gradients
from test_qwen38_training import tiny_training_config

PRIMES = (17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79)


def configuration():
    config = tiny_training_config()
    config.update(linear_num_key_heads=2, linear_num_value_heads=4)
    return config


def build(directory, cp, *, ep=1, ple_owner_sharding=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(configuration()))
    cfg = MegatronLiteConfig(
        model_name='qwen3_8_flash_next',
        hf_path=str(directory),
        load_hf_weights=False,
        optimizer=OptimizerConfig(lr=0.003),
        parallel=ParallelConfig(cp=cp, ep=ep, etp=1),
        impl_cfg={'ngram_primes': PRIMES, 'ple_owner_sharding': ple_owner_sharding},
    )
    runtime = MegatronLiteRuntime(str(directory), cfg)
    handle = runtime.build_model()
    return runtime, handle, unwrap_model(handle._model)


def batch_for_step(step):
    ids = (torch.arange(13, device='cuda') * 3 + 7 + step) % 127
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[10] = False
    return PackedBatch(
        ids, ids.clone(), torch.tensor([5, 8], device='cuda'), loss_mask=mask
    )


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    args = parser.parse_args()
    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    assert world in (1, 2)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(1234)
    runtime, handle, model = build(args.output / f'config-{rank}', world)
    if world == 2:
        model.load_state_dict(torch.load(args.initial, weights_only=True))
        handle._optimizer.reload_model_params()
    initial = snapshot(model)
    if world == 1:
        torch.save(initial, args.initial)
    traces = {}
    hooks = []
    for name, child in model.named_modules():
        if name and any(
            name.endswith(s)
            for s in (
                'linear_attn',
                'self_attn',
                'ple',
                'mlp',
                'embed_tokens',
                'lm_head',
            )
        ):

            def hook(module, inputs, output, name=name):
                traces[name] = {
                    'input': inputs[0].detach().cpu().clone(),
                    'output': output.detach().cpu().clone(),
                }
                if output.requires_grad:
                    output.register_hook(
                        lambda grad, name=name: traces[name].update(
                            dy=grad.detach().cpu().clone()
                        )
                    )
                if inputs[0].requires_grad:
                    inputs[0].register_hook(
                        lambda grad, name=name: traces[name].update(
                            dx=grad.detach().cpu().clone()
                        )
                    )

            hooks.append(child.register_forward_hook(hook))
    outputs = {}

    def objective(output, batch):
        outputs.update({k: v.detach().cpu().clone() for k, v in output.items()})
        return output['loss'], {}

    runtime.zero_grad(handle)
    runtime.forward_backward(handle, [batch_for_step(0)], objective, num_microbatches=1)
    gradients = reduced_gradients(handle._model, model)
    scaling = [
        dict(
            scale=float(b.gradient_scaling_factor),
            group=dist.get_process_group_ranks(b.data_parallel_group),
        )
        for b in [*handle._model.buffers, *handle._model.expert_parallel_buffers]
    ]
    success, norm, _ = runtime.optimizer_step(handle)
    result = dict(
        world=world,
        rank=rank,
        initial=initial,
        outputs=outputs,
        gradients=gradients,
        state=snapshot(model),
        traces=traces,
        scaling=scaling,
        norm=norm,
        success=success,
    )
    torch.save(result, args.output / f'rank{rank}.pt')
    print('CP_MODEL_NATIVE_STEP', rank, success, norm, scaling, flush=True)
    assert success, 'CP_MODEL_OPTIMIZER_STEP'
    for h in hooks:
        h.remove()
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

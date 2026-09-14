"""PLE owner capacity proof using real model construction and native dist_opt.

The independent full table uses source-rank-concatenated, externally specified
requests and cotangents. It never consumes tested outputs or gradients. Only
PLE participates in these parity steps; a separate full-model step is smoke.
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from qwen38_cp_model_probe import PRIMES, batch_for_step, configuration, snapshot
from qwen38_dp_probe import reduced_gradients
from torch.distributed.elastic.multiprocessing.errors import record

NAME = 'layers.1.ple.ple_embedding.ngram_embedding.weight'


def build(directory, mutation=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(configuration()))
    cfg = MegatronLiteConfig(
        model_name='qwen3_8_flash_next',
        hf_path=str(directory),
        load_hf_weights=False,
        optimizer=OptimizerConfig(lr=0.003),
        parallel=ParallelConfig(ep=2, etp=1),
        impl_cfg={'ngram_primes': PRIMES, 'ple_owner_sharding': True},
    )
    runtime = MegatronLiteRuntime(str(directory), cfg)
    from megatron.lite.model.qwen3_8_flash_next import protocol

    classifier = protocol.is_expert_param
    if mutation == 'optimizer_group':

        def wrong(name, **kwargs):
            return False if name == NAME else classifier(name, **kwargs)

        with patch.object(protocol, 'is_expert_param', wrong):
            handle = runtime.build_model()
    else:
        handle = runtime.build_model()
    model = unwrap_model(handle._model)
    return runtime, handle, model


def table(model):
    return model.layers[1].ple.ple_embedding.ngram_embedding


def equal_dict(a, b):
    return a.keys() == b.keys() and all(torch.equal(v, b[n]) for n, v in a.items())


def requests(rows, width, step):
    # Definition contains repeated IDs and requests crossing owner boundaries.
    ids = [
        torch.tensor(v, device='cuda')
        for v in (
            [rows - 1, 0, rows // 2, 1, rows - 1, 0, 5],
            [1, rows // 2, 1, rows - 2, 2],
        )
    ]
    grad = [
        (
            (
                torch.arange(x.numel() * width, device='cuda').reshape(-1, width)
                + 7 * r
                + step
            )
            .remainder(19)
            .float()
            .sub(9)
            .div(16)
        ).to(torch.bfloat16)
        for r, x in enumerate(ids)
    ]
    return ids, grad


def one_step(runtime, handle, model, ids, dy, mutation=None):
    runtime.zero_grad(handle)
    t = table(model)
    output = t(ids)
    output.backward(dy)
    before = t.weight.main_grad.detach().clone()
    handle._extras['finalize_grads']()
    reduced = reduced_gradients(handle._model, model)
    success, norm, _ = runtime.optimizer_step(handle)
    return dict(
        output=output.detach().cpu(),
        before=before.cpu(),
        gradients=reduced,
        state=snapshot(model),
        success=success,
        norm=norm,
    )


def install_mutation(model, mutation):
    # Only the tested product module is changed; the full reference never calls it.
    if mutation == 'return_order':
        table(model).register_forward_hook(lambda m, args, output: output.flip(0))
    elif mutation == 'skip_contribution' and dist.get_rank() == 1:
        def drop(m, args, output):
            def skip(grad):
                grad = grad.clone()
                grad[0] = 0
                return grad
            output.register_hook(skip)
        table(model).register_forward_hook(drop)


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--mutation', choices=['return_order', 'skip_contribution', 'optimizer_group']
    )
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2, 'PLE_OWNER_TWO_RANKS'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(1234)
    rr, rh, reference = build(args.output / f'reference-{rank}')
    rt = table(reference)
    rows, width = rt.lookup.num_embeddings, rt.weight.shape[1]
    # Full reference values exist before tested construction and are independent.
    full = (
        torch.arange(rows * width, device='cuda')
        .remainder(97)
        .float()
        .sub(48)
        .div(128)
        .reshape(rows, width)
        .to(torch.bfloat16)
    ).requires_grad_()
    with torch.no_grad():
        rt.weight.copy_(full[rt.global_row_start : rt.global_row_end])
    rh._optimizer.reload_model_params()
    runtime, handle, model = build(args.output / f'tested-{rank}', args.mutation)
    model.load_state_dict(reference.state_dict())
    handle._optimizer.reload_model_params()
    t = table(model)
    install_mutation(model, args.mutation)
    b = next(
        b
        for b in [*handle._model.buffers, *handle._model.expert_parallel_buffers]
        if t.weight in b.param_index_map
    )
    metadata = dict(
        local_shape=list(t.weight.shape),
        global_shape=[rows, width],
        main_grad_shape=list(t.weight.main_grad.shape),
        main_grad_dtype=str(t.weight.main_grad.dtype),
        owner_group=dist.get_process_group_ranks(t.lookup.process_group),
        gradient_group=dist.get_process_group_ranks(b.data_parallel_group),
        gradient_scale=float(b.gradient_scaling_factor),
        weight_bytes=t.weight.numel() * t.weight.element_size(),
        main_grad_bytes=t.weight.main_grad.numel() * t.weight.main_grad.element_size(),
    )
    print('PLE_OWNER_NATIVE_STORAGE', rank, json.dumps(metadata), flush=True)
    records = []
    for step in range(3):
        ids, dy = requests(rows, width, step)
        full.grad = None
        expected = torch.nn.functional.embedding(torch.cat(ids), full)
        expected.backward(torch.cat(dy))
        local_grad = full.grad[t.global_row_start : t.global_row_end].float()
        rr.zero_grad(rh)
        rt.weight.main_grad.copy_(local_grad)
        rh._extras['finalize_grads']()
        expected_grad = reduced_gradients(rh._model, reference)
        expected_success, expected_norm, _ = rr.optimizer_step(rh)
        actual = one_step(runtime, handle, model, ids[rank], dy[rank], args.mutation)
        expected_output = expected.detach().split([x.numel() for x in ids])[rank].cpu()
        checks = {
            'PLE_OWNER_FORWARD_BITWISE': torch.equal(actual['output'], expected_output),
            'PLE_OWNER_LOCAL_GRAD_BITWISE': torch.equal(
                actual['before'], local_grad.cpu()
            ),
            'PLE_OWNER_REDUCED_GRAD_BITWISE': equal_dict(
                actual['gradients'], expected_grad
            ),
            'PLE_OWNER_OPTIMIZER_STEP_BITWISE': equal_dict(
                actual['state'], snapshot(reference)
            ),
            'PLE_OWNER_NORM_BITWISE': actual['norm'] == expected_norm,
            'PLE_OWNER_SUCCESS': bool(actual['success'] and expected_success),
            'PLE_OWNER_LOCAL_STORAGE': t.weight.numel() * 2 == rows * width
            and t.weight.main_grad.shape == t.weight.shape
            and t.weight.main_grad.dtype == torch.float32,
            'PLE_OWNER_OPTIMIZER_GROUP': metadata['gradient_group'] == [rank]
            and metadata['gradient_scale'] == 0.5
            and not t.weight.allreduce,
        }
        record = dict(
            actual=actual,
            checks=checks,
            reference_output=expected_output,
            reference_local_grad=local_grad.cpu(),
            reference_gradients=expected_grad,
            reference_state=snapshot(reference),
            reference_norm=expected_norm,
            metadata=metadata,
        )
        torch.save(record, args.output / f'step{step}-rank{rank}.pt')
        print('PLE_OWNER_CHECKS', rank, step, json.dumps(checks), flush=True)
        if args.mutation:
            required = {
                'return_order': (
                    'PLE_OWNER_FORWARD_BITWISE',
                    ['PLE_OWNER_LOCAL_STORAGE', 'PLE_OWNER_OPTIMIZER_GROUP'],
                ),
                'skip_contribution': (
                    'PLE_OWNER_LOCAL_GRAD_BITWISE',
                    [
                        'PLE_OWNER_FORWARD_BITWISE',
                        'PLE_OWNER_LOCAL_STORAGE',
                        'PLE_OWNER_OPTIMIZER_GROUP',
                    ],
                ),
                'optimizer_group': (
                    'PLE_OWNER_OPTIMIZER_GROUP',
                    [
                        'PLE_OWNER_FORWARD_BITWISE',
                        'PLE_OWNER_LOCAL_GRAD_BITWISE',
                        'PLE_OWNER_LOCAL_STORAGE',
                    ],
                ),
            }
            target, others = required[args.mutation]
            failed = [tag for tag, passed in checks.items() if not passed]
            detected = target in failed
            # skip_contribution's local failure belongs to owner of row 1 (rank0).
            flag = torch.tensor(int(detected), device='cuda')
            dist.all_reduce(flag)
            rejection = dict(
                rank=rank,
                mutation=args.mutation,
                failures=failed,
                non_target_passed={tag: checks[tag] for tag in others},
                detected_some_rank=bool(flag.item()),
            )
            print(
                'PLE_OWNER_MUTATION_NAMED_REJECTION', json.dumps(rejection), flush=True
            )
            dist.barrier()
            assert flag.item() and all(checks[tag] for tag in others), (
                'PLE_OWNER_MUTATION_INVALID',
                rejection,
            )
            assert not flag.item(), (target, rank, args.mutation)
        for tag, ok in checks.items():
            assert ok, (tag, rank, step)
        records.append(actual)
        # Next reference state comes solely from the independent native optimizer.
        parts = [torch.empty_like(rt.weight) for _ in range(2)]
        dist.all_gather(parts, rt.weight.detach(), group=rt.lookup.process_group)
        with torch.no_grad():
            full.copy_(torch.cat(parts))
        if step == 1:
            runtime.save_checkpoint(
                handle, str(args.output / 'checkpoint'), step=2, save_rng=False
            )
    restored = runtime.load_checkpoint(
        handle, str(args.output / 'checkpoint'), load_rng=False
    )
    restore_equal = equal_dict(snapshot(model), records[1]['state'])
    ids, dy = requests(rows, width, 2)
    continuation = one_step(runtime, handle, model, ids[rank], dy[rank])
    checks = dict(
        PLE_OWNER_CHECKPOINT_STEP=restored == 2,
        PLE_OWNER_CHECKPOINT_RESTORE=restore_equal,
        PLE_OWNER_CHECKPOINT_CONTINUATION=equal_dict(
            continuation['state'], records[2]['state']
        ),
        PLE_OWNER_CHECKPOINT_GRAD=equal_dict(
            continuation['gradients'], records[2]['gradients']
        ),
        PLE_OWNER_CHECKPOINT_NORM=continuation['norm'] == records[2]['norm'],
    )
    torch.save(
        dict(continuation=continuation, checks=checks),
        args.output / f'continued-rank{rank}.pt',
    )
    print('PLE_OWNER_CHECKPOINT_CHECKS', rank, json.dumps(checks), flush=True)
    for tag, ok in checks.items():
        assert ok, (tag, rank)
    # Actual whole-model wiring; this step is explicitly connectivity, not parity.
    runtime.zero_grad(handle)
    runtime.forward_backward(
        handle, [batch_for_step(rank)], lambda o, b: (o['loss'], {}), num_microbatches=1
    )
    success, norm, _ = runtime.optimizer_step(handle)
    print('PLE_OWNER_FULL_MODEL_SMOKE', rank, success, norm, flush=True)
    assert success, 'PLE_OWNER_FULL_MODEL_SMOKE'
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

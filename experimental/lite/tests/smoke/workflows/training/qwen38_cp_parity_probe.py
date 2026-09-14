"""Independent virtual CP models versus actual CP2, including native optimizer."""

import argparse
import json
import os
from copy import copy
from pathlib import Path

import torch
import torch.distributed as dist
from fla_kernel_observer import FLAKernelObserver
from qwen38_cp_model_probe import PRIMES, batch_for_step, build, snapshot
from qwen38_cp_reference import forward as ordered_forward
from qwen38_dp_probe import reduced_gradients
from torch.distributed.elastic.multiprocessing.errors import record


def build_virtual(reference):
    from megatron.lite.model.qwen3_8_flash_next.model import Qwen38Model

    ps = copy(reference.ps)
    ps.cp_size, ps.cp_rank, ps.cp_group = 1, 0, None
    result = []
    for _ in range(2):
        model = Qwen38Model(
            reference.config, ps, ngram_primes=PRIMES, fuse_wgrad_accumulation=True
        ).to(dtype=torch.bfloat16, device='cuda')
        model.load_state_dict(reference.state_dict())
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
            p.grad_added_to_main_grad = False

            def accumulate(param):
                # Same leaf storage contract as MCore DDP; no collective/reference
                # gradient enters here. Native TE grouped wgrad writes FP32 directly.
                if param.grad is not None and not param.grad_added_to_main_grad:
                    param.main_grad.add_(param.grad)
                param.grad = None

            p.register_post_accumulate_grad_hook(accumulate)
        result.append(model)
    return result


def gradients(model):
    return {
        n: p.main_grad.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def compare(actual, expected):
    return {
        n: int((v != expected[n]).count_nonzero())
        for n, v in actual.items()
        if not torch.equal(v, expected[n])
    }


def inject_mutation(model, mutation, rank):
    """Change only the tested arm's real computation, never the reference."""
    if mutation in ('boundary_target', 'loss_population'):

        def alter(module, args, kwargs):
            kwargs = dict(kwargs)
            if mutation == 'boundary_target' and rank == 0:
                # Incorrect shard-local EOS at global query 7 inside document 5..13.
                kwargs['labels'] = kwargs['labels'].clone()
                kwargs['labels'][0, -1] = -100
            elif mutation == 'loss_population':
                # Router population (13) is not the valid shifted-target count (10).
                kwargs['loss_token_count'] = kwargs['loss_token_count'].new_tensor(13)
            return args, kwargs

        model.register_forward_pre_hook(alter, with_kwargs=True)
    elif mutation == 'router_padding':

        def include_padding(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs['token_mask'] = torch.ones_like(kwargs['token_mask'])
            return args, kwargs

        for layer in model.layers:
            layer.mlp.router.register_forward_pre_hook(
                include_padding, with_kwargs=True
            )


def mutation_record(mutation, rank, checks):
    non_targets = ['CP_MODEL_ORDERED_FORWARD', 'CP_MODEL_EXECUTION_CONTRACT']
    if mutation == 'router_padding' or (mutation == 'boundary_target' and rank == 1):
        non_targets.append('CP_MODEL_ORDERED_LOSS')
    failures = [tag for tag, passed in checks.items() if not passed]
    return dict(
        mutation=mutation,
        rank=rank,
        failures=failures,
        non_target_passed={tag: checks[tag] for tag in non_targets},
        detected='CP_MODEL_ORDERED_REDUCED_GRAD' in failures
        and all(checks[tag] for tag in non_targets),
    )


def train_actual(runtime, handle, model, batch, rank):
    saved = {'tested_trace': {}}
    hooks = []
    for name, child in model.named_modules():
        if name and any(
            name.endswith(part) for part in ('linear_attn', 'self_attn', 'ple', 'mlp')
        ):

            def observe(module, args, output, name=name):
                from qwen38_cp_reference import capture

                capture(saved['tested_trace'], rank, name, 'input', args[0])
                capture(saved['tested_trace'], rank, name, 'output', output)

            hooks.append(child.register_forward_hook(observe))
    original_finalize = handle._extras['finalize_grads']

    def finalize():
        saved['local_gradients'] = gradients(model)
        original_finalize()

    handle._extras['finalize_grads'] = finalize

    def objective(output, batch):
        saved['outputs'] = {k: v.detach().cpu().clone() for k, v in output.items()}
        return output['loss'], {}

    runtime.zero_grad(handle)
    observer = FLAKernelObserver()
    original_exchange = dist.all_to_all_single
    exchanges = []

    def exchange(*args, **kwargs):
        if kwargs.get('group') is model.ps.cp_group:
            exchanges.append(
                [list(x.shape) for x in args if isinstance(x, torch.Tensor)]
            )
        return original_exchange(*args, **kwargs)

    dist.all_to_all_single = exchange
    try:
        with observer:
            runtime.forward_backward(handle, [batch], objective, num_microbatches=1)
    finally:
        dist.all_to_all_single = original_exchange
        handle._extras['finalize_grads'] = original_finalize
        for hook in hooks:
            hook.remove()
    from megatron.lite.primitive.ops.fla_l2norm import (
        assert_kernel_configs_match,
        kernel_policy,
    )

    observed = [
        {k: row[k] for k in ('name', 'shape', 'dtype', 'config')}
        for row in observer.records
    ]
    per_rank = [None, None]
    dist.all_gather_object(per_rank, observed, group=model.ps.cp_group)
    assert_kernel_configs_match(per_rank)
    saved.update(
        actual_fla_kernels=observer.records,
        kernel_policy=kernel_policy(),
        cp_exchanges=exchanges,
    )
    saved['execution_contract'] = len(exchanges) == 4 and all(
        values['input'].shape[0 if name.endswith('linear_attn') else 1] == 8
        for name, values in saved['tested_trace'][str(rank)].items()
    )
    actual_gradients = reduced_gradients(handle._model, model)
    success, norm, _ = runtime.optimizer_step(handle)
    saved.update(
        gradients=actual_gradients, success=success, norm=norm, state=snapshot(model)
    )
    return saved


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--mutation', choices=['boundary_target', 'loss_population', 'router_padding']
    )
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 2, 'CP_PARITY_TWO_RANKS'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.manual_seed(1234)
    # Reference optimizer/model owns its initial state before tested construction.
    rr, rh, reference = build(args.output / f'reference-config-{rank}', 2)
    initial = snapshot(reference)
    runtime, handle, model = build(args.output / f'tested-config-{rank}', 2)
    model.load_state_dict(initial)
    handle._optimizer.reload_model_params()
    virtual = build_virtual(reference)
    inject_mutation(model, args.mutation, rank)
    records = []
    for step in range(3):
        for m in virtual:
            m.load_state_dict(reference.state_dict())
            for p in m.parameters():
                if p.requires_grad:
                    p.main_grad.zero_()
                    p.grad = None
                    p.grad_added_to_main_grad = False
        from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

        MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.0, device='cuda'))
        batch = batch_for_step(step)
        reference_trace = {}
        predicted, losses = ordered_forward(virtual, batch, reference_trace)
        sum(losses).backward()
        local_references = [gradients(m) for m in virtual]
        # A separate native CP optimizer consumes independently computed local
        # gradients, reproducing owner ranges, averaging and clipping order.
        rr.zero_grad(rh)
        for n, p in reference.named_parameters():
            if p.requires_grad:
                p.main_grad.copy_(local_references[rank][n])
        rh._extras['finalize_grads']()
        expected_gradients = reduced_gradients(rh._model, reference)
        expected_success, expected_norm, _ = rr.optimizer_step(rh)
        saved = train_actual(runtime, handle, model, batch, rank)
        saved['reference_trace'] = reference_trace
        actual_gradients, success, norm = (
            saved['gradients'],
            saved['success'],
            saved['norm'],
        )
        saved.update(
            initial=initial,
            local_references=local_references,
            reference_logits=predicted[rank].detach().cpu(),
            reference_loss=losses[rank].detach().cpu(),
            expected_gradients=expected_gradients,
            gradients=actual_gradients,
            reference_state=snapshot(reference),
            state=snapshot(model),
            norm=norm,
            reference_norm=expected_norm,
        )
        checks = {
            'CP_MODEL_EXECUTION_CONTRACT': saved['execution_contract'],
            'CP_MODEL_ORDERED_FORWARD': torch.equal(
                saved['outputs']['logits'], saved['reference_logits']
            ),
            'CP_MODEL_ORDERED_LOSS': torch.equal(
                saved['outputs']['loss'], saved['reference_loss']
            ),
            'CP_MODEL_PRE_REDUCTION_GRAD': not compare(
                saved['local_gradients'], local_references[rank]
            ),
            'CP_MODEL_ORDERED_REDUCED_GRAD': not compare(
                actual_gradients, expected_gradients
            ),
            'CP_MODEL_ORDERED_OPTIMIZER_STEP': success
            and expected_success
            and not compare(saved['state'], saved['reference_state']),
            'CP_MODEL_ORDERED_NORM': norm == expected_norm,
        }
        saved['checks'] = checks
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.output / f'step{step}-rank{rank}.pt')
        print('CP_MODEL_ORDERED_CHECKS', rank, checks, flush=True)
        print(
            'CP_MODEL_LOCAL_GRAD_DIFFERENCES',
            rank,
            compare(saved['local_gradients'], local_references[rank]),
            flush=True,
        )
        if args.mutation:
            rejection = mutation_record(args.mutation, rank, checks)
            (args.output / f'mutation-rank{rank}.json').write_text(
                json.dumps(rejection, indent=2)
            )
            print(
                'CP_MODEL_MUTATION_NAMED_REJECTION', json.dumps(rejection), flush=True
            )
            # Persist both ranks' evidence before either emits its real assertion.
            dist.barrier()
            assert rejection['detected'], ('CP_MODEL_MUTATION_INVALID', rejection)
        for tag, passed in checks.items():
            assert passed, (tag, rank, step)
        assert args.mutation is None, 'CP_MODEL_MUTATION_SURVIVED'
        records.append(saved)
        if step == 1:
            runtime.save_checkpoint(
                handle, str(args.output / 'checkpoint'), step=2, save_rng=False
            )
    restored = runtime.load_checkpoint(
        handle, str(args.output / 'checkpoint'), load_rng=False
    )
    checkpoint_checks = {
        'CP_MODEL_CHECKPOINT_STEP': restored == 2,
        'CP_MODEL_CHECKPOINT_RESTORE': not compare(
            snapshot(model), records[1]['state']
        ),
    }
    continued = train_actual(runtime, handle, model, batch_for_step(2), rank)
    for field in ('state', 'gradients', 'local_gradients', 'outputs'):
        checkpoint_checks['CP_MODEL_CHECKPOINT_CONTINUATION_' + field.upper()] = (
            not compare(continued[field], records[2][field])
        )
    checkpoint_checks['CP_MODEL_CHECKPOINT_NORM'] = (
        continued['norm'] == records[2]['norm']
    )
    checkpoint_checks['CP_MODEL_CHECKPOINT_SUCCESS'] = continued['success']
    continued['checks'] = checkpoint_checks
    torch.save(continued, args.output / f'continued-rank{rank}.pt')
    print('CP_MODEL_CHECKPOINT_CHECKS', rank, checkpoint_checks, flush=True)
    for tag, passed in checkpoint_checks.items():
        assert passed, (tag, rank)
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

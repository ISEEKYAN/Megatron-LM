"""Independent virtual CP models versus actual CP2, including native optimizer."""

import argparse
import os
from copy import copy
from pathlib import Path

import torch
import torch.distributed as dist
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


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
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
    for step in range(1):
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
        saved = {'reference_trace': reference_trace, 'tested_trace': {}}
        hooks = []
        for name, child in model.named_modules():
            if name and any(
                name.endswith(part)
                for part in ('linear_attn', 'self_attn', 'ple', 'mlp')
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
        runtime.forward_backward(handle, [batch], objective, num_microbatches=1)
        actual_gradients = reduced_gradients(handle._model, model)
        success, norm, _ = runtime.optimizer_step(handle)
        for hook in hooks:
            hook.remove()
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
        for tag, passed in checks.items():
            assert passed, (tag, rank, step)
    from megatron.core import parallel_state as mpu

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

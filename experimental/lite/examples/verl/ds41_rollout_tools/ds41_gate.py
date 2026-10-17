#!/usr/bin/env python3
"""DS4.1 training gate driver: one fixed training step, receipts for bitwise comparison.

MLite only (no verl, Ray or vLLM engine). Builds the recipe's actor (quantized
W4A8 experts, Muon with segmented host state, frozen Engram) on the requested
partition and runs one forward_backward over a fixed global batch: every record
of --records is one sequence, data-parallel ranks take contiguous blocks, one
sequence per micro-batch (the recipe's micro batch 1 per GPU).

Receipts (rank 0 writes):
  logprobs.jsonl  per global sequence: the token log-probabilities (FP32 values)
  grads.json      per parameter (global name): bit fingerprint, L2 norm and a
                  fixed signed-projection sketch, after the dense SUM over DP
                  (experts are reported by their owner rank)
  record.json     configuration, memory and timings

Gradients are read after the runtime's finalize_grads, i.e. exactly what the
optimizer consumes (dense averaged over DP x CP, experts summed over expert-DP
and scaled the same way); replicas must agree bitwise.

--save-grads DIR writes every parameter's FP32 gradient once (by its owner
rank) under DIR, after checking the free space; --compare-grads DIR makes the
owner of each parameter stream the saved gradient of another run back and
record its error: max |a - b| and that over max |b| (the test_cp_ep_parity
measures). Both are meant for node-local storage inside one job.

Comparisons (gate_compare.py): same configuration in fresh processes or
--repeats inside one process must be bitwise; a partition change that keeps
the DP size must be bitwise; one that changes it compares log-probs bitwise,
gradients element-wise within tolerance (--compare-grads), and each side must
be bitwise run to run.

  torchrun --nproc_per_node 4 ds41_gate.py --model M --records R --out O --ep 4 --path deployment
"""
import argparse
import json
import os
import time
from pathlib import Path

GiB = 2**30
SKETCH = 16


def parse():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--model', required=True)
    ap.add_argument(
        '--records', required=True, help='fixed sequences (common.py records)'
    )
    ap.add_argument('--out', required=True)
    ap.add_argument(
        '--sequences', type=int, default=None, help='global batch (default: all)'
    )
    ap.add_argument(
        '--ep', type=int, default=None, help='expert parallel (default WORLD_SIZE)'
    )
    ap.add_argument('--cp', type=int, default=1)
    ap.add_argument('--pp', type=int, default=1)
    ap.add_argument('--pipeline-split-layer', type=int, default=None)
    ap.add_argument('--path', choices=('deployment', 'standard'), default='deployment')
    ap.add_argument('--recompute', choices=('none', 'full'), default='none')
    ap.add_argument('--host-engram', action='store_true')
    ap.add_argument('--repeats', type=int, default=1, help='steps inside this process')
    ap.add_argument(
        '--save-grads', default=None, help='write FP32 gradients (first step) here'
    )
    ap.add_argument(
        '--compare-grads',
        default=None,
        help='element-wise error of the first step against gradients saved here',
    )
    return ap.parse_args()


def impl_cfg(args, token_map):
    cfg = {
        'use_thd': True,
        'runtime_plugins': {},
        'qat': {'enabled': False, 'format': 'mxfp4'},
        'quantized': True,
        'w4a8_experts': True,
        'deployment_math': args.path == 'deployment',
        'trainable_engram': False,
        'use_deepep': False,
        'dtype': 'bfloat16',
        'recompute': None if args.recompute == 'none' else 'full',
        'token_map': token_map,
        'cross_entropy_fusion': False,
        'mtp_enable': False,
        'mtp_enable_train': False,
        'mtp_detach_encoder': False,
        'mtp_loss_scaling_factor': 0.1,
        'deterministic': True,
        'optimizer': 'muon',
        'optimizer_config': {
            'lr': 1e-06,
            'ns_steps': 2,
            'coefficient_type': 'quintic',
            'segmented_host': True,
        },
    }
    if args.host_engram:
        cfg['host_engram'] = True
    if args.pp > 1:
        cfg['pipeline_split_layer'] = args.pipeline_split_layer
    return cfg


def _words(tensor):
    import torch

    flat = tensor.detach().contiguous().reshape(-1)
    if flat.numel() * flat.element_size() % 4 == 0:
        return flat.view(torch.int32)
    return flat.view(torch.uint8)


def fingerprint(tensor, chunk=1 << 26):
    """Two 64-bit sums over the raw bits, on the device; chunking cannot change them."""
    import torch

    words = _words(tensor)
    plain = weighted = 0
    for start in range(0, words.numel(), chunk):
        part = words[start : start + chunk].to(torch.int64)
        index = torch.arange(start, start + part.numel(), device=part.device)
        plain = (plain + int(part.sum())) % 2**64
        weighted = (
            weighted + int((part * ((index * 2654435761) % 2147483647 + 1)).sum())
        ) % 2**64
    return [plain, weighted]


def sketch(tensor, chunk=1 << 24):
    """Fixed signed projections of the FP64 values (index-hashed signs), for tolerance checks."""
    import torch

    flat = tensor.detach().reshape(-1)
    out = torch.zeros(SKETCH, dtype=torch.float64, device=flat.device)
    for start in range(0, flat.numel(), chunk):
        part = flat[start : start + chunk].double()
        index = torch.arange(start, start + part.numel(), device=flat.device)
        for j in range(SKETCH):
            sign = (((index * (2 * j + 3) * 2654435761 + j) >> 7) & 1) * 2 - 1
            out[j] += (part * sign).sum()
    return [float(v) for v in out]


def gpu_memory(device):
    import torch

    return dict(
        allocated=torch.cuda.memory_allocated(device) / GiB,
        peak=torch.cuda.max_memory_allocated(device) / GiB,
    )


def _grad_path(directory, name):
    return Path(directory) / f'{name}.pt'


def check_space(directory, handle, owner):
    """Fail before writing when the node-local disk cannot hold every owner's gradients."""
    import shutil

    import torch
    import torch.distributed as dist

    Path(directory).mkdir(parents=True, exist_ok=True)
    mine = sum(
        p.numel() * 4
        for chunk in handle._extras.get('model_chunks', [handle._model])
        for name, p in chunk.named_parameters()
        if p.requires_grad and owner(name)
    )
    total = torch.tensor(mine, dtype=torch.int64, device='cuda')
    dist.all_reduce(total)  # one node: every rank writes to the same disk
    free = shutil.disk_usage(directory).free
    if free < int(total.item() * 1.05):
        raise SystemExit(
            f'DS41_GATE_SPACE: {total.item() / GiB:.1f} GiB of gradients, '
            f'{free / GiB:.1f} GiB free in {directory}'
        )


def save_grads(directory, owned):
    import torch

    for name, grad in owned:
        torch.save(grad.cpu(), _grad_path(directory, name))


def versus(directory, name, grad):
    """test_cp_ep_parity's per-parameter error: max |a - b| and over max |b|."""
    import torch

    path = _grad_path(directory, name)
    if not path.exists():
        return dict(missing=True)
    reference = torch.load(path, map_location=grad.device)
    if reference.shape != grad.shape:
        return dict(shape=[list(reference.shape), list(grad.shape)])
    error = float((grad - reference).abs().max()) if grad.numel() else 0.0
    scale = float(reference.abs().max()) if grad.numel() else 0.0
    del reference
    return dict(abs=error, rel=error / (scale or 1.0), max_reference=scale)


def main():
    args = parse()
    os.environ.setdefault('VLLM_BATCH_INVARIANT', '1')
    import torch
    import torch.distributed as dist
    from verl.workers.engine.utils import enable_full_determinism

    # As MegatronLiteEngine.initialize with full_determinism (actor and ref).
    enable_full_determinism(seed=42)
    from common import load_records
    from megatron.lite.runtime import create_runtime
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.contracts import LossContext, PackedBatch
    from megatron.lite.runtime.contracts.config import ParallelConfig, RuntimeConfig

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    token_map = json.loads((Path(args.model) / 'token-map.json').read_text())
    backend = MegatronLiteConfig(
        model_name='deepseek_v41',
        impl='lite',
        hf_path=args.model,
        parallel=ParallelConfig(
            tp=1, etp=1, ep=args.ep or world, pp=args.pp, vpp=1, cp=args.cp
        ),
        attention_backend_override=None,
        router_aux_loss_coef=0.0,
        load_hf_weights=True,
        impl_cfg=impl_cfg(args, token_map),
    )
    t0 = time.time()
    rt = create_runtime(
        RuntimeConfig(backend='mlite', hf_path=args.model, backend_cfg=backend)
    )
    handle = rt.build_model()
    ps = handle._parallel_state
    rank = dist.get_rank()
    device = torch.device('cuda', torch.cuda.current_device())
    if args.cp > 1:
        # CP ranks share a sequence and slice it; that assignment is not built yet.
        raise NotImplementedError(
            'ds41_gate.py does not assign CP-sliced sequences yet'
        )
    dp_group = ps.dp_group
    if dp_group is None:
        dp_size, dp_rank = 1, 0
    else:
        dp_size, dp_rank = dist.get_world_size(dp_group), dist.get_rank(dp_group)
    ep_dp_rank = 0 if ps.ep_dp_group is None else dist.get_rank(ps.ep_dp_group)

    def owner(name):
        # One writer/comparer per parameter: expert-DP rank 0 for experts,
        # DP rank 0 (of each pipeline stage) for dense parameters.
        return (ep_dp_rank if '.ffn.experts.' in name else dp_rank) == 0

    records = load_records(args.records)[: args.sequences]
    if len(records) % dp_size:
        raise ValueError(f'{len(records)} sequences do not split over DP {dp_size}')
    per_rank = len(records) // dp_size
    mine = list(range(dp_rank * per_rank, (dp_rank + 1) * per_rank))
    record = dict(
        args=vars(args),
        world=world,
        dp_size=dp_size,
        ep_size=ps.ep_size,
        pp_size=ps.pp_size,
        cp_size=ps.cp_size,
        sequences=len(records),
        build_s=time.time() - t0,
    )

    def batch_of(index):
        ids = torch.tensor(
            records[index]['prompt_ids'] + records[index]['response_ids'], device=device
        )
        return (
            PackedBatch(
                input_ids=ids,
                labels=ids,
                loss_mask=torch.ones(ids.numel(), device=device),
                seq_lens=torch.tensor([ids.numel()], dtype=torch.int64, device=device),
            ),
            LossContext(temperature=1.0, return_log_probs=True, loss_scale=1.0),
        )

    for repeat in range(args.repeats):
        rt.zero_grad(handle)
        torch.cuda.reset_peak_memory_stats(device)
        collected = []

        def loss_fn(raw_output, runtime_batch, loss_context):
            log_probs = raw_output['log_probs']
            collected.append(log_probs.detach().float().cpu())
            mask = runtime_batch.loss_mask
            return -(log_probs.float() * mask).sum() / mask.sum(), {}

        t1 = time.time()
        with rt.train_mode(handle):
            rt.forward_backward(
                handle,
                iter([batch_of(i) for i in mine]),
                loss_fn=loss_fn,
                num_microbatches=len(mine),
                forward_only=False,
            )
        step_s = time.time() - t1
        # Log-probs by global sequence index (last PP stage holds them).
        local = {index: lp.tolist() for index, lp in zip(mine, collected)}
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        logprobs = {}
        for part in gathered:
            logprobs.update(part or {})
        grads, owned = {}, []
        if repeat == 0 and args.save_grads:
            check_space(args.save_grads, handle, owner)
        for chunk in handle._extras.get('model_chunks', [handle._model]):
            for name, parameter in chunk.named_parameters():
                grad = getattr(parameter, 'main_grad', None)
                grad = parameter.grad if grad is None else grad
                if grad is None:
                    continue
                # finalize_grads already synchronized replicas; take it as is.
                grad = grad.detach().float()
                grads[name] = dict(
                    fingerprint=fingerprint(grad),
                    norm=float(grad.double().norm()),
                    sketch=sketch(grad),
                )
                if repeat == 0 and owner(name):
                    if args.save_grads:
                        owned.append((name, grad))
                    if args.compare_grads:
                        grads[name]['versus'] = versus(args.compare_grads, name, grad)
        if owned:
            save_grads(args.save_grads, owned)
            owned.clear()
        everyone = [None] * world
        dist.all_gather_object(everyone, grads)
        merged, conflicts = {}, []
        for part in everyone:
            for name, value in part.items():
                if (
                    name in merged
                    and merged[name]['fingerprint'] != value['fingerprint']
                ):
                    conflicts.append(name)  # replicas must agree after the SUM
                merged.setdefault(name, value)
        suffix = '' if repeat == 0 else f'-repeat{repeat}'
        if rank == 0:
            with (out / f'logprobs{suffix}.jsonl').open('w') as f:
                for index in sorted(logprobs):
                    f.write(
                        json.dumps(dict(sequence=index, logprobs=logprobs[index]))
                        + '\n'
                    )
            (out / f'grads{suffix}.json').write_text(json.dumps(merged))
        record[f'step{suffix or "-0"}'] = dict(
            step_s=step_s,
            memory=gpu_memory(device),
            parameters=len(merged),
            replica_conflicts=conflicts[:10],
        )
    if rank == 0:
        (out / 'record.json').write_text(json.dumps(record, indent=1))
    print('DS41_GATE_DONE', rank, json.dumps(record)[:1500], flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

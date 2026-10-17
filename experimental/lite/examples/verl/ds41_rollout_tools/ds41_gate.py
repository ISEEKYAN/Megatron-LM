#!/usr/bin/env python3
"""DS4.1 training gate driver: one fixed training step, log-probabilities for bitwise comparison.

MLite only (no verl, Ray or vLLM engine). Builds the recipe's actor (quantized
W4A8 experts, Muon with segmented host state, frozen Engram) on the requested
partition and runs one forward_backward over a fixed global batch: every record
of --records is one sequence, data-parallel ranks take contiguous blocks, one
sequence per micro-batch (the recipe's micro batch 1 per GPU). Under CP the CP
ranks of a DP rank score the same sequences, each its own rows of the CP-A
whole-pack split (contiguous, ceil(T / cp) rows per rank, tail padding).

Receipts (rank 0 writes):
  logprobs.jsonl  per global sequence: the token log-probabilities (FP32 values)
  record.json     configuration, memory, timings and the gradient check: every
                  finalized gradient (what the optimizer consumes) finite

Judgement (gate_compare.py): forward only -- log-probabilities bitwise, for
the same configuration and across parallel partitions alike; gradients only
have to be finite.

  torchrun --nproc_per_node 4 ds41_gate.py --model M --records R --out O --ep 4 --path deployment
"""
import argparse
import json
import os
import time
from pathlib import Path

GiB = 2**30


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


def gpu_memory(device):
    import torch

    return dict(
        allocated=torch.cuda.memory_allocated(device) / GiB,
        peak=torch.cuda.max_memory_allocated(device) / GiB,
    )


def cp_rows(total, cp_rank, cp_size, local_length):
    """Global row of each of this rank's ``local_length`` rows (-1 for padding).

    CP-A layout contract (PackedCPGeometry): the whole pack is split
    contiguously, rank ``r`` owning rows ``[r * width, (r + 1) * width)`` with
    ``width = ceil(T / cp)``; only the tail is padded. A runtime may return the
    padded width or only the real rows; both start at ``r * width``.
    """
    import torch

    width = -(-total // cp_size)
    start = cp_rank * width
    real = max(min(total - start, width), 0)
    if local_length not in (real, width):
        raise RuntimeError(
            f'CP rank {cp_rank}/{cp_size} returned {local_length} rows of a '
            f'{total}-token sequence (expected {real} or {width})'
        )
    rows = torch.arange(start, start + local_length)
    return torch.where(rows < total, rows, torch.full_like(rows, -1))


def cp_loss(log_probs, rows, total, cp_size):
    """Mean negative log-probability of the sequence, this rank's share.

    The runtime averages gradients over DP x CP, so each CP rank's partial sum
    over its own rows is scaled by the CP size (a power of two, exact); for CP1
    this is the plain mean.
    """
    import torch

    mask = (rows >= 0).to(device=log_probs.device, dtype=torch.float32)
    return -(log_probs.float() * mask).sum() / float(total) * cp_size


def fragment(rows, log_probs):
    keep = rows >= 0
    return dict(rows=rows[keep].tolist(), logprobs=log_probs[keep].tolist())


def merge_fragments(parts):
    """Per global sequence, the CP fragments back in row order; every row once."""
    pieces = {}
    for part in parts:
        for index, piece in (part or {}).items():
            pieces.setdefault(index, {})
            for row, value in zip(piece['rows'], piece['logprobs']):
                if pieces[index].setdefault(row, value) != value:
                    raise RuntimeError(f'sequence {index}: replicas disagree at {row}')
    merged = {}
    for index, rows in pieces.items():
        if sorted(rows) != list(range(len(rows))):
            raise RuntimeError(f'sequence {index}: rows missing after the CP merge')
        merged[index] = [rows[r] for r in range(len(rows))]
    return merged


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
    dp_group = ps.dp_group
    if dp_group is None:
        dp_size, dp_rank = 1, 0
    else:
        dp_size, dp_rank = dist.get_world_size(dp_group), dist.get_rank(dp_group)
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
            log_probs = raw_output['log_probs'].reshape(-1)
            total = int(runtime_batch.seq_lens.sum())
            rows = cp_rows(total, ps.cp_rank, ps.cp_size, log_probs.numel())
            collected.append((rows, log_probs.detach().float().cpu()))
            return cp_loss(log_probs, rows, total, ps.cp_size), {}

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
        local = {
            index: fragment(rows, lp) for index, (rows, lp) in zip(mine, collected)
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        logprobs = merge_fragments(gathered)
        # Gradients are not compared, only checked: every finalized gradient
        # (what the optimizer consumes) must be finite.
        nonfinite, checked = [], 0
        for chunk in handle._extras.get('model_chunks', [handle._model]):
            for name, parameter in chunk.named_parameters():
                grad = getattr(parameter, 'main_grad', None)
                grad = parameter.grad if grad is None else grad
                if grad is None:
                    continue
                checked += 1
                if not bool(torch.isfinite(grad).all()):
                    nonfinite.append(name)
        everyone = [None] * world
        dist.all_gather_object(everyone, (checked, nonfinite))
        suffix = '' if repeat == 0 else f'-repeat{repeat}'
        if rank == 0:
            with (out / f'logprobs{suffix}.jsonl').open('w') as f:
                for index in sorted(logprobs):
                    f.write(
                        json.dumps(dict(sequence=index, logprobs=logprobs[index]))
                        + '\n'
                    )
        record[f'step{suffix or "-0"}'] = dict(
            step_s=step_s,
            memory=gpu_memory(device),
            gradients_checked=sum(count for count, _ in everyone),
            gradients_nonfinite=sorted({n for _, names in everyone for n in names})[
                :20
            ],
        )
    if rank == 0:
        (out / 'record.json').write_text(json.dumps(record, indent=1))
    print('DS41_GATE_DONE', rank, json.dumps(record)[:1500], flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

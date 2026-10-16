# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real 4L release prefix on four GB200s: deployment log-probs under CP x EP.

Opt-in (DS41_GPU_PARITY_PREFIX points at a prepared 4L release prefix). One
spawn per partition, same packed batch (its second document has no rows on CP
rank 0), deployment math with native kernels. Per partition:

* ``full``: forward + backward; ``full_again``: identical repeat (must be
  bitwise, log-probs and every gradient);
* ``recompute``: recompute='full' must keep log-probs and gradients bitwise
  (backward-time recomputation launches vLLM kernels on autograd's stream);
* EP2 x CP2 and EP4 x CP2: every CP rank's log-probs equal its CP1 rows.

Built like the reference engine (BF16 visible weights, the deployment-math
values the actor also sees); routed experts are frozen to fit EP2 on 4 GPUs.
"""

import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [pytest.mark.optional, pytest.mark.gpus(4, min_architecture='blackwell')]
PREFIX = os.environ.get('DS41_GPU_PARITY_PREFIX')
SEQ_LENS = (2500, 1596)
PARTITIONS = tuple(
    tuple(int(v) for v in item.split('x'))
    for item in os.environ.get('DS41_GPU_PARITY_PARTITIONS', '4x1,2x2,4x2').split(',')
)


def _batch():
    total = sum(SEQ_LENS)
    ids = torch.randint(
        1, 129280, (total,), generator=torch.Generator().manual_seed(17)
    ).cuda()
    return SimpleNamespace(
        input_ids=ids,
        labels=ids,
        loss_mask=None,
        seq_lens=torch.tensor(SEQ_LENS),
        cu_seqlens=torch.tensor([0, SEQ_LENS[0], total], dtype=torch.int32).cuda(),
        position_ids=None,
        total_tokens=total,
        extras={},
        routed_experts=None,
        r3_replay_mask=None,
    )


def _step(model, batch, denominator):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context

    model.zero_grad(set_to_none=True)
    with use_loss_context(LossContext(normalization_denominator=denominator)):
        out = protocol._forward_step(model, batch)
    out['loss'].backward()
    torch.cuda.synchronize()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return out['log_probs'].detach().cpu().clone(), grads


def _unequal(left, right):
    if left.keys() != right.keys() or not left:
        return ['<grad keys differ>']
    return [n for n in left if not torch.equal(left[n], right[n])]


def _worker(rank, directory, ep, cp):
    if cp > 1:
        # Report the failing kernel itself rather than a later synchronizing op.
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # enable_full_determinism (below) also sets these; NCCL/cuBLAS read them at
    # initialization, so they must be in place before the process group exists.
    os.environ.update(
        CUBLAS_WORKSPACE_CONFIG=':16:8',
        NCCL_DETERMINISTIC='1',
        NCCL_ALGO='Ring',
        NCCL_PROTO='Simple',
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        init_method=f'file://{directory}/store-{ep}x{cp}',
        rank=rank,
        world_size=4,
        # Fail fast instead of hanging if a peer rank errors in a collective.
        timeout=timedelta(seconds=120),
    )
    from verl_mlite.compat import apply_runtime_patches

    apply_runtime_patches()
    # Same determinism setup as the engine with full_determinism=True
    # (verl_mlite/engine/mlite_engine.py), so repeats are judged like production.
    from verl.workers.engine.utils import enable_full_determinism

    enable_full_determinism(seed=42)
    from megatron.lite.model.deepseek_v41.lite import checkpoint, protocol
    from megatron.lite.model.deepseek_v41.lite.model import (
        owns_attention_state,
        recompute_ffn,
    )
    from megatron.lite.runtime.contracts import ParallelConfig

    path = Path(directory) / f'{ep}x{cp}-{rank}.pt'
    saved = dict(errors={}, log_probs={})
    try:
        token_map = json.loads((Path(PREFIX) / 'token-map.json').read_text())
        impl = protocol.ImplConfig(
            device='cuda',
            dtype=torch.bfloat16,
            quantized=True,
            w4a8_experts=True,
            deployment_math=True,
            token_map=token_map,
            trainable_engram=False,
            parallel=ParallelConfig(ep=ep, cp=cp),
        )
        model = protocol.build_model(
            protocol.build_model_config(PREFIX), impl_cfg=impl
        ).chunks[0]
        checkpoint.load_model(model, PREFIX)
        for name, parameter in model.named_parameters():
            if '.ffn.experts.' in name:
                parameter.requires_grad_(False)
        model.train()
        batch = _batch()
        denominator = float(protocol._cp_targets(batch, None)[2])
        runs = {}
        for name in ('full', 'full_again', 'recompute', 'recompute_again'):
            saved['step'] = name
            if name == 'recompute' and model.recompute != 'full':
                model.recompute = 'full'
                for layer in model.layers:
                    if layer is not None and owns_attention_state(layer):
                        recompute_ffn(layer.ffn)
            runs[name] = _step(model, batch, denominator)
            saved['log_probs'][name] = runs[name][0]
            saved['max_memory_gib'] = torch.cuda.max_memory_allocated() / 2**30
            torch.save(saved, path)
        saved['repeat_unequal'] = _unequal(runs['full_again'][1], runs['full'][1])
        saved['recompute_unequal'] = _unequal(runs['recompute'][1], runs['full'][1])
        # Resume and repeat checks rely on recompute being run-to-run bitwise.
        saved['recompute_repeat_unequal'] = _unequal(
            runs['recompute_again'][1], runs['recompute'][1]
        )
        saved['recompute_max_rel_grad'] = max(
            float((runs['recompute'][1][n] - runs['full'][1][n]).abs().max())
            / (float(runs['full'][1][n].abs().max()) or 1.0)
            for n in runs['full'][1]
        )
        torch.save(saved, path)
    except Exception:
        # Persist the traceback, then fail so every rank stops together.
        saved['errors'][saved.get('step', 'setup')] = traceback.format_exc()[-6000:]
        torch.save(saved, path)
        raise
    dist.barrier()
    dist.destroy_process_group()


def _run(directory, ep, cp):
    try:
        mp.start_processes(
            _worker,
            args=(str(directory), ep, cp),
            nprocs=4,
            join=True,
            start_method='spawn',
        )
        crashed = None
    except Exception as error:  # noqa: BLE001 - keep later partitions running
        crashed = f'{type(error).__name__}: {error}'[-2000:]
    records = []
    for rank in range(4):
        path = directory / f'{ep}x{cp}-{rank}.pt'
        record = (
            torch.load(path, weights_only=False)
            if path.exists()
            else dict(errors={'load': 'no record'}, log_probs={})
        )
        if crashed:
            record['errors']['process'] = crashed
        records.append(record)
    return records


@pytest.mark.skipif(PREFIX is None, reason='set DS41_GPU_PARITY_PREFIX to a 4L prefix')
def test_cp_native_parity(tmp_path):
    """Run every partition first, then judge all checks (one GPU attempt)."""
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    output = Path(os.environ.get('DS41_GPU_PARITY_OUTPUT', tmp_path))
    runs = {(ep, cp): _run(output, ep, cp) for ep, cp in PARTITIONS}
    reference = runs.get((4, 1)) or [
        torch.load(output / f'4x1-{rank}.pt', weights_only=False) for rank in range(1)
    ]
    expected = reference[0]['log_probs'].get('full')
    summary, failed = {}, False
    for (ep, cp), records in runs.items():
        entry = dict(
            errors=[r['errors'] for r in records],
            max_memory_gib=[r.get('max_memory_gib') for r in records],
            repeat_lp_equal=[
                'full_again' in r['log_probs']
                and torch.equal(r['log_probs']['full_again'], r['log_probs']['full'])
                for r in records
            ],
            repeat_unequal_grads=[
                r.get('repeat_unequal', ['<missing>'])[:10] for r in records
            ],
            recompute_lp_equal=[
                'recompute' in r['log_probs']
                and torch.equal(r['log_probs']['recompute'], r['log_probs']['full'])
                for r in records
            ],
            recompute_unequal_grads=[
                r.get('recompute_unequal', ['<missing>'])[:10] for r in records
            ],
            recompute_max_rel_grad=[r.get('recompute_max_rel_grad') for r in records],
            recompute_repeat_lp_equal=[
                'recompute_again' in r['log_probs']
                and torch.equal(
                    r['log_probs']['recompute_again'], r['log_probs']['recompute']
                )
                for r in records
            ],
            recompute_repeat_unequal_grads=[
                r.get('recompute_repeat_unequal', ['<missing>'])[:10] for r in records
            ],
        )
        failed |= any(r['errors'] for r in records)
        failed |= not all(entry['repeat_lp_equal'])
        failed |= any(entry['repeat_unequal_grads'])
        failed |= not all(entry['recompute_lp_equal'])
        failed |= not all(entry['recompute_repeat_lp_equal'])
        failed |= any(entry['recompute_repeat_unequal_grads'])
        failed |= any((m is None or m > 1e-2) for m in entry['recompute_max_rel_grad'])
        if cp > 1:
            equal = []
            for rank, record in enumerate(records):
                local = record['log_probs'].get('full')
                rows = (
                    None
                    if expected is None
                    else ContiguousCPSequence(expected.shape[0], rank % cp, cp).slice(
                        expected[None], seq_dim=1
                    )[0]
                )
                equal.append(
                    local is not None and rows is not None and torch.equal(local, rows)
                )
            entry['log_probs_equal_cp1_rows'] = equal
            failed |= not all(equal)
        summary[f'ep{ep}xcp{cp}'] = entry
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print('B2_GPU_PARITY_SUMMARY', json.dumps(summary, default=str), flush=True)
    assert not failed, summary

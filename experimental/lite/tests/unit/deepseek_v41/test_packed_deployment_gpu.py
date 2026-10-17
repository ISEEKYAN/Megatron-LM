# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed deployment CSA on four GB200s: whole slice equals per document.

Opt-in (DS41_GPU_PARITY_PREFIX points at a prepared 4L release prefix). EP4 at
CP1 and at CP2 on the same four GPUs; the documents (2000, 40, 2061) leave CP
rank 1 without rows of the first two. Each partition runs one forward and
backward on the packed deployment path and one with the per-document path
forced. Checked: log-probs of the two paths are bitwise equal, CP2 log-probs
equal their CP1 rows bitwise, and every gradient is finite.
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
SEQ_LENS = (2000, 40, 2061)
PARTITIONS = ((4, 1), (4, 2))


def _batch():
    total = sum(SEQ_LENS)
    ids = torch.randint(
        1, 129280, (total,), generator=torch.Generator().manual_seed(17)
    ).cuda()
    bounds = [0]
    for n in SEQ_LENS:
        bounds.append(bounds[-1] + n)
    return SimpleNamespace(
        input_ids=ids,
        labels=ids,
        loss_mask=None,
        seq_lens=torch.tensor(SEQ_LENS),
        cu_seqlens=torch.tensor(bounds, dtype=torch.int32).cuda(),
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
    finite = all(
        torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.grad is not None
    )
    return out['log_probs'].detach().cpu().clone(), finite


def _worker(rank, directory, ep, cp):
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
        timeout=timedelta(seconds=120),
    )
    from verl_mlite.compat import apply_runtime_patches

    apply_runtime_patches()
    from verl.workers.engine.utils import enable_full_determinism

    enable_full_determinism(seed=42)
    from megatron.lite.model.deepseek_v41.lite import checkpoint, protocol
    from megatron.lite.primitive.modules.attention import packed_deployment
    from megatron.lite.runtime.contracts import ParallelConfig

    path = Path(directory) / f'{ep}x{cp}-{rank}.pt'
    saved = dict(errors={}, log_probs={}, finite={})
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
        saved['cp_rank'] = model.ps.cp_rank
        batch = _batch()
        denominator = float(protocol._cp_targets(batch, None)[2])
        fits = packed_deployment.fits
        for name in ('packed', 'documents'):
            saved['step'] = name
            packed_deployment.fits = fits if name == 'packed' else lambda *a: False
            saved['log_probs'][name], saved['finite'][name] = _step(
                model, batch, denominator
            )
        packed_deployment.fits = fits
        saved['fallbacks'] = sum(packed_deployment.FALLBACKS.values())
        torch.save(saved, path)
    except Exception:
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
    except Exception as error:  # noqa: BLE001 - judge every partition
        crashed = f'{type(error).__name__}: {error}'[-2000:]
    else:
        crashed = None
    records = []
    for rank in range(4):
        path = directory / f'{ep}x{cp}-{rank}.pt'
        record = (
            torch.load(path, weights_only=False)
            if path.exists()
            else dict(errors={'load': 'no record'}, log_probs={}, finite={})
        )
        if crashed:
            record['errors']['process'] = crashed
        records.append(record)
    return records


@pytest.mark.skipif(PREFIX is None, reason='set DS41_GPU_PARITY_PREFIX to a 4L prefix')
def test_packed_deployment_equals_documents_and_cp1(tmp_path):
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    output = Path(os.environ.get('DS41_GPU_PARITY_OUTPUT', tmp_path))
    runs = {key: _run(output, *key) for key in PARTITIONS}
    summary, failed = {}, False
    for (ep, cp), records in runs.items():
        entry = dict(
            errors=[r['errors'] for r in records],
            packed_equals_documents=[
                torch.equal(r['log_probs']['packed'], r['log_probs']['documents'])
                for r in records
                if {'packed', 'documents'} <= r['log_probs'].keys()
            ],
            finite=[all(r['finite'].values()) for r in records],
            # Only the forced per-document pass may fall back (one per layer).
            fallbacks=[r.get('fallbacks') for r in records],
        )
        failed |= any(r['errors'] for r in records)
        failed |= len(entry['packed_equals_documents']) != 4
        failed |= not all(entry['packed_equals_documents'] + entry['finite'])
        summary[f'ep{ep}xcp{cp}'] = entry
    expected = runs[PARTITIONS[0]][0]['log_probs'].get('packed')
    cp = PARTITIONS[1][1]
    equal = []
    for record in runs[PARTITIONS[1]]:
        local = record['log_probs'].get('packed')
        rows = None
        if expected is not None and record.get('cp_rank') is not None:
            rows = ContiguousCPSequence(expected.shape[0], record['cp_rank'], cp).slice(
                expected[None], seq_dim=1
            )[0]
        equal.append(
            local is not None and rows is not None and torch.equal(local, rows)
        )
    summary['cp2_equals_cp1_rows'] = equal
    failed |= not all(equal)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print('PACKED_DEPLOYMENT_SUMMARY', json.dumps(summary, default=str), flush=True)
    assert not failed, summary

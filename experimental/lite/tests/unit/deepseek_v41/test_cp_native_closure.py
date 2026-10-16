# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The per-document CP path launches only enumerated native entries, never on empty inputs.

CPU: the static scan finds the known entries; the tracker flags empty inputs
and unregistered dispatcher ops. GPU (opt-in, DS41_GPU_PARITY_PREFIX): a packed
batch whose documents give every CP rank the zero-, one- and whole-document
states, run under EP2 x CP2 and its EP4 x CP1 reference.
"""
import json
import os
import sys
import types
from datetime import timedelta
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cp_native_closure import (  # noqa: E402
    EMPTY_SAFE,
    NativeCallTracker,
    cp_document_state,
    scan_native_entries,
    track_cp_documents,
)

PREFIX = os.environ.get('DS41_GPU_PARITY_PREFIX')


def test_scan_finds_the_per_document_native_entries():
    entries = scan_native_entries()
    for known in (
        '_C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert',
        'vllm.v1.attention.ops.flashmla:flash_mla_sparse_fwd',
        'vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant:fused_inv_rope_fp8_quant',
        'vllm.model_executor.kernels.linear.cute_dsl.batch_invariant:bf16_fixed_mm',
        'vllm.model_executor.kernels.batch_invariant:top_k_per_row_prefill',
        'vllm.model_executor.kernels.mhc.tilelang:mhc_pre_delayed_tilelang',
        'megatron.lite.primitive.kernels.sparse_metadata:sparse_metadata',
        'deep_gemm:m_grouped_fp8_fp4_gemm_nt_contiguous',
    ):
        assert known in entries, known
    assert set(EMPTY_SAFE) <= set(entries) | {'vllm::bf16_fixed_mm'}


def test_tracker_flags_empty_inputs_and_unregistered_ops():
    fake = types.ModuleType('b1_fake_native')
    fake.kernel = lambda x: x + 1
    sys.modules['b1_fake_native'] = fake
    library = torch.library.Library('b1closure', 'FRAGMENT')
    library.define('twice(Tensor x) -> Tensor')
    library.impl('twice', lambda x: x * 2, 'CPU')
    tracker = NativeCallTracker(['b1_fake_native:kernel'], device_type='cpu').install()
    try:
        with tracker:
            with tracker.document_state('one'):
                sys.modules['b1_fake_native'].kernel(torch.ones(1, 3))
            with tracker.document_state('zero'):
                sys.modules['b1_fake_native'].kernel(torch.ones(0, 3))
                torch.ops.b1closure.twice(torch.ones(2))
    finally:
        tracker.uninstall()
    report = tracker.report()
    assert report['calls']['b1_fake_native:kernel'] == {'one': 1, 'zero': 1}
    assert report['empty'] == {'b1_fake_native:kernel': {'zero': 1}}
    assert report['unregistered'] == {'b1closure::twice@zero': 1}
    assert len(tracker.violations()) == 2
    assert sys.modules['b1_fake_native'].kernel is fake.kernel  # restored


def test_document_states(v41_core_te):
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    # Width 4 over 8 tokens; documents [0,3) [3,5) [5,8).
    seq = ContiguousCPSequence(8, 0, 2)
    assert [
        cp_document_state(seq.document(b, e)) for b, e in ((0, 3), (3, 5), (5, 8))
    ] == ['whole', 'one', 'zero']
    seq = ContiguousCPSequence(8, 1, 2)
    assert [
        cp_document_state(seq.document(b, e)) for b, e in ((0, 3), (3, 5), (5, 8))
    ] == ['zero', 'one', 'whole']


# -- GPU: every rank sees zero, one and whole documents --------------------
# Width W = 1024 per CP2 rank: CP rank 0 owns [0, 1023) whole, row 1023 of the
# 2-token document, and no row of the last; CP rank 1 the mirror image. Run as
# EP4 x CP2 on four GPUs (EP spans both CP groups); EP2 x CP2 does not fit the
# 4L prefix on one node (7998685 / 7998083 OOM).
SEQ_LENS = (1023, 2, 1023)


def _batch():
    from types import SimpleNamespace

    total = sum(SEQ_LENS)
    ids = torch.randint(
        1, 129280, (total,), generator=torch.Generator().manual_seed(23)
    ).cuda()
    bounds = torch.tensor(
        (0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()), dtype=torch.int32
    )
    return SimpleNamespace(
        input_ids=ids,
        labels=ids,
        loss_mask=None,
        seq_lens=torch.tensor(SEQ_LENS),
        cu_seqlens=bounds.cuda(),
        position_ids=None,
        total_tokens=total,
        extras={},
        routed_experts=None,
        r3_replay_mask=None,
    )


def _worker(rank, directory, ep, cp):
    import torch.distributed as dist

    os.environ.update(
        CUBLAS_WORKSPACE_CONFIG=':16:8',
        NCCL_DETERMINISTIC='1',
        NCCL_ALGO='Ring',
        NCCL_PROTO='Simple',
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        init_method=f'file://{directory}/closure-store-{ep}x{cp}',
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    from verl_mlite.compat import apply_runtime_patches

    apply_runtime_patches()
    from verl.workers.engine.utils import enable_full_determinism

    enable_full_determinism(seed=42)
    from megatron.lite.model.deepseek_v41.lite import checkpoint, protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    path = Path(directory) / f'closure-{ep}x{cp}-{rank}.pt'
    saved = dict(error=None)
    tracker = None
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
        tracker = NativeCallTracker(scan_native_entries()).install()
        # Forward only: the native entries all run in the forward (backward is
        # reference arithmetic), and EP2 x CP2 training of the 4L prefix does
        # not fit with gradients (7998685 OOM). Graph connectivity and
        # gradients are covered by test_cp_native_parity_gpu.
        from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context

        with track_cp_documents(tracker), tracker, torch.no_grad():
            with use_loss_context(LossContext(normalization_denominator=denominator)):
                log_probs = (
                    protocol._forward_step(model, batch)['log_probs']
                    .detach()
                    .cpu()
                    .clone()
                )
        torch.cuda.synchronize()
        torch.ones(1, device='cuda').add_(1)  # a residual CUDA error surfaces here
        saved.update(
            log_probs=log_probs,
            report=tracker.report(),
            violations=tracker.violations(),
        )
    except Exception:
        import traceback

        saved['error'] = traceback.format_exc()[-6000:]
        if tracker is not None:
            saved.update(report=tracker.report(), violations=tracker.violations())
        torch.save(saved, path)
        raise
    finally:
        if tracker is not None:
            tracker.uninstall()
    torch.save(saved, path)
    dist.barrier()
    dist.destroy_process_group()


def _run(directory, ep, cp):
    import torch.multiprocessing as mp

    try:
        mp.start_processes(
            _worker,
            args=(str(directory), ep, cp),
            nprocs=4,
            join=True,
            start_method='spawn',
        )
    except Exception:  # noqa: BLE001 - records carry the per-rank error
        pass
    return [
        (
            torch.load(
                Path(directory) / f'closure-{ep}x{cp}-{r}.pt', weights_only=False
            )
            if (Path(directory) / f'closure-{ep}x{cp}-{r}.pt').exists()
            else dict(error='no record')
        )
        for r in range(4)
    ]


@pytest.mark.gpus(4, min_architecture='blackwell')
@pytest.mark.skipif(PREFIX is None, reason='set DS41_GPU_PARITY_PREFIX to a 4L prefix')
def test_cp_native_entries_closed_over_document_states(tmp_path):
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    output = Path(os.environ.get('DS41_GPU_PARITY_OUTPUT', tmp_path))
    reference = _run(output, 4, 1)
    sharded = _run(output, 4, 2)
    inventory = {}
    for (ep, cp), records in (((4, 1), reference), ((4, 2), sharded)):
        for rank, record in enumerate(records):
            for entry, states in record.get('report', {}).get('calls', {}).items():
                merged = inventory.setdefault(entry, {})
                for state, count in states.items():
                    merged[state] = merged.get(state, 0) + count
    summary = dict(
        seq_lens=SEQ_LENS,
        errors={
            f'{ep}x{cp}-{r}': rec['error']
            for (ep, cp), recs in (((4, 1), reference), ((4, 2), sharded))
            for r, rec in enumerate(recs)
            if rec.get('error')
        },
        violations={
            f'{ep}x{cp}-{r}': rec.get('violations')
            for (ep, cp), recs in (((4, 1), reference), ((4, 2), sharded))
            for r, rec in enumerate(recs)
            if rec.get('violations')
        },
        native_entries=inventory,
        empty_safe=EMPTY_SAFE,
    )
    expected = reference[0].get('log_probs')
    unequal = []
    for rank, record in enumerate(sharded):
        local = record.get('log_probs')
        rows = (
            None
            if expected is None
            else ContiguousCPSequence(expected.shape[0], rank % 2, 2).slice(
                expected[None], seq_dim=1
            )[0]
        )
        if local is None or rows is None or not torch.equal(local, rows):
            unequal.append(rank)
    summary['cp_rows_unequal_cp1'] = unequal
    (output / 'cp-native-entries.json').write_text(
        json.dumps(summary, indent=1, default=str)
    )
    print('B1_CP_NATIVE_CLOSURE', json.dumps(summary, default=str)[:20000])
    assert not summary['errors'], summary['errors']
    assert not summary['violations'], summary['violations']
    # Every document state was exercised on the sharded run.
    states = {s for states in inventory.values() for s in states}
    assert {'zero', 'one', 'whole'} <= states, states
    assert not unequal, unequal

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""W4A8 MoE over real normal DeepEP (EP4) against unfused all-to-all, on GPU.

Same global expert bank and tokens on both transports; uneven token counts per
rank, idle experts, the official route-weight-before-FC2 contract and an
identical repeat (no stale DeepEP handle). Forward and expert/router gradients
are bitwise; dX agrees to BF16 rounding (DeepEP combine owns its sum).
"""
import copy
import json
import os
import runpy
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [pytest.mark.optional, pytest.mark.gpus(4, min_architecture='blackwell')]
EP = 4
HIDDEN = 256


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(HIDDEN, 128, bias=False)
        self.w2 = torch.nn.Linear(128, HIDDEN, bias=False)
        self.w3 = torch.nn.Linear(HIDDEN, 128, bias=False)
        self.swiglu_limit = 10.0


def _worker(rank, store, directory, official):
    try:
        _run(rank, store, directory, official)
    except BaseException:
        Path(directory, f'error-{rank}-{int(official)}.txt').write_text(
            traceback.format_exc()
        )
        raise


def _run(rank, store, directory, official):
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.lite.moe import DeepseekV41MoE, ModalityRouter
    from megatron.lite.primitive.modules.route_preserving_deepep import (
        RoutePreservingDeepEPDispatcher,
    )
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        init_method=f'file://{store}/store',
        rank=rank,
        world_size=EP,
        timeout=timedelta(seconds=300),
    )
    ps = init_parallel(ParallelConfig(ep=EP, etp=1))
    cfg = NS(
        n_routed_experts=16,
        num_experts_per_tok=4,
        hidden_size=HIDDEN,
        norm_topk_prob=True,
        topk_method='noaux_tc',
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    torch.manual_seed(907)
    router = ModalityRouter(cfg, ps)
    # Experts 12..15 (all of rank 3's) never receive tokens.
    router.router.gate.weight.data.zero_()
    router.router.gate.weight.data[:12].normal_(std=0.1)
    bank = [_Expert() for _ in range(16)]
    local = 16 // EP
    owned = [
        e if rank * local <= i < (rank + 1) * local else None
        for i, e in enumerate(bank)
    ]
    models = [
        DeepseekV41MoE(
            copy.deepcopy(router),
            copy.deepcopy(owned),
            ps=ps,
            use_deepep=use_deepep,
            w4a8=True,
            official_expert_contract=official,
        ).cuda()
        for use_deepep in (False, True)
    ]
    assert isinstance(models[1].dispatcher, RoutePreservingDeepEPDispatcher)
    assert models[1].dispatcher.use_deepep
    torch.manual_seed(100 + rank)
    data = torch.randn(1, 37 * (rank + 1), HIDDEN).bfloat16().cuda()
    incoming = torch.randn_like(data)
    report = {}
    results = []
    for model in models:
        runs = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            x = data.clone().requires_grad_()
            out = model(x)
            out.backward(incoming)
            grads = {
                f'{i}.{n}': getattr(e, n).weight.grad.clone()
                for i, e in enumerate(model.experts)
                if e is not None
                for n in ('w1', 'w2', 'w3')
                if getattr(e, n).weight.grad is not None
            }
            grads['gate'] = model.gate.router.gate.weight.grad.clone()
            runs.append((out.detach(), x.grad, grads))
        report[f'repeat_bitwise_{len(results)}'] = torch.equal(
            runs[0][0], runs[1][0]
        ) and all(torch.equal(runs[0][2][k], runs[1][2][k]) for k in runs[0][2])
        results.append(runs[0])
    (out0, dx0, g0), (out1, dx1, g1) = results
    report['forward_bitwise'] = torch.equal(out0, out1)
    report['dx_max_abs'] = (dx0.float() - dx1.float()).abs().max().item()
    report['grad_keys_equal'] = sorted(g0) == sorted(g1)
    report['grads_bitwise'] = {k: torch.equal(g0[k], g1[k]) for k in g0 if k in g1}
    Path(directory, f'report-{rank}-{int(official)}.json').write_text(
        json.dumps(report, indent=1)
    )
    assert report['forward_bitwise'], report
    assert report['repeat_bitwise_0'] and report['repeat_bitwise_1'], report
    torch.testing.assert_close(dx0, dx1, rtol=2**-7, atol=2**-10)
    assert report['grad_keys_equal'], report
    for key in g0:
        torch.testing.assert_close(g0[key], g1[key], rtol=0, atol=0, msg=key)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize('official', [False, True])
def test_w4a8_deepep_ep4_matches_alltoall(v41_core_te, tmp_path, monkeypatch, official):
    pytest.importorskip('deep_ep')
    monkeypatch.setenv('MEGATRON_LITE_MOE_PERMUTE_FUSION', '0')
    out = Path(os.environ.get('B4_G1_REPORT_DIR', tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    mp.start_processes(
        _worker,
        args=(str(tmp_path), str(out), official),
        nprocs=EP,
        join=True,
        start_method='spawn',
    )

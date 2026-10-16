# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""W4A8 MoE over real normal DeepEP (EP4) against unfused all-to-all, on GPU.

* ``moe``: same global expert bank and tokens on both transports; uneven token
  counts per rank, idle experts and the official route-weight-before-FC2
  contract. Forward, router and expert gradients are bitwise across transports;
  an identical repeat is bitwise on each transport, dX included.
* ``transport``: identity experts and known per-route gradients isolate the dX
  reduction. Each transport repeats bitwise and stays within two BF16 ulps of
  the FP64 sum.

Everything runs in the recipe's deterministic mode; the all-to-all repeat
outside it is only recorded.
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
EXPERTS = 16
TOPK = 4


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(HIDDEN, 128, bias=False)
        self.w2 = torch.nn.Linear(128, HIDDEN, bias=False)
        self.w3 = torch.nn.Linear(HIDDEN, 128, bias=False)
        self.swiglu_limit = 10.0


def _production_determinism():
    # What the recipe's full_determinism sets (verl enable_full_determinism).
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker(rank, store, directory, mode):
    try:
        _setup(rank, store)
        _production_determinism()
        report = _MODES[mode](rank)
        Path(directory, f'report-{mode}-{rank}.json').write_text(
            json.dumps(report, indent=1)
        )
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        Path(directory, f'error-{mode}-{rank}.txt').write_text(traceback.format_exc())
        raise


def _setup(rank, store):
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl',
        init_method=f'file://{store}/store',
        rank=rank,
        world_size=EP,
        timeout=timedelta(seconds=300),
    )


def _ps():
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    return init_parallel(ParallelConfig(ep=EP, etp=1))


def _moe(rank, official):
    from megatron.lite.model.deepseek_v41.lite.moe import DeepseekV41MoE, ModalityRouter
    from megatron.lite.primitive.modules.route_preserving_deepep import (
        RoutePreservingDeepEPDispatcher,
    )

    ps = _ps()
    cfg = NS(
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOPK,
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
    bank = [_Expert() for _ in range(EXPERTS)]
    local = EXPERTS // EP
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
    report, results = {}, []
    for name, model in zip(('alltoall', 'deepep'), models):
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
        (o0, x0, g0), (o1, x1, g1) = runs
        report[f'{name}_repeat_forward_bitwise'] = torch.equal(o0, o1)
        report[f'{name}_repeat_dx_bitwise'] = torch.equal(x0, x1)
        report[f'{name}_repeat_grads_bitwise'] = all(
            torch.equal(g0[k], g1[k]) for k in g0
        )
        results.append(runs[0])
    (out0, dx0, g0), (out1, dx1, g1) = results
    report['forward_bitwise'] = torch.equal(out0, out1)
    report['dx_max_abs_across_transports'] = (
        (dx0.float() - dx1.float()).abs().max().item()
    )
    report['grad_keys_equal'] = sorted(g0) == sorted(g1)
    report['grads_bitwise'] = all(torch.equal(g0[k], g1[k]) for k in g0 if k in g1)
    return report


def _transport(rank):
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
    from megatron.lite.primitive.modules.route_preserving_deepep import (
        RoutePreservingDeepEPDispatcher,
    )

    ps = _ps()
    tokens = 300 + 97 * rank
    generator = torch.Generator().manual_seed(500 + rank)
    # Distinct experts per token among 0..11; rank 3 owns no routed expert.
    ids = torch.stack(
        [torch.randperm(12, generator=generator)[:TOPK] for _ in range(tokens)]
    )
    ids = ids.cuda()
    weights = torch.rand(tokens, TOPK, generator=generator).cuda()
    hidden = (torch.randn(tokens, HIDDEN, generator=generator) * 0.5).bfloat16().cuda()
    route_grads = torch.randn(tokens, TOPK, HIDDEN, generator=generator) * 0.08
    route_grads = route_grads.bfloat16().cuda()
    exact = route_grads.double().sum(1)
    # A sum of TOPK BF16 terms rounds at most TOPK - 1 times, each within half
    # an ulp of the running |sum| <= sum |terms|: two ulps of that bound.
    scale = route_grads.double().abs().sum(1)
    bound = 2.0 * torch.exp2(torch.floor(torch.log2(scale.clamp_min(2.0**-100))) - 7)

    def run(dispatcher):
        x = hidden.clone().requires_grad_()
        rows, _, _ = dispatcher.dispatch(x, weights, ids)
        back, row_token, row_expert = dispatcher.combine_unreduced(rows)
        slot = (ids[row_token] == row_expert.unsqueeze(-1)).int().argmax(-1)
        (back.float() * route_grads[row_token, slot].float()).sum().backward()
        torch.cuda.synchronize()
        return x.grad

    def dispatcher(cls, use_deepep):
        return cls(EXPERTS, HIDDEN, ps, use_deepep=use_deepep)

    report = {}
    paths = {
        'alltoall': lambda: dispatcher(TokenDispatcher, False),
        'deepep': lambda: dispatcher(RoutePreservingDeepEPDispatcher, True),
    }
    assert not paths['alltoall']().use_deepep
    assert paths['deepep']().use_deepep
    first = {}
    for name, make in paths.items():
        d = make()
        a, b = run(d), run(d)
        first[name] = a
        err = (a.double() - exact).abs()
        report[f'{name}_repeat_dx_bitwise'] = torch.equal(a, b)
        report[f'{name}_within_2ulp_fp64'] = bool((err <= bound).all())
        report[f'{name}_max_err_over_bound'] = (err / bound).max().item()
    # Recorded only: outside deterministic mode the all-to-all unpermute VJP
    # (index_add_) uses atomics and need not repeat bitwise.
    torch.use_deterministic_algorithms(False)
    try:
        d = paths['alltoall']()
        report['info_alltoall_default_mode_repeat_dx_bitwise'] = str(
            torch.equal(run(d), run(d))
        )
    finally:
        _production_determinism()
    report['dx_max_abs_across_transports'] = (
        (first['alltoall'].float() - first['deepep'].float()).abs().max().item()
    )
    return report


_MODES = {
    'moe0': lambda rank: _moe(rank, False),
    'moe1': lambda rank: _moe(rank, True),
    'transport': _transport,
}


@pytest.mark.parametrize('mode', sorted(_MODES))
def test_w4a8_deepep_ep4(v41_core_te, tmp_path, monkeypatch, mode):
    pytest.importorskip('deep_ep')
    monkeypatch.setenv('MEGATRON_LITE_MOE_PERMUTE_FUSION', '0')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out = Path(os.environ.get('B4_G1_REPORT_DIR', tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    mp.start_processes(
        _worker,
        args=(str(tmp_path), str(out), mode),
        nprocs=EP,
        join=True,
        start_method='spawn',
    )
    reports = [
        json.loads(Path(out, f'report-{mode}-{rank}.json').read_text())
        for rank in range(EP)
    ]
    for rank, report in enumerate(reports):
        failed = {k: v for k, v in report.items() if isinstance(v, bool) and not v}
        assert not failed, (rank, report)

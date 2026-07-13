# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Primitive-level parity/contract tests for static-capacity MoE dispatch.

These CPU tests qualify the #5258 dispatch *contract* ported into MLite's own
dispatcher primitive (see ``docs/cuda-graph-design.md`` §"How upstream captures
dropless MoE dispatch (#5258)"):

1. ``compute_static_capacity`` derives ``M``/``C`` from static config only.
2. Static-capacity dispatch is numerically equal to the existing dynamic path
   when no expert overflows its budget (EP=1 and EP=2/gloo).
3. Over-budget raises the device overflow flag and ``raise_if_over_budget``
   fails loud — never silent truncation.
4. Dispatch + combine perform zero device->host reads (graph-safe): ``.item`` /
   ``.tolist`` / ``.cpu`` on any tensor inside the captured span raises.

GPU 8-card qualification is out of scope here (TASK-1.21.8).
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch


def _install_te_stub() -> None:
    """Install a minimal Transformer Engine stub so the moe utils import on CPU.

    The static-capacity path only uses the non-fused (argsort) permute/unpermute,
    so the fused kernels are never called; the stub just satisfies module imports.
    Works without pytest monkeypatch so it is reusable inside spawned subprocesses.
    """
    try:
        import transformer_engine.pytorch  # noqa: F401

        return
    except (ModuleNotFoundError, OSError):
        pass

    def unavailable(*args, **kwargs):
        raise RuntimeError("Transformer Engine fused kernel is not installed.")

    root = types.ModuleType("transformer_engine")
    root.__version__ = "0.0.0"
    pytorch = types.ModuleType("transformer_engine.pytorch")
    permutation = types.ModuleType("transformer_engine.pytorch.permutation")
    router = types.ModuleType("transformer_engine.pytorch.router")
    cpp_extensions = types.ModuleType("transformer_engine.pytorch.cpp_extensions")
    module = types.ModuleType("transformer_engine.pytorch.module")
    module_base = types.ModuleType("transformer_engine.pytorch.module.base")

    for name in (
        "moe_permute",
        "moe_permute_and_pad_with_probs",
        "moe_permute_with_probs",
        "moe_unpermute",
    ):
        setattr(permutation, name, unavailable)
    for name in (
        "fused_compute_score_for_moe_aux_loss",
        "fused_moe_aux_loss",
        "fused_topk_with_score_function",
    ):
        setattr(router, name, unavailable)
    cpp_extensions.general_gemm = lambda *a, **k: None
    module_base.get_workspace = lambda: None
    module.base = module_base
    pytorch.permutation = permutation
    pytorch.router = router
    pytorch.cpp_extensions = cpp_extensions
    pytorch.module = module
    root.pytorch = pytorch
    sys.modules.update(
        {
            "transformer_engine": root,
            "transformer_engine.pytorch": pytorch,
            "transformer_engine.pytorch.permutation": permutation,
            "transformer_engine.pytorch.router": router,
            "transformer_engine.pytorch.cpp_extensions": cpp_extensions,
            "transformer_engine.pytorch.module": module,
            "transformer_engine.pytorch.module.base": module_base,
        }
    )


_install_te_stub()

from megatron.lite.primitive.modules.dispatcher import (  # noqa: E402
    StaticCapacityConfig,
    TokenDispatcher,
    compute_static_capacity,
)
from megatron.lite.primitive.parallel.state import ParallelState  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_routing(num_tokens, num_experts, topk, seed):
    """Synthesize a unique-topk router output (scores, indices)."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, generator=gen, dtype=torch.float32)
    topk_logits, topk_indices = torch.topk(logits, topk, dim=1)
    topk_scores = torch.softmax(topk_logits, dim=1)
    return topk_scores, topk_indices


def _apply_experts(dispatched, permuted_probs, chunk_sizes, weights):
    """Emulate grouped-GEMM experts: per-expert linear then scale by probs.

    ``weights[i]`` applies to the ``i``-th contiguous chunk (expert). Probs are
    applied AFTER the linear, matching ``swiglu_with_probs`` so that padding slots
    (prob 0) are numerically inert regardless of their hidden content.
    """
    outs = []
    off = 0
    for size, w in zip(chunk_sizes, weights, strict=True):
        chunk = dispatched[off : off + size]
        outs.append(chunk @ w)
        off += size
    out = torch.cat(outs, dim=0)
    return out * permuted_probs.unsqueeze(-1)


def _ps_single() -> ParallelState:
    return ParallelState(ep_size=1)


# --------------------------------------------------------------------------- #
# 1. Contract math
# --------------------------------------------------------------------------- #
def test_compute_static_capacity_contract():
    # Per-rank budget rounds up to the alignment; capacity covers the routed load.
    cfg = compute_static_capacity(
        max_seqlen_per_dp_cp_rank=1000,
        num_experts=8,
        moe_router_topk=2,
        capacity_factor=1.0,
        token_alignment=8,
    )
    assert cfg.num_tokens == 1000  # already a multiple of 8
    # 1000*2/8 = 250 -> aligned up to 256
    assert cfg.expert_capacity == 256

    # Sequence parallel divides the per-rank token count by TP first.
    cfg_sp = compute_static_capacity(
        max_seqlen_per_dp_cp_rank=1024,
        num_experts=4,
        moe_router_topk=1,
        tensor_model_parallel_size=4,
        sequence_parallel=True,
        capacity_factor=1.5,
        token_alignment=8,
    )
    assert cfg_sp.num_tokens == 256  # 1024/4, aligned
    # ceil(1.5*256*1/4) = 96 -> aligned to 96
    assert cfg_sp.expert_capacity == 96


def test_compute_static_capacity_rejects_bad_input():
    with pytest.raises(ValueError):
        compute_static_capacity(
            max_seqlen_per_dp_cp_rank=0, num_experts=8, moe_router_topk=2
        )


# --------------------------------------------------------------------------- #
# 2a. Parity: EP=1 static vs dynamic-local
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_static_ep1_parity_with_dynamic_local(dtype):
    torch.use_deterministic_algorithms(True, warn_only=True)
    num_tokens, hidden, num_experts, topk = 64, 16, 8, 2
    scores, indices = _make_routing(num_tokens, num_experts, topk, seed=0)

    gen = torch.Generator().manual_seed(1)
    x = torch.randn(num_tokens, hidden, generator=gen, dtype=dtype)
    weights = [
        torch.randn(hidden, hidden, generator=gen, dtype=dtype) for _ in range(num_experts)
    ]

    ps = _ps_single()

    # Dynamic reference.
    dyn = TokenDispatcher(num_experts, hidden, ps, use_deepep=False)
    d_disp, d_tpe, d_probs = dyn.dispatch(x, scores, indices)
    d_out = _apply_experts(d_disp, d_probs, d_tpe.tolist(), weights)
    d_final = dyn.combine(d_out)

    # Static-capacity (budget large enough => no overflow => exact parity).
    cap = int(indices.new_zeros(num_experts).scatter_add_(
        0, indices.reshape(-1), torch.ones(num_tokens * topk, dtype=torch.long)
    ).max()) + 8
    cfg = StaticCapacityConfig(num_tokens=num_tokens, expert_capacity=cap)
    stat = TokenDispatcher(num_experts, hidden, ps, use_deepep=False, static_capacity=cfg)
    s_disp, s_tpe, s_probs = stat.dispatch(x, scores, indices)
    assert s_disp.shape[0] == num_experts * cap
    assert s_tpe.tolist() == [cap] * num_experts
    s_out = _apply_experts(s_disp, s_probs, s_tpe.tolist(), weights)
    s_final = stat.combine(s_out)

    assert not bool(stat.over_budget.item())
    max_diff = (s_final - d_final).abs().max().item()
    assert torch.allclose(s_final, d_final, atol=1e-6, rtol=0), f"max_diff={max_diff}"


def test_static_ep1_backward_parity():
    """Gradients through the static path match the dynamic path (no overflow)."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    num_tokens, hidden, num_experts, topk = 48, 8, 4, 2
    scores, indices = _make_routing(num_tokens, num_experts, topk, seed=7)
    gen = torch.Generator().manual_seed(3)
    x0 = torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float64)
    weights = [
        torch.randn(hidden, hidden, generator=gen, dtype=torch.float64)
        for _ in range(num_experts)
    ]
    ps = _ps_single()

    def run(static):
        x = x0.clone().requires_grad_(True)
        if static:
            cap = num_tokens  # generous
            cfg = StaticCapacityConfig(num_tokens=num_tokens, expert_capacity=cap)
            disp = TokenDispatcher(num_experts, hidden, ps, use_deepep=False, static_capacity=cfg)
            d, tpe, probs = disp.dispatch(x, scores, indices)
            out = _apply_experts(d, probs, tpe.tolist(), weights)
        else:
            disp = TokenDispatcher(num_experts, hidden, ps, use_deepep=False)
            d, tpe, probs = disp.dispatch(x, scores, indices)
            out = _apply_experts(d, probs, tpe.tolist(), weights)
        final = disp.combine(out)
        final.sum().backward()
        return final.detach(), x.grad.detach()

    s_final, s_grad = run(True)
    d_final, d_grad = run(False)
    assert torch.allclose(s_final, d_final, atol=1e-9, rtol=0)
    assert torch.allclose(s_grad, d_grad, atol=1e-9, rtol=0)


# --------------------------------------------------------------------------- #
# 3. Over-budget fail-loud
# --------------------------------------------------------------------------- #
def test_static_over_budget_flag_and_raise():
    num_tokens, hidden, num_experts, topk = 64, 8, 4, 2
    # Force all tokens onto expert 0 -> heavy overflow for any small capacity.
    scores = torch.full((num_tokens, topk), 0.5)
    indices = torch.zeros(num_tokens, topk, dtype=torch.long)
    indices[:, 1] = 1  # second slot expert 1 so topk are distinct
    x = torch.randn(num_tokens, hidden)
    ps = _ps_single()

    cfg = StaticCapacityConfig(num_tokens=num_tokens, expert_capacity=8)
    disp = TokenDispatcher(num_experts, hidden, ps, use_deepep=False, static_capacity=cfg)
    disp.dispatch(x, scores, indices)
    assert bool(disp.over_budget.item()) is True
    with pytest.raises(RuntimeError, match="over budget"):
        disp.raise_if_over_budget()


def test_static_under_budget_no_raise():
    num_tokens, hidden, num_experts, topk = 32, 8, 8, 2
    scores, indices = _make_routing(num_tokens, num_experts, topk, seed=2)
    x = torch.randn(num_tokens, hidden)
    ps = _ps_single()
    cfg = StaticCapacityConfig(num_tokens=num_tokens, expert_capacity=num_tokens)
    disp = TokenDispatcher(num_experts, hidden, ps, use_deepep=False, static_capacity=cfg)
    disp.dispatch(x, scores, indices)
    assert bool(disp.over_budget.item()) is False
    disp.raise_if_over_budget()  # no raise


def test_static_rejects_wrong_token_count():
    num_experts, hidden = 4, 8
    ps = _ps_single()
    cfg = StaticCapacityConfig(num_tokens=64, expert_capacity=32)
    disp = TokenDispatcher(num_experts, hidden, ps, use_deepep=False, static_capacity=cfg)
    scores, indices = _make_routing(48, num_experts, 2, seed=0)
    with pytest.raises(ValueError, match="fixed token count"):
        disp.dispatch(torch.randn(48, hidden), scores, indices)


def test_static_disables_deepep():
    ps = _ps_single()
    cfg = StaticCapacityConfig(num_tokens=64, expert_capacity=32)
    disp = TokenDispatcher(8, 16, ps, use_deepep=True, static_capacity=cfg)
    assert disp.use_deepep is False


# --------------------------------------------------------------------------- #
# 4. No host synchronization in the captured span
# --------------------------------------------------------------------------- #
def test_static_dispatch_combine_no_host_sync():
    num_tokens, hidden, num_experts, topk = 64, 8, 8, 2
    scores, indices = _make_routing(num_tokens, num_experts, topk, seed=5)
    x = torch.randn(num_tokens, hidden)
    ps = _ps_single()
    cfg = StaticCapacityConfig(num_tokens=num_tokens, expert_capacity=num_tokens)
    disp = TokenDispatcher(num_experts, hidden, ps, use_deepep=False, static_capacity=cfg)

    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_cpu = torch.Tensor.cpu

    def _boom(name):
        def f(self, *a, **k):
            raise AssertionError(f"host sync via .{name}() inside captured span")

        return f

    # m_splits must be a python constant (no .tolist on the tpe tensor).
    torch.Tensor.item = _boom("item")
    torch.Tensor.tolist = _boom("tolist")
    torch.Tensor.cpu = _boom("cpu")
    try:
        d, tpe, probs = disp.dispatch(x, scores, indices)
        m_splits = disp._local_tpe_list  # constant list, no tensor read
        out = _apply_experts(d, probs, m_splits, [torch.eye(hidden) for _ in range(num_experts)])
        disp.combine(out)
    finally:
        torch.Tensor.item = orig_item
        torch.Tensor.tolist = orig_tolist
        torch.Tensor.cpu = orig_cpu


# --------------------------------------------------------------------------- #
# 2b. Parity: EP=2 static vs dynamic all-to-all (gloo, CPU spawn)
# --------------------------------------------------------------------------- #
def _ep2_worker(rank, world_size, num_tokens, hidden, num_experts, topk, ret):
    import torch.distributed as dist

    _install_te_stub()
    from megatron.lite.primitive.modules.dispatcher import (
        StaticCapacityConfig,
        TokenDispatcher,
    )
    from megatron.lite.primitive.parallel.state import ParallelState

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.use_deterministic_algorithms(True, warn_only=True)
    group = dist.group.WORLD
    nle = num_experts // world_size
    ps = ParallelState(ep_group=group, ep_size=world_size)

    # Same global expert weights on every rank (seed-derived); slice this rank's.
    gen = torch.Generator().manual_seed(100)
    global_w = [
        torch.randn(hidden, hidden, generator=gen, dtype=torch.float64)
        for _ in range(num_experts)
    ]
    local_w = global_w[rank * nle : (rank + 1) * nle]

    # Per-rank distinct tokens/routing.
    xgen = torch.Generator().manual_seed(200 + rank)
    x = torch.randn(num_tokens, hidden, generator=xgen, dtype=torch.float64)
    logits = torch.randn(num_tokens, num_experts, generator=xgen, dtype=torch.float64)
    topk_logits, indices = torch.topk(logits, topk, dim=1)
    scores = torch.softmax(topk_logits, dim=1).to(torch.float64)

    # Dynamic all-to-all reference.
    dyn = TokenDispatcher(num_experts, hidden, ps, use_deepep=False)
    d_disp, d_tpe, d_probs = dyn.dispatch(x, scores, indices)
    d_out = _apply_experts(d_disp, d_probs, d_tpe.tolist(), local_w)
    d_final = dyn.combine(d_out)

    # Static-capacity path (generous budget => no overflow => parity).
    cfg = StaticCapacityConfig(num_tokens=num_tokens, expert_capacity=num_tokens)
    stat = TokenDispatcher(num_experts, hidden, ps, use_deepep=False, static_capacity=cfg)
    s_disp, s_tpe, s_probs = stat.dispatch(x, scores, indices)
    assert s_tpe.tolist() == [world_size * num_tokens] * nle
    s_out = _apply_experts(s_disp, s_probs, s_tpe.tolist(), local_w)
    s_final = stat.combine(s_out)

    over = bool(stat.over_budget.item())
    max_diff = (s_final - d_final).abs().max().item()
    ret[rank] = (over, max_diff)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.distributed
def test_static_ep2_parity_with_dynamic_alltoall():
    import torch.multiprocessing as mp

    world_size, num_tokens, hidden, num_experts, topk = 2, 24, 8, 4, 2
    manager = mp.Manager()
    ret = manager.dict()
    mp.spawn(
        _ep2_worker,
        args=(world_size, num_tokens, hidden, num_experts, topk, ret),
        nprocs=world_size,
        join=True,
    )
    assert len(ret) == world_size
    for rank in range(world_size):
        over, max_diff = ret[rank]
        assert over is False, f"rank {rank} unexpectedly over budget"
        assert max_diff < 1e-9, f"rank {rank} max_diff={max_diff}"

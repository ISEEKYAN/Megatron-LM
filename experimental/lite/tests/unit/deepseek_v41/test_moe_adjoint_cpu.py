# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit routed MegaMoE adjoint vs dense autograd of the replayed graph (CPU)."""
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from megatron.lite.model.deepseek_v41.vllm import moe_adjoint
from megatron.lite.model.deepseek_v41.vllm.batching import packed_weights


def _quantize(x):
    # Stand-in for the CUDA FP8 activation codec: E4M3 codes, unit UE8M0 scales.
    return x.to(torch.float8_e4m3fn), torch.ones(x.shape[0], x.shape[1] // 32)


class _Grouped:
    def __init__(self, weights):
        self.num_gemms = len(weights)
        for i, w in enumerate(weights):
            setattr(self, f'weight{i}', w)


def _lora(world, local, hidden, inter, rank, generator):
    def p(*shape):
        return torch.nn.Parameter(torch.randn(*shape, generator=generator).bfloat16() * 0.1)

    experts = world * local
    full = SimpleNamespace(
        a1=p(rank, hidden),
        b2=p(hidden, rank),
        b1=[p(2 * inter, rank) for _ in range(experts)],
        a2=[p(rank, inter) for _ in range(experts)],
    )
    return full


def _owner(full, rank_id, local, scale):
    sl = slice(rank_id * local, (rank_id + 1) * local)
    return SimpleNamespace(
        fc1_lora_a=full.a1,
        fc2_lora_b=full.b2,
        scale=scale,
        rank=full.a1.shape[0],
        fc1_lora_b=_Grouped(full.b1[sl]),
        fc2_lora_a=_Grouped(full.a2[sl]),
        per_expert=staticmethod(lambda g: [getattr(g, f'weight{i}') for i in range(g.num_gemms)]),
    )


def _case(seed, world, local, rows, hidden=64, inter=32, rank=8, topk=3, limit=10.0, empty=False):
    g = torch.Generator().manual_seed(seed)
    experts = world * local
    # Clamp boundary is left to the Core SwiGLU VJP tests: BF16 rounding of GEMM1 flips
    # its mask against an FP32 reference.
    fc1 = [(torch.randn(2 * inter, hidden, generator=g) * 0.05).bfloat16() for _ in range(experts)]
    fc2 = [torch.randn(hidden, inter, generator=g).bfloat16() * 0.2 for _ in range(experts)]
    x = [torch.randn(rows, hidden, generator=g).bfloat16() for _ in range(world)]
    ids = [
        torch.stack([torch.randperm(experts, generator=g)[:topk] for _ in range(rows)])
        for _ in range(world)
    ]
    if empty:  # expert 0 never selected; some slots unrouted
        for i in ids:
            i[i == 0] = -1
    ids = [i.int() for i in ids]
    scores = [torch.rand(rows, topk, generator=g) for _ in range(world)]
    grad = [torch.randn(rows, hidden, generator=g).bfloat16() for _ in range(world)]
    lora = _lora(world, local, hidden, inter, rank, g)
    return fc1, fc2, x, ids, scores, grad, lora, limit


def _reference(fc1, fc2, x, ids, scores, grad, lora, limit, scale):
    """Dense FP32 autograd of the replay: STE through the FP8 input, decoded experts."""
    w1 = [moe_adjoint._decoded(packed_weights([w])).float()[0] for w in fc1]
    w2 = [moe_adjoint._decoded(packed_weights([w])).float()[0] for w in fc2]
    xs = [t.float().requires_grad_() for t in x]
    ss = [t.clone().requires_grad_() for t in scores]
    params = [lora.a1, lora.b2, *lora.b1, *lora.a2]
    leaves = [p.detach().float().requires_grad_() for p in params]
    a1, b2 = leaves[:2]
    n = len(lora.b1)
    b1, a2 = leaves[2 : 2 + n], leaves[2 + n :]
    total = 0
    for xr, ir, sr, gr in zip(xs, ids, ss, grad):
        q = _quantize(xr.detach())[0].float()
        a = xr + (q - xr).detach()
        xa = xr @ a1.T
        out = torch.zeros_like(xr)
        for t in range(xr.shape[0]):
            for j in range(ir.shape[1]):
                e = int(ir[t, j])
                if e < 0:
                    continue
                f = a[t] @ w1[e].T + xa[t] @ (b1[e] * scale).T
                gate, up = f.chunk(2)
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                h = torch.nn.functional.silu(gate) * up * sr[t, j]
                out[t] = out[t] + h @ w2[e].T + (h @ a2[e].T) @ (b2 * scale).T
        total = total + (out * gr.float()).sum()
    total.backward()
    for leaf in leaves:
        if leaf.grad is None:
            leaf.grad = torch.zeros_like(leaf)
    return [t.grad for t in xs], [t.grad for t in ss], leaves


def _close(actual, expected, tol=0.03):
    actual, expected = actual.float(), expected.float()
    err = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    assert torch.isfinite(actual).all() and err <= tol, float(err)


def _run(rank_id, world, local, case, scale, group):
    fc1, fc2, x, ids, scores, grad, lora, limit = case
    sl = slice(rank_id * local, (rank_id + 1) * local)
    owner = _owner(lora, rank_id, local, scale)
    return moe_adjoint.routed_adjoint(
        x[rank_id],
        scores[rank_id],
        ids[rank_id],
        grad[rank_id],
        fc1[sl],
        fc2[sl],
        owner,
        limit,
        group,
        overlap=lambda: 'shared',
    )


@pytest.mark.parametrize('empty', [False, True])
def test_single_rank_matches_dense_autograd(monkeypatch, empty):
    monkeypatch.setattr(moe_adjoint, '_quantize', _quantize)
    case = _case(0, 1, 4, 10, empty=empty)
    scale = 2.0
    dx, ds, grads, shared = _run(0, 1, 4, case, scale, None)
    assert shared == 'shared'
    rdx, rds, leaves = _reference(*case, scale)
    _close(dx, rdx[0])
    _close(ds, rds[0])
    n = len(case[6].b1)
    da1, db1, da2, db2 = grads
    _close(da1, leaves[0].grad)
    _close(db2, leaves[1].grad)
    _close(db1, torch.stack([l.grad for l in leaves[2 : 2 + n]]))
    _close(da2, torch.stack([l.grad for l in leaves[2 + n :]]))


def test_empty_source_rows(monkeypatch):
    monkeypatch.setattr(moe_adjoint, '_quantize', _quantize)
    fc1, fc2, x, ids, scores, grad, lora, limit = _case(1, 1, 3, 3)
    x, ids, scores, grad = (t[0][:0] for t in (x, ids, scores, grad))
    dx, ds, grads, _ = moe_adjoint.routed_adjoint(
        x, scores, ids, grad, fc1, fc2, _owner(lora, 0, 2, 1.0), limit, None
    )
    assert dx.shape == (0, 64) and ds.shape == ids.shape
    assert all(float(g.abs().sum()) == 0 for g in grads)


def _worker(rank_id, world, port, result):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.distributed.init_process_group('gloo', rank=rank_id, world_size=world)
    moe_adjoint._quantize = _quantize
    try:
        case = _case(5, world, 2, 7, empty=True)
        out = _run(rank_id, world, 2, case, 0.5, torch.distributed.group.WORLD)
        dx, ds, (da1, db1, da2, db2), _ = out
        result[rank_id] = (dx, ds, da1, db1, da2, db2)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_exchange_matches_dense_autograd():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    world = 2
    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(_worker, args=(world, port, result), nprocs=world, join=True)
    case = _case(5, world, 2, 7, empty=True)
    rdx, rds, leaves = _reference(*case, 0.5)
    n = len(case[6].b1)
    rb1 = torch.stack([l.grad for l in leaves[2 : 2 + n]])
    ra2 = torch.stack([l.grad for l in leaves[2 + n :]])
    # Shared factors: each rank holds a partial sum (DDP reduces them).
    _close(result[0][2] + result[1][2], leaves[0].grad)
    _close(result[0][5] + result[1][5], leaves[1].grad)
    for r in range(world):
        dx, ds, _, db1, da2, _ = result[r]
        _close(dx, rdx[r])
        _close(ds, rds[r])
        _close(db1, rb1[r * 2 : (r + 1) * 2])
        _close(da2, ra2[r * 2 : (r + 1) * 2])

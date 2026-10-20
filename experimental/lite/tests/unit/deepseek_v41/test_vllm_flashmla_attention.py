# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashMLA sparse attention core: serving-bitwise forward, FP64-checked backward, CP.

Forward gate: every output row (BF16 attention output, then the FP8 ``o_proj``
input and its scales) equals vLLM's own ``forward_mqa`` on paged caches for the
same tokens, whether serving ran them as one prefill batch, chunked prefill or
batch-invariant decode rows. Backward gate (tolerances fixed before running):
against an FP64 autograd reference of the STE contract (no cuDNN, no closed-form
RoPE transposes), every gradient has relative L2 error <= 1e-2 and norm ratio in
[0.99, 1.01]. The candidate rounds Q, KV, O, dO, P and dS to BF16 (relative
step 2^-8) inside sums of 128-640 terms, so its expected error is a few 1e-3;
1e-2 leaves ~3x margin and a planted RoPE-transpose or sink error must exceed it.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _flashmla_fixture import Serving, make_inputs, output_quant, reference_grads, v41_rope

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")]
REL_L2, NORM_RATIO = 1e-2, 0.01


@pytest.fixture(autouse=True)
def _batch_invariant(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")


def _module():
    from megatron.lite.model.deepseek_v41.vllm import flashmla_attention

    return flashmla_attention


def _cfg(cr):
    return _module().FlashMLALayerConfig(compress_ratio=cr)


def _visible(inputs, cfg, cos_sin):
    """Training forward internals: (q_rope, workspace, indices, lens, out, lse, plan).

    ``out``/``lse`` come from the attention path the installed vLLM serves;
    ``workspace``/``indices``/``lens`` are the BF16 gather view backward uses.
    """
    m = _module()
    total = inputs.q.shape[0]
    plan = m._plan(inputs.cu, 0, total, cfg, inputs.q.device)
    latent = inputs.latent.contiguous() if inputs.latent is not None else None
    q_rope, caches = m._insert(
        inputs.q,
        inputs.kv,
        inputs.kv,
        latent,
        inputs.positions,
        inputs.positions,
        cfg,
        cos_sin,
        plan,
        False,
    )
    attend = m._attend_paged if m.inference_path() == "paged" else m._attend_gather
    out, lse = attend(q_rope, caches, inputs.topk, inputs.sink, cfg, plan)
    workspace, indices, lens = m._gather(caches, inputs.topk, total, cfg, plan)
    return q_rope, workspace, indices, lens, out, lse, plan


def _serving_schedules(lengths):
    whole = [(r, 0, n) for r, n in enumerate(lengths)]
    n = lengths[0]
    cuts = sorted({0, 1, 127, 128, n // 2, n - 1, n})
    chunked = [[(0, a, b)] for a, b in zip(cuts, cuts[1:]) if a < b]
    decode = [[(0, p, p + 1)] for p in sorted({0, 126, 127, 128, n - 1}) if p < n]
    batch_decode = [[(r, k - 1, k) for r, k in enumerate(lengths)]]
    return [whole, whole[::-1], *chunked, *decode, *batch_decode]


FORWARD_CASES = [
    (0, [300]),
    (0, [700, 65, 129]),
    (2, [300]),
    (2, [700, 65, 129]),
    (2, [4500]),
    (1, [300]),
    (1, [700, 65, 129]),
    (1, [4500]),
    (2, [16384]),
    (1, [16384]),
]


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("cr,lengths", FORWARD_CASES)
def test_forward_matches_serving_bitwise(cr, lengths):
    device = torch.device("cuda")
    cos_sin = v41_rope(cr, device)
    inputs = make_inputs(lengths, cr, seed=11 + cr, device=device)
    cfg = _cfg(cr)
    q_rope, _, _, _, out, _, _ = _visible(inputs, cfg, cos_sin)
    result = _module().flashmla_attention_core(
        inputs.q,
        inputs.kv,
        inputs.latent,
        inputs.topk,
        inputs.positions,
        inputs.sink,
        cfg,
        cos_sin,
        cu_seqlens=inputs.cu,
    )
    serving = Serving(inputs, cr, cos_sin, device)
    try:
        assert torch.equal(serving.q_rope, q_rope)
        for schedule in _serving_schedules(lengths):
            rows, expected = serving.run(schedule)
            assert torch.equal(out[rows], expected), schedule
        data, scale = output_quant(out, inputs.positions, cos_sin)
        _, whole = serving.run([(r, 0, n) for r, n in enumerate(lengths)])
        want_data, want_scale = output_quant(whole, inputs.positions, cos_sin)
        assert torch.equal(want_data.view(torch.uint8), data.view(torch.uint8))
        assert torch.equal(want_scale, scale)
        assert torch.equal(result.data.view(torch.uint8), data.view(torch.uint8))
        assert torch.equal(result.scale, scale)
        assert result.grad.shape == result.data.shape and result.grad.requires_grad is False
    finally:
        serving.close()


def _metrics(g, ref):
    g, ref = g.double().flatten(), ref.double().flatten()
    rn = ref.norm()
    return {
        "rel_l2": ((g - ref).norm() / rn.clamp_min(1e-300)).item(),
        "norm_ratio": (g.norm() / rn.clamp_min(1e-300)).item(),
        "max_abs": (g - ref).abs().max().item(),
        "ref_max": ref.abs().max().item(),
    }


def _token_view(inputs, cfg, cos_sin):
    """Serving-visible forward values mapped to tokens for the reference."""
    from types import SimpleNamespace

    q_rope, workspace, indices, lens, out, lse, plan = _visible(inputs, cfg, cos_sin)
    flat = workspace.view(-1, workspace.shape[-1])
    total = inputs.q.shape[0]
    swa = torch.zeros(total, flat.shape[1], dtype=flat.dtype, device=flat.device)
    mask = plan.swa_dest >= 0
    swa[plan.swa_dest[mask]] = flat[mask]
    row_of = torch.full((flat.shape[0],), -1, dtype=torch.int64, device=flat.device)
    row_of[mask] = plan.swa_dest[mask]
    comp = comp_positions = None
    if cfg.compress_ratio:
        cmask = plan.comp_dest >= 0
        comp = torch.zeros(
            inputs.latent.shape[0], flat.shape[1], dtype=flat.dtype, device=flat.device
        )
        comp[plan.comp_dest[cmask]] = flat[cmask]
        row_of[cmask] = total + plan.comp_dest[cmask]
        comp_positions = plan.comp_positions
    width = indices.shape[1]
    inside = torch.arange(width, device=indices.device)[None, :] < lens[:, None]
    keys = torch.where(
        inside & (indices >= 0),
        row_of[indices.clamp(min=0).long()],
        torch.full_like(indices, -1, dtype=torch.int64),
    )
    return SimpleNamespace(
        q_rope=q_rope, swa=swa, comp=comp, comp_positions=comp_positions, keys=keys, out=out
    )


def _candidate_grads(inputs, cfg, cos_sin, d_o, deterministic=False):
    q = inputs.q.clone().requires_grad_()
    kv = inputs.kv.clone().requires_grad_()
    sink = inputs.sink.clone().requires_grad_()
    latent = inputs.latent.clone().requires_grad_() if inputs.latent is not None else None
    result = _module().flashmla_attention_core(
        q,
        kv,
        latent,
        inputs.topk,
        inputs.positions,
        sink,
        cfg,
        cos_sin,
        cu_seqlens=inputs.cu,
        deterministic=deterministic,
    )
    result.grad.backward(d_o.to(result.grad.dtype).view(result.grad.shape))
    return q.grad, kv.grad, None if latent is None else latent.grad, sink.grad


def _d_o(inputs, kind, seed):
    shape = inputs.q.shape
    g = torch.Generator(device=inputs.q.device).manual_seed(seed)
    d_o = torch.randn(shape, device=inputs.q.device, generator=g).bfloat16().float()
    if kind == "sparse":
        keep = torch.zeros(shape[0], dtype=torch.bool, device=d_o.device)
        keep[torch.randperm(shape[0], generator=g, device=d_o.device)[: max(shape[0] // 16, 1)]] = (
            True
        )
        d_o = d_o * keep[:, None, None]
    return d_o


BACKWARD_CASES = [
    (0, [300]),
    (0, [700, 65, 129]),
    (2, [300]),
    (2, [700, 65, 129]),
    (2, [4500]),
    (1, [300]),
    (1, [700, 65, 129]),
    (1, [4500]),
    (2, [8192]),
    (1, [8192]),
]


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("kind", ["dense", "sparse"])
@pytest.mark.parametrize("cr,lengths", BACKWARD_CASES)
def test_backward_matches_fp64_reference(cr, lengths, kind, record_property):
    device = torch.device("cuda")
    cos_sin = v41_rope(cr, device)
    inputs = make_inputs(lengths, cr, seed=23 + cr, device=device)
    cfg = _cfg(cr)
    d_o = _d_o(inputs, kind, seed=31)
    view = _token_view(inputs, cfg, cos_sin)
    ref = reference_grads(inputs, cfg, cos_sin, view, d_o)
    fwd = _metrics(view.out.float(), ref.out)
    assert fwd["rel_l2"] < REL_L2, fwd  # the reference sees the visible forward
    got = _candidate_grads(inputs, cfg, cos_sin, d_o)
    names = ["dq", "d_swa_kv", "d_comp_latent", "d_sink"]
    refs = [ref.q, ref.kv, ref.latent, ref.sink]
    report = {}
    for name, g, r in zip(names, got, refs):
        if r is None:
            assert g is None
            continue
        assert g is not None and torch.isfinite(g).all(), name
        report[name] = _metrics(g, r)
    record_property("metrics", report)
    print(f"cr={cr} lengths={lengths} kind={kind} {report}")
    for name, m in report.items():
        assert m["rel_l2"] <= REL_L2, (name, m)
        assert abs(m["norm_ratio"] - 1) <= NORM_RATIO, (name, m)


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_backward_zero_output_gradient_is_zero():
    device = torch.device("cuda")
    cos_sin = v41_rope(2, device)
    inputs = make_inputs([300], 2, seed=3, device=device)
    for g in _candidate_grads(inputs, _cfg(2), cos_sin, torch.zeros_like(inputs.q.float())):
        assert torch.count_nonzero(g) == 0


@pytest.mark.gpus(1, min_architecture="blackwell")
def test_backward_deterministic_mode_is_bitwise_reproducible():
    device = torch.device("cuda")
    cos_sin = v41_rope(2, device)
    inputs = make_inputs([700, 65, 129], 2, seed=4, device=device)
    d_o = _d_o(inputs, "dense", seed=5)
    first = _candidate_grads(inputs, _cfg(2), cos_sin, d_o, deterministic=True)
    second = _candidate_grads(inputs, _cfg(2), cos_sin, d_o, deterministic=True)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("planted", ["no_rope_transpose", "no_sink"])
def test_backward_gate_rejects_planted_errors(planted, monkeypatch):
    """The 1e-2 gate resolves a missing RoPE transpose and a dropped sink gradient."""
    device = torch.device("cuda")
    cos_sin = v41_rope(2, device)
    inputs = make_inputs([700, 65, 129], 2, seed=23, device=device)
    cfg = _cfg(2)
    d_o = _d_o(inputs, "dense", seed=31)
    ref = reference_grads(inputs, cfg, cos_sin, _token_view(inputs, cfg, cos_sin), d_o)
    m = _module()
    if planted == "no_rope_transpose":
        monkeypatch.setattr(
            m,
            "_rope_transpose",
            lambda grad, *a, out_dtype=None, **k: grad.to(out_dtype or grad.dtype),
        )
        dq = _candidate_grads(inputs, cfg, cos_sin, d_o)[0]
        assert _metrics(dq, ref.q)["rel_l2"] > REL_L2
    else:
        original = m._sparse_backward

        def drop_sink(*args):
            dq, dkv, d_sink = original(*args)
            return dq, dkv, torch.zeros_like(d_sink)

        monkeypatch.setattr(m, "_sparse_backward", drop_sink)
        d_sink = _candidate_grads(inputs, cfg, cos_sin, d_o)[3]
        assert _metrics(d_sink, ref.sink)["rel_l2"] > REL_L2


# ---------------------------------------------------------------------------
# Context parallel
# ---------------------------------------------------------------------------

CP_LENGTHS = [700, 65, 129, 130]  # 1024 rows; CP2 splits document 0 at row 512


def _cp_worker(rank, world, root, cr, out_path):
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", f"file://{root}/store", rank=rank, world_size=world)
    device = torch.device("cuda", rank)
    cos_sin = v41_rope(cr, device)
    inputs = make_inputs(CP_LENGTHS, cr, seed=41, device=device)
    cfg = _cfg(cr)
    d_o = _d_o(inputs, "dense", seed=43)
    ref = reference_grads(inputs, cfg, cos_sin, _token_view(inputs, cfg, cos_sin), d_o)
    full = _candidate_grads(inputs, cfg, cos_sin, d_o)
    full_out = _module().flashmla_attention_core(
        inputs.q,
        inputs.kv,
        inputs.latent,
        inputs.topk,
        inputs.positions,
        inputs.sink,
        cfg,
        cos_sin,
        cu_seqlens=inputs.cu,
    )

    total = inputs.q.shape[0]
    a, b = rank * total // world, (rank + 1) * total // world
    if cr:
        boundary = (inputs.positions + 1) % cr == 0
        ea, eb = int(boundary[:a].sum()), int(boundary[:b].sum())
    q = inputs.q[a:b].clone().requires_grad_()
    kv = inputs.kv[a:b].clone().requires_grad_()
    sink = inputs.sink.clone().requires_grad_()
    latent = inputs.latent[ea:eb].clone().requires_grad_() if cr else None
    result = _module().flashmla_attention_core(
        q,
        kv,
        latent,
        inputs.topk[a:b] if cr else None,
        inputs.positions[a:b],
        sink,
        cfg,
        cos_sin,
        cu_seqlens=inputs.cu,
        cp_group=dist.group.WORLD,
    )
    result.grad.backward(d_o[a:b].to(result.grad.dtype).view(result.grad.shape))
    sink_sum = sink.grad.clone()
    dist.all_reduce(sink_sum)
    pairs = {
        "dq": (q.grad, full[0][a:b], ref.q[a:b]),
        "d_swa_kv": (kv.grad, full[1][a:b], ref.kv[a:b]),
        "d_sink_sum": (sink_sum, full[3], ref.sink),
    }
    if cr:
        pairs["d_comp_latent"] = (latent.grad, full[2][ea:eb], ref.latent[ea:eb])
    report = {
        "data_equal": torch.equal(
            result.data.view(torch.uint8), full_out.data[a:b].view(torch.uint8)
        ),
        "scale_equal": torch.equal(result.scale, full_out.scale[a:b]),
        "dq_equal_cp1": torch.equal(q.grad, full[0][a:b]),
        "vs_ref": {k: _metrics(g, r) for k, (g, _, r) in pairs.items()},
        "cp1_vs_ref": {k: _metrics(c, r) for k, (_, c, r) in pairs.items()},
        "vs_cp1": {k: _metrics(g, c) for k, (g, c, _) in pairs.items()},
    }
    torch.save(report, f"{out_path}.{rank}")
    dist.destroy_process_group()


@pytest.mark.gpus(2, min_architecture="blackwell")
@pytest.mark.parametrize("cr", [0, 2, 1])
def test_cp2_matches_cp1(cr, tmp_path):
    """CP2 forward and dq equal CP1 bitwise; CP2 gradients pass the FP64 gate.

    dKV differs from CP1 by summation order and by the BF16 rounding of each
    rank's partial dKV (cuDNN emits BF16) before the FP32 reduce-scatter; that
    difference is reported (``vs_cp1``), the gate is the reference one.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    out = str(tmp_path / "report")
    mp.spawn(_cp_worker, args=(2, str(tmp_path), cr, out), nprocs=2, join=True)
    for rank in range(2):
        report = torch.load(f"{out}.{rank}")
        print(f"cr={cr} rank={rank} {report}")
        assert report["data_equal"] and report["scale_equal"] and report["dq_equal_cp1"]
        for name, m in report["vs_ref"].items():
            assert m["rel_l2"] <= REL_L2 and abs(m["norm_ratio"] - 1) <= NORM_RATIO, (name, m)


_LOAD_ORDER = r"""
import importlib, sys
import torch
from _flashmla_fixture import make_inputs, v41_rope
from megatron.lite.model.deepseek_v41.vllm import flashmla_attention as m, load_vllm_flashmla
if sys.argv[1] == "vllm_first":
    load_vllm_flashmla()
importlib.import_module("megatron.core.transformer.attention")
assert "flash_mla.cuda" in sys.modules
x = make_inputs([600], 2, seed=3, device=torch.device("cuda"))
try:
    m.flashmla_attention_core(
        x.q, x.kv, x.latent, x.topk, x.positions, x.sink, m.FlashMLALayerConfig(2),
        v41_rope(2, torch.device("cuda")),
    )
    torch.cuda.synchronize()
    print("RESULT ok")
except RuntimeError as exc:
    print("RESULT", exc)
"""


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("order", ["vllm_first", "megatron_first"])
def test_core_runs_beside_megatron_flash_mla(order):
    """pip ``flash_mla`` (loaded by Megatron Core) and vLLM's FlashMLA share kernel
    symbols and the first loaded serves both: with vLLM's loaded first the core
    runs in a process that also imports Megatron's attention; the other order
    fails at entry with an actionable error, not inside the kernel."""
    env = {**os.environ, "VLLM_BATCH_INVARIANT": "1"}
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(__file__), env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-c", _LOAD_ORDER, order],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    result = [line for line in out.stdout.splitlines() if line.startswith("RESULT")]
    assert result, out.stderr[-2000:]
    if order == "vllm_first":
        assert result[-1] == "RESULT ok"
    else:
        assert "pip flash_mla was loaded before vLLM's FlashMLA" in result[-1]

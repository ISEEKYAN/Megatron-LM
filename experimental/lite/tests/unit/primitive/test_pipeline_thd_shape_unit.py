# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for THD dynamic-batch P2P shapes in the pipeline layer.

Under THD + dynamic batching every micro-batch packs a different token count, so
the inter-stage hidden — and its P2P recv buffer — is a different shape per
micro-batch. A single fixed buffer sized from the first batch truncates the recv
of later, larger micro-batches (NCCL size mismatch -> hang).

The fix is Megatron's *dynamic shape exchange*: before every inter-stage tensor
transfer the sender transmits the exact ``size()`` of the tensor it is about to
send, and the receiver sizes its recv buffer from that — no local shape
derivation, no fixed fallback. These tests cover three layers:

* ``_communicate_shapes`` — the shape hop itself (right ops, right return, fail
  loud on a desync), with the dist fabric mocked.
* ``_send_recv_pipeline(dynamic_shape=True)`` — the recv buffer is sized from the
  exchanged shape, not the passed ``tensor_shape``.
* ``_1f1b_schedule`` — every recv site consumes the peer-sent shape for its own
  micro-batch (off-by-one in the recv<->mb map = silent deadlock on GPU).
"""

from __future__ import annotations

import multiprocessing
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.parallel import pipeline as pl


def _make_ps(pp_size: int, pp_rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        pp_size=pp_size,
        pp_rank=pp_rank,
        pp_is_first=(pp_rank == 0),
        pp_is_last=(pp_rank == pp_size - 1),
        pp_prev_rank=pp_rank - 1,
        pp_next_rank=pp_rank + 1,
        pp_group=None,
    )


# ══════════════════════════════════════════════════════════════════════
# _communicate_shapes: the dynamic shape hop
# ══════════════════════════════════════════════════════════════════════
class _FakeReq:
    def wait(self):
        return None


class _FakeP2POp:
    """Stand-in for dist.P2POp that needs no initialized process group."""

    def __init__(self, op, tensor, peer, group=None, tag=0):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group


def _install_fake_fabric(monkeypatch, incoming: dict[int, tuple[int, ...]]):
    """Mock the dist layer so _communicate_shapes runs on CPU with a scripted peer.

    ``incoming`` maps peer-rank -> the shape that peer "sent" us; irecv ops for
    that peer are filled with it. Sends are recorded for assertion. Returns the
    list that captures ``(peer, sent_shape)`` for every isend.
    """
    sends: list[tuple[int, tuple[int, ...]]] = []

    def fake_batch_isend_irecv(ops):
        for o in ops:
            if o.op is dist.isend:
                sends.append((o.peer, tuple(int(x) for x in o.tensor.tolist())))
            elif o.op is dist.irecv:
                o.tensor.copy_(torch.tensor(incoming[o.peer], dtype=torch.int64))
        return [_FakeReq()]

    monkeypatch.setattr(pl.torch.cuda, "current_device", lambda: "cpu", raising=False)
    monkeypatch.setattr(pl.torch.cuda, "synchronize", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(pl.dist, "P2POp", _FakeP2POp, raising=False)
    monkeypatch.setattr(pl.dist, "batch_isend_irecv", fake_batch_isend_irecv, raising=False)
    return sends


def test_communicate_shapes_middle_stage_send_and_recv(monkeypatch):
    """A middle stage sends its hidden size forward and receives the prev shape."""
    ps = _make_ps(4, 1)  # prev=0, next=2
    send_fwd = torch.zeros(7, 1, 8)  # this stage forwards a [7,1,8] hidden
    incoming = {0: (5, 1, 8)}  # the prev stage tells us the next mb is [5,1,8]
    sends = _install_fake_fabric(monkeypatch, incoming)

    recv_fwd_shape, recv_bwd_shape = pl._communicate_shapes(
        send_fwd=send_fwd, send_bwd=None, recv_fwd=True, recv_bwd=False, ps=ps
    )

    assert recv_fwd_shape == (5, 1, 8)
    assert recv_bwd_shape is None
    # exactly one isend, to the NEXT rank, carrying send_fwd's true size
    assert sends == [(ps.pp_next_rank, (7, 1, 8))]


def test_communicate_shapes_backward_direction(monkeypatch):
    """recv_bwd reads a shape from the NEXT rank; send_bwd goes to the PREV rank."""
    ps = _make_ps(4, 1)
    send_bwd = torch.zeros(9, 1, 8)
    incoming = {2: (3, 1, 8)}  # grad shape arrives from the next rank
    sends = _install_fake_fabric(monkeypatch, incoming)

    recv_fwd_shape, recv_bwd_shape = pl._communicate_shapes(
        send_fwd=None, send_bwd=send_bwd, recv_fwd=False, recv_bwd=True, ps=ps
    )

    assert recv_fwd_shape is None
    assert recv_bwd_shape == (3, 1, 8)
    assert sends == [(ps.pp_prev_rank, (9, 1, 8))]


def test_communicate_shapes_fail_loud_on_desync(monkeypatch):
    """A non-positive received dim (stage disagreement) must raise, not truncate."""
    ps = _make_ps(2, 1)
    incoming = {0: (0, 1, 8)}  # prev "sent" a zero dim -> protocol desync
    _install_fake_fabric(monkeypatch, incoming)
    monkeypatch.setattr(pl.dist, "get_rank", lambda: 1, raising=False)

    with pytest.raises(RuntimeError, match="non-positive dim"):
        pl._communicate_shapes(
            send_fwd=None, send_bwd=None, recv_fwd=True, recv_bwd=False, ps=ps
        )


# ══════════════════════════════════════════════════════════════════════
# _send_recv_pipeline(dynamic_shape=True): buffer sized from the exchange
# ══════════════════════════════════════════════════════════════════════
def test_send_recv_pipeline_sizes_recv_buffer_from_exchange(monkeypatch):
    """The recv buffer is the EXCHANGED shape, not the passed tensor_shape."""
    ps = _make_ps(4, 1)

    # The exchange reports a [5,1,8] forward input regardless of tensor_shape.
    monkeypatch.setattr(
        pl, "_communicate_shapes", lambda *a, **k: ((5, 1, 8), None), raising=True
    )

    captured: dict[str, torch.Tensor] = {}

    def fake_batch_isend_irecv(ops):
        for o in ops:
            if o.op is dist.irecv:
                captured["fwd"] = o.tensor
        return [_FakeReq()]

    real_empty = torch.empty

    def cpu_empty(*a, **kw):
        kw.pop("device", None)
        return real_empty(*a, **kw)

    monkeypatch.setattr(pl.torch, "empty", cpu_empty, raising=False)
    monkeypatch.setattr(pl.dist, "P2POp", _FakeP2POp, raising=False)
    monkeypatch.setattr(pl.dist, "batch_isend_irecv", fake_batch_isend_irecv, raising=False)
    monkeypatch.setattr(pl.dist, "get_rank", lambda: 1, raising=False)

    fwd_buf, _ = pl._send_recv_pipeline(
        None, None, True, False, ps,
        (999, 1, 8),  # deliberately WRONG fixed shape; must be ignored
        dynamic_shape=True,
    )

    assert tuple(fwd_buf.shape) == (5, 1, 8)
    assert tuple(captured["fwd"].shape) == (5, 1, 8)


# ══════════════════════════════════════════════════════════════════════
# _1f1b_schedule: recv<->microbatch mapping under variable shapes
# ══════════════════════════════════════════════════════════════════════
class _MockModel(torch.nn.Module):
    """Records the input_tensor set on it so we can catch a None/garbage input."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.weight = torch.nn.Parameter(torch.ones(hidden, hidden))
        self._input_tensor = None

    def set_input_tensor(self, t):
        self._input_tensor = t


def _run_schedule(pp_size, pp_rank, seq_lens, hidden=8):
    """Run the real _1f1b_schedule for one rank; return recorded recv shapes.

    ``_send_recv_pipeline`` is mocked to play the peer: it must be called with
    ``dynamic_shape=True`` (the schedule tracks no shapes itself) and fabricates a
    recv tensor of the peer-sent shape, pulled in transfer order. Recording those
    shapes proves the schedule requests each recv for the right micro-batch.
    """
    ps = _make_ps(pp_size, pp_rank)
    num_mb = len(seq_lens)
    batches = [{"S": s} for s in seq_lens]
    fwd_shapes = [(int(s), 1, hidden) for s in seq_lens]

    model = _MockModel(hidden)
    recorded_fwd: list[tuple[int, ...]] = []
    recorded_bwd: list[tuple[int, ...]] = []
    # Forward inputs arrive in mb order; backward grads arrive oldest-first, which
    # is also mb order — so both queues are just the per-mb shapes.
    fwd_q = list(fwd_shapes)
    bwd_q = list(fwd_shapes)

    def fake_srp(
        send_fwd, send_bwd, recv_fwd, recv_bwd, ps_, tensor_shape,
        *, fwd_recv_buf=None, bwd_recv_buf=None, batch_p2p=True, clone_recv=False,
        dynamic_shape=False,
    ):
        assert dynamic_shape, "1F1B schedule must use Megatron dynamic shape exchange"
        assert fwd_recv_buf is None and bwd_recv_buf is None, (
            "dynamic-shape recv must not pre-size a buffer"
        )
        fwd_out = None
        bwd_out = None
        if recv_fwd:
            shp = fwd_q.pop(0)
            recorded_fwd.append(shp)
            fwd_out = torch.ones(shp, requires_grad=True)  # peer-sent hidden
        if recv_bwd:
            shp = bwd_q.pop(0)
            recorded_bwd.append(shp)
            bwd_out = torch.ones(shp)  # peer-sent grad
        return fwd_out, bwd_out

    def forward_step_fn(m, batch):
        s = int(batch["S"])
        if ps.pp_is_first:
            base = torch.ones(s, 1, hidden, requires_grad=True)
        else:
            inp = unwrap_model(m)._input_tensor
            assert inp is not None, "middle/last stage forwarded with a None input"
            assert tuple(inp.shape) == (s, 1, hidden), (
                f"input shape {tuple(inp.shape)} != expected {(s, 1, hidden)} for mb S={s}"
            )
            base = inp
        hidden_t = base * m.weight.sum()
        out = {"hidden_states": hidden_t}
        if ps.pp_is_last:
            out["loss"] = hidden_t.float().sum()
        return out

    orig_srp = pl._send_recv_pipeline
    pl._send_recv_pipeline = fake_srp
    try:
        pl._1f1b_schedule(
            forward_step_fn,
            model,
            iter([(b, None) for b in batches]),
            num_mb,
            SimpleNamespace(num_microbatches=num_mb),
            ps,
            fwd_shapes[0],
        )
    finally:
        pl._send_recv_pipeline = orig_srp

    return recorded_fwd, recorded_bwd, fwd_shapes


VARLEN = [5, 9, 3, 7]  # deliberately non-uniform, ascending & descending mix


@pytest.mark.parametrize("pp_rank", [1, 2])
def test_middle_stage_recv_shapes_match_each_microbatch(pp_rank):
    """Every fwd/bwd recv on a middle stage is sized for its own micro-batch."""
    recorded_fwd, recorded_bwd, fwd_shapes = _run_schedule(4, pp_rank, VARLEN)
    assert recorded_fwd == fwd_shapes, (recorded_fwd, fwd_shapes)
    assert recorded_bwd == fwd_shapes, (recorded_bwd, fwd_shapes)


def test_last_stage_recv_shapes_match_each_microbatch():
    recorded_fwd, recorded_bwd, fwd_shapes = _run_schedule(4, 3, VARLEN)
    # last stage receives every forward input, sends no forward -> no bwd recv
    assert recorded_fwd == fwd_shapes, (recorded_fwd, fwd_shapes)
    assert recorded_bwd == [], recorded_bwd


def test_first_stage_recv_shapes_match_each_microbatch():
    recorded_fwd, recorded_bwd, fwd_shapes = _run_schedule(4, 0, VARLEN)
    # first stage never receives a forward input; it receives a bwd grad per mb
    assert recorded_fwd == [], recorded_fwd
    assert recorded_bwd == fwd_shapes, (recorded_bwd, fwd_shapes)


def test_pp2_recv_shapes_match_each_microbatch():
    """PP2 first stage: no fwd recv, one bwd grad per mb in mb order."""
    recorded_fwd, recorded_bwd, fwd_shapes = _run_schedule(2, 0, VARLEN)
    assert recorded_fwd == []
    assert recorded_bwd == fwd_shapes, (recorded_bwd, fwd_shapes)
    # PP2 last stage: one fwd input per mb, no bwd recv.
    recorded_fwd2, recorded_bwd2, _ = _run_schedule(2, 1, VARLEN)
    assert recorded_fwd2 == fwd_shapes, (recorded_fwd2, fwd_shapes)
    assert recorded_bwd2 == []


# ══════════════════════════════════════════════════════════════════════
# Tier-A: real gloo 2-process PP2, real _send_recv_pipeline, variable seq
# ══════════════════════════════════════════════════════════════════════
# The tests above mock the dist fabric. This tier runs the *actual*
# ``_send_recv_pipeline`` across two real gloo processes (PP2: rank0=first stage,
# rank1=last stage) feeding variable-length micro-batches S=[5, 3, 7] — the THD
# dynamic-batch case. It is the zero-GPU isolation proof of the "Megatron way":
#
#   * NEW (``dynamic_shape=True``): the recv buffer is sized from the shape the
#     sender puts on the wire, so every micro-batch — including the S=7 one that
#     is larger than the first (S=5) — round-trips bitwise. 0 mismatch, 0 fallback.
#   * OLD (fixed buffer sized from the first micro-batch, ``dynamic_shape=False``):
#     the S=7 send overflows the S=5 recv buffer; gloo enforces the size and aborts
#     the receiver (``op.preamble.length <= op.nbytes`` -> SIGABRT), surfacing as a
#     spawn failure. NEW green, OLD red — exactly the shape_fn-free contract.
#
# gloo point-to-point does not support ``batch_isend_irecv``, so these drive the
# direct isend/irecv path (``batch_p2p=False``); the shape-exchange logic is
# identical either way. bf16 p2p is a CUDA/NCCL concern, orthogonal to buffer
# sizing, so the tensor dtype is pinned to fp32 for the gloo fabric.

_PP_H = 8
_PP_VARLEN = [5, 3, 7]  # micro-batch seq lengths; 7 > first(5) is the overflow case


def _pp_hidden(mb_idx: int, seqlen: int) -> torch.Tensor:
    """Deterministic, rank-independent inter-stage hidden [S, 1, H] for a mb."""
    gen = torch.Generator().manual_seed(20260715 + mb_idx)
    return torch.randn(seqlen, 1, _PP_H, generator=gen)


def _pp_state(rank: int) -> SimpleNamespace:
    return SimpleNamespace(
        pp_size=2,
        pp_rank=rank,
        pp_is_first=(rank == 0),
        pp_is_last=(rank == 1),
        pp_prev_rank=rank - 1,
        pp_next_rank=rank + 1,
        pp_group=None,  # default (world) group == the PP group for world_size=2
    )


def _dynamic_shape_worker(rank, world, port, results):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world)
    )
    import torch.distributed as dist

    from megatron.lite.primitive.parallel import pipeline as pl

    pl._PIPELINE_TENSOR_DTYPE = torch.float32  # gloo-safe; buffer sizing is dtype-agnostic
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        ps = _pp_state(rank)
        detail = []
        for i, s in enumerate(_PP_VARLEN):
            if rank == 0:
                hid = _pp_hidden(i, s)
                pl._send_recv_pipeline(
                    hid, None, False, False, ps, (1, 1, _PP_H),
                    batch_p2p=False, dynamic_shape=True,
                )
            else:
                fwd_buf, _ = pl._send_recv_pipeline(
                    None, None, True, False, ps, (1, 1, _PP_H),
                    batch_p2p=False, dynamic_shape=True,
                )
                expect = _pp_hidden(i, s)
                shape_ok = tuple(fwd_buf.shape) == (s, 1, _PP_H)
                bitwise_ok = bool(torch.equal(fwd_buf.detach().float(), expect.float()))
                detail.append((s, shape_ok, bitwise_ok))
        results.append((rank, detail))
    finally:
        dist.destroy_process_group()


def _fixed_buffer_worker(rank, world, port, results):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world)
    )
    import torch.distributed as dist

    from megatron.lite.primitive.parallel import pipeline as pl

    pl._PIPELINE_TENSOR_DTYPE = torch.float32
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        ps = _pp_state(rank)
        first_len = _PP_VARLEN[0]  # fixed recv buffer sized from the FIRST mb (S=5)
        overflow_idx, overflow_len = 2, _PP_VARLEN[2]  # the S=7 overflow mb
        if rank == 0:
            hid = _pp_hidden(overflow_idx, overflow_len)  # sends [7, 1, H]
            pl._send_recv_pipeline(
                hid, None, False, False, ps, (first_len, 1, _PP_H),
                batch_p2p=False, dynamic_shape=False,
            )
        else:
            fixed_buf = torch.empty(first_len, 1, _PP_H, dtype=torch.float32)  # only fits S=5
            # Recv of the [7,1,H] send into the [5,1,H] buffer: gloo enforces the
            # size and aborts the process. Reaching here without abort is the bug.
            pl._send_recv_pipeline(
                None, None, True, False, ps, (first_len, 1, _PP_H),
                fwd_recv_buf=fixed_buf, batch_p2p=False, dynamic_shape=False,
            )
        results.append((rank, "no_abort"))
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_dynamic_shape_variable_len_recv_gloo():
    """NEW: dynamic shape-over-wire sizes each recv buffer correctly (0 mismatch)."""
    import torch.multiprocessing as mp

    mgr = multiprocessing.Manager()
    results = mgr.list()
    mp.spawn(_dynamic_shape_worker, args=(2, 29663, results), nprocs=2, join=True)
    by_rank = dict(results)
    assert set(by_rank) == {0, 1}, f"missing ranks: {list(by_rank)}"
    recv_detail = by_rank[1]  # last stage recorded every fwd recv
    assert len(recv_detail) == len(_PP_VARLEN), recv_detail
    for s, shape_ok, bitwise_ok in recv_detail:
        assert shape_ok, f"S={s}: recv buffer not sized from the wire"
        assert bitwise_ok, f"S={s}: recv hidden not bitwise-equal to sent hidden"


@pytest.mark.distributed
def test_fixed_buffer_truncates_variable_len_gloo():
    """OLD: a first-mb-sized fixed buffer overflows on the larger mb -> abort (red)."""
    import torch.multiprocessing as mp

    mgr = multiprocessing.Manager()
    results = mgr.list()
    # gloo enforces the recv size mismatch by aborting the receiver, which
    # surfaces as a spawn failure. Green here (no abort) would mean silent
    # truncation — the exact corruption the Megatron way removes.
    with pytest.raises(Exception):
        mp.spawn(_fixed_buffer_worker, args=(2, 29664, results), nprocs=2, join=True)

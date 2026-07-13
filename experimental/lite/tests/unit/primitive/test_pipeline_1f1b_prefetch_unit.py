# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU schedule-invariant test for the 1F1B forward-input prefetch logic.

The real ``_1f1b_schedule`` allocates ``device="cuda"`` P2P buffers, so it cannot
run under a CPU unit test. Instead this test models the *forward-input receive
decision* exactly as the schedule drives it -- warmup prefetch + steady
self-receive -- and asserts the two invariants that must hold for the pipeline
not to hang or forward on a missing input:

1. every non-first stage forward consumes a valid (received) input tensor;
2. the number of forward tensors each stage sends equals the number the next
   stage receives (P2P balance), and every non-first stage receives exactly
   ``num_microbatches`` forward inputs in total.

It exercises the shipped condition ``(k + 1) < num_microbatches`` and pins the
pre-fix condition ``k < num_warmup - 1`` as *broken* for the penultimate stage,
including the ``num_microbatches < pp_size`` ("num_microbatch < warmup") case.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.mlite


def _simulate_forward_recv(pp_size: int, num_microbatches: int, *, prefix_fix: bool):
    """Replay the fwd-input receive/send decisions of ``_1f1b_schedule``.

    Returns (problems, sends, recvs). ``problems`` is empty iff the schedule is
    valid. ``prefix_fix=True`` uses the shipped ``(k + 1) < num_microbatches``
    condition; ``False`` uses the pre-fix ``k < num_warmup - 1`` condition.
    """
    problems: list[str] = []
    sends = [0] * pp_size
    recvs = [0] * pp_size

    for r in range(pp_size):
        is_first = r == 0
        is_last = r == pp_size - 1
        # Mirrors pipeline._1f1b_schedule: num_warmup is clamped by num_microbatches.
        num_warmup = min(pp_size - r - 1, num_microbatches)
        num_steady = num_microbatches - num_warmup
        has_input = False  # whether fwd_input currently holds a valid tensor

        # ── Warmup ──
        for k in range(num_warmup):
            if not is_first and k == 0:
                has_input = True
                recvs[r] += 1
            if not is_first and not has_input:
                problems.append(f"r{r} warmup k={k}: forward with no input")
            if prefix_fix:
                need_recv_next = not is_first and (k + 1) < num_microbatches
            else:
                need_recv_next = not is_first and k < num_warmup - 1
            if not is_last:
                sends[r] += 1
                has_input = need_recv_next
                if need_recv_next:
                    recvs[r] += 1
            elif need_recv_next:
                has_input = True
                recvs[r] += 1

        # ── Steady ──
        for k in range(num_steady):
            if not is_first and k == 0 and num_warmup == 0:
                has_input = True
                recvs[r] += 1
            if not is_first and not has_input:
                problems.append(f"r{r} steady k={k}: forward with no input")
            if not is_last:
                sends[r] += 1
            need_fwd = not is_first and k < num_steady - 1
            has_input = bool(need_fwd)
            if need_fwd:
                recvs[r] += 1

    # Adjacent-stage P2P balance: stage r's fwd sends == stage r+1's fwd recvs.
    for r in range(pp_size - 1):
        if sends[r] != recvs[r + 1]:
            problems.append(f"imbalance r{r}->r{r+1}: sends={sends[r]} recvs={recvs[r+1]}")
    # Every non-first stage must receive exactly one input per microbatch.
    for r in range(1, pp_size):
        if recvs[r] != num_microbatches:
            problems.append(f"r{r} total recvs={recvs[r]} != num_microbatches={num_microbatches}")

    return problems, sends, recvs


_CONFIGS = [
    (pp, nmb)
    for pp in (2, 3, 4, 8)
    for nmb in (1, 2, 3, 4, 8)
]


@pytest.mark.parametrize("pp_size,num_microbatches", _CONFIGS)
def test_shipped_condition_keeps_schedule_valid(pp_size, num_microbatches):
    problems, _sends, _recvs = _simulate_forward_recv(
        pp_size, num_microbatches, prefix_fix=True
    )
    assert problems == [], f"pp={pp_size} nmb={num_microbatches}: {problems}"


@pytest.mark.parametrize(
    "pp_size,num_microbatches",
    [(3, 2), (3, 8), (4, 2), (4, 3), (4, 8), (8, 4)],
)
def test_prefix_condition_is_broken_for_penultimate_stage(pp_size, num_microbatches):
    """The pre-fix `k < num_warmup - 1` misses the warmup->steady transition.

    Includes num_microbatches < pp_size cases (e.g. pp=4, nmb=2/3) -- the
    "num_microbatch < warmup" situation the audit was asked to check.
    """
    problems, _sends, _recvs = _simulate_forward_recv(
        pp_size, num_microbatches, prefix_fix=False
    )
    assert problems, (
        f"pre-fix condition unexpectedly passed for pp={pp_size} "
        f"nmb={num_microbatches}; the regression it guards may be gone"
    )


def test_all_warmup_no_extra_recv_when_no_steady():
    """When every microbatch runs in warmup/cooldown (num_steady == 0) the shipped
    condition must NOT prefetch an extra input past the last microbatch -- this
    matches Megatron's `if num_microbatches_remaining > 0` guard. A non-first,
    non-last stage with num_warmup == num_microbatches receives exactly nmb."""
    # pp=4, rank1: num_warmup=min(2,2)=2, num_steady=0.
    problems, sends, recvs = _simulate_forward_recv(4, 2, prefix_fix=True)
    assert problems == []
    # rank1 receives exactly num_microbatches, never over-receives.
    assert recvs[1] == 2


# ──────────────────────────────────────────────────────────────────────────
# Retained-input buffer-aliasing invariant
#
# ``_1f1b_schedule`` retains each received forward input in ``input_tensors``
# until its backward pass, but every recv reuses the single ``_fwd_recv_buf``.
# If the P2P helper returns that buffer directly (``clone_recv=False``), all
# retained inputs alias one storage, so a later ``irecv`` overwrites an earlier
# microbatch's activation/autograd leaf -> corrupted grads. The shipped fix
# passes ``clone_recv=True``, giving every microbatch a distinct retained
# tensor. This model replays the retain/pop order of the real schedule (warmup
# prefetch + steady 1F1B + cooldown drain) and checks that each input, when its
# backward runs, still holds the microbatch it was forwarded with.
# ──────────────────────────────────────────────────────────────────────────


class _Buf:
    """Mutable stand-in for the shared ``_fwd_recv_buf``; ``gen`` = last recv."""

    def __init__(self):
        self.gen = None


def _simulate_input_identity(pp_size: int, num_microbatches: int, *, clone: bool):
    """Replay input retention for every non-first stage; return corruption list.

    ``clone=True`` models ``clone_recv=True`` (retained input frozen at recv);
    ``clone=False`` models returning the shared buffer (retained input reads the
    buffer's live, later-overwritten value).
    """
    problems: list[str] = []

    for r in range(1, pp_size):  # only non-first stages retain forward inputs
        is_last = r == pp_size - 1
        num_warmup = min(pp_size - r - 1, num_microbatches)
        num_steady = num_microbatches - num_warmup

        buf = _Buf()
        state = {"recv_gen": 0, "fwd_gen": None, "fwd_stored": None}
        input_tensors: list[tuple[int, tuple]] = []

        def do_recv():
            state["recv_gen"] += 1
            g = state["recv_gen"]
            buf.gen = g
            state["fwd_gen"] = g
            state["fwd_stored"] = ("frozen", g) if clone else ("live", buf)

        def read(stored):
            kind, val = stored
            return val if kind == "frozen" else val.gen

        # ── Warmup ── (mirrors: capture current_input, forward, then recv-next)
        for k in range(num_warmup):
            if k == 0:
                do_recv()
            used_gen = state["fwd_gen"]
            current = state["fwd_stored"]  # captured BEFORE the recv-next below
            need_recv_next = (k + 1) < num_microbatches
            if not is_last:
                if need_recv_next:
                    do_recv()
            elif need_recv_next:
                do_recv()
            input_tensors.append((used_gen, current))

        # ── Steady ── (forward, retain, then backward on oldest, then recv-next)
        for k in range(num_steady):
            if k == 0 and num_warmup == 0:
                do_recv()
            used_gen = state["fwd_gen"]
            input_tensors.append((used_gen, state["fwd_stored"]))
            exp, stored = input_tensors.pop(0)
            got = read(stored)
            if got != exp:
                problems.append(f"r{r} steady k={k}: retained input mb{exp} read as mb{got}")
            if k < num_steady - 1:
                do_recv()

        # ── Cooldown ── (drain remaining backwards)
        for k in range(num_warmup):
            exp, stored = input_tensors.pop(0)
            got = read(stored)
            if got != exp:
                problems.append(f"r{r} cooldown k={k}: retained input mb{exp} read as mb{got}")

    return problems


@pytest.mark.parametrize("pp_size,num_microbatches", _CONFIGS)
def test_clone_recv_keeps_retained_inputs_uncorrupted(pp_size, num_microbatches):
    """With clone_recv (shipped), every retained input survives buffer reuse."""
    problems = _simulate_input_identity(pp_size, num_microbatches, clone=True)
    assert problems == [], f"pp={pp_size} nmb={num_microbatches}: {problems}"


@pytest.mark.parametrize(
    "pp_size,num_microbatches",
    [(3, 2), (3, 4), (4, 3), (4, 8), (8, 4)],
)
def test_missing_clone_recv_corrupts_retained_inputs(pp_size, num_microbatches):
    """Without clone_recv a middle stage's retained input is overwritten by a
    later recv into the shared buffer -- the defect the moe panel flagged.

    These configs all have a middle stage that retains an input across a
    subsequent forward recv (pp>=3, num_microbatches>=2)."""
    problems = _simulate_input_identity(pp_size, num_microbatches, clone=False)
    assert problems, (
        f"pp={pp_size} nmb={num_microbatches}: missing-clone aliasing no longer "
        f"corrupts retained inputs; the clone_recv guard may be gone"
    )


# ══════════════════════════════════════════════════════════════════════════
# Real-function CPU execution harness
#
# The tests above model the *decisions* of ``_1f1b_schedule``; the tests below
# execute the **actual** ``_1f1b_schedule`` -- its real warmup/steady/cooldown
# control flow, real forward/backward, real buffer reuse and real ``clone_recv``
# path -- one rank at a time on CPU. Two things are stubbed, nothing is
# re-implemented:
#   1. ``torch.empty(..., device="cuda")`` (the pre-allocated recv buffers) is
#      redirected to CPU via a thin shim, so the schedule needs no GPU.
#   2. ``_send_recv_pipeline`` is replaced by an in-process spy that mirrors the
#      real helper's recv tail exactly (write fresh data into the shared buffer,
#      then ``clone()`` iff ``clone_recv``) and records what the schedule asked
#      for. There is no live peer, so each rank runs independently and the spy
#      supplies the received tensor.
#
# This directly exercises: (a) the warmup->steady prefetch fix -- a missing
# prefetch would make the schedule forward a non-first stage on a ``None`` input;
# (b) the ``clone_recv=True`` fix -- with real cloning each retained microbatch
# input gets distinct storage, and the ``force_no_clone`` control shows the
# real schedule aliasing them when the clone is removed.
# ══════════════════════════════════════════════════════════════════════════
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

from megatron.lite.primitive.parallel import pipeline as _pl  # noqa: E402

_SHAPE = (2, 3)


def _mk_activation():
    """A differentiable leaf activation the schedule can send/retain/backward."""
    return torch.ones(_SHAPE, dtype=_pl._PIPELINE_TENSOR_DTYPE, requires_grad=True)


def _run_real_schedule_for_rank(pp_size, num_microbatches, rank, *, force_no_clone, mp):
    """Execute the real ``_1f1b_schedule`` for a single ``rank`` on CPU.

    Returns a record: fwd send/recv counts, the ``clone_recv`` flag seen on each
    fwd recv, the ``data_ptr`` of the fwd input each forward consumed, and
    whether any non-first forward ran on a ``None`` input.
    """
    is_first = rank == 0
    is_last = rank == pp_size - 1
    ps = SimpleNamespace(
        pp_size=pp_size, pp_rank=rank, pp_is_first=is_first, pp_is_last=is_last
    )
    rec = {"sends": 0, "recvs": 0, "clone_flags": [], "fwd_input_ptrs": [], "none_input": False}
    recv_n = {"v": 0}

    orig_empty = torch.empty

    def _empty_cpu(*a, **k):
        if k.get("device") == "cuda":
            k = {**k, "device": "cpu"}
        return orig_empty(*a, **k)

    mp.setattr(torch, "empty", _empty_cpu)

    def _spy_send_recv(
        send_fwd, send_bwd, recv_fwd, recv_bwd, ps_, tensor_shape,
        *, fwd_recv_buf=None, bwd_recv_buf=None, batch_p2p=True, clone_recv=False,
    ):
        if send_fwd is not None:
            rec["sends"] += 1
        fwd_buf = None
        bwd_buf = None
        if recv_fwd:
            rec["recvs"] += 1
            rec["clone_flags"].append(clone_recv)
            recv_n["v"] += 1
            fwd_buf = (
                fwd_recv_buf
                if fwd_recv_buf is not None
                else orig_empty(tensor_shape, dtype=_pl._PIPELINE_TENSOR_DTYPE, device="cpu")
            )
            # ``.data`` write mirrors irecv's raw-storage fill (bypasses autograd),
            # so reusing a grad-requiring buffer behaves like the real P2P recv.
            fwd_buf.data.fill_(float(recv_n["v"]))
            # Mirror _send_recv_pipeline's recv tail exactly.
            if clone_recv and not force_no_clone:
                fwd_buf = fwd_buf.clone()
            fwd_buf.grad = None
            fwd_buf.requires_grad_()
        if recv_bwd:
            bwd_buf = (
                bwd_recv_buf
                if bwd_recv_buf is not None
                else orig_empty(tensor_shape, dtype=_pl._PIPELINE_TENSOR_DTYPE, device="cpu")
            )
            if clone_recv and not force_no_clone:
                bwd_buf = bwd_buf.clone()
        return fwd_buf, bwd_buf

    mp.setattr(_pl, "_send_recv_pipeline", _spy_send_recv)

    def forward_step_fn(model, batch):  # first stage entry
        return {"hidden_states": _mk_activation()}

    class _MidModel:
        def __call__(self, hidden_states=None, position_ids=None, packed_seq_params=None, **kw):
            if hidden_states is None:
                rec["none_input"] = True
                hidden_states = _mk_activation()
            else:
                rec["fwd_input_ptrs"].append(hidden_states.data_ptr())
            return {"hidden_states": hidden_states * 1.0}  # keep it differentiable

    class _LastModel:
        def __call__(self, input_ids=None, hidden_states=None, **kw):
            if hidden_states is None:
                rec["none_input"] = True
                hidden_states = _mk_activation()
            else:
                rec["fwd_input_ptrs"].append(hidden_states.data_ptr())
            return {"loss": hidden_states.sum()}

    model = _LastModel() if is_last else _MidModel()
    data_iter = iter([{} for _ in range(num_microbatches)])
    _pl._1f1b_schedule(
        forward_step_fn, model, data_iter, num_microbatches,
        SimpleNamespace(), ps, _SHAPE,
        grad_sync_fn=None, pre_forward_hook=None, loss_fn=None,
    )
    return rec


_REAL_CONFIGS = [(3, 1), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 8), (8, 4)]


@pytest.mark.parametrize("pp_size,num_microbatches", _REAL_CONFIGS)
def test_real_1f1b_schedule_runs_clean_on_cpu(pp_size, num_microbatches, monkeypatch):
    """Running the actual ``_1f1b_schedule`` for every rank: no stage forwards on
    a None input, fwd send/recv counts balance across adjacent stages, and every
    fwd recv requests ``clone_recv=True``. Covers num_microbatches < pp_size."""
    recs = []
    for rank in range(pp_size):
        with monkeypatch.context() as m:
            recs.append(
                _run_real_schedule_for_rank(
                    pp_size, num_microbatches, rank, force_no_clone=False, mp=m
                )
            )
    for r in range(pp_size):
        assert not recs[r]["none_input"], f"pp={pp_size} nmb={num_microbatches} rank{r}: forward on None input"
    for r in range(pp_size - 1):
        assert recs[r]["sends"] == recs[r + 1]["recvs"], (
            f"pp={pp_size} nmb={num_microbatches}: r{r} sends={recs[r]['sends']} "
            f"!= r{r+1} recvs={recs[r+1]['recvs']}"
        )
    for r in range(1, pp_size):
        assert recs[r]["recvs"] == num_microbatches, (
            f"pp={pp_size} nmb={num_microbatches} rank{r}: recvs={recs[r]['recvs']}"
        )
        assert recs[r]["clone_flags"] and all(recs[r]["clone_flags"]), (
            f"pp={pp_size} nmb={num_microbatches} rank{r}: schedule did not request clone_recv on every fwd recv"
        )


@pytest.mark.parametrize("pp_size,num_microbatches", [(3, 2), (3, 4), (4, 3), (4, 8), (8, 4)])
def test_real_schedule_clone_gives_distinct_retained_inputs(pp_size, num_microbatches, monkeypatch):
    """A middle stage of the real schedule holds input k while the recv feeding
    input k+1 lands, so consecutive inputs are simultaneously live and must not
    alias. With the shipped ``clone_recv=True`` every consecutive pair differs.

    (Whole-run distinctness is *not* asserted: once a retained input is popped
    and backward-ed, its clone is freed and the allocator legitimately reuses
    that storage for a later, non-overlapping microbatch.)"""
    with monkeypatch.context() as m:
        rec = _run_real_schedule_for_rank(pp_size, num_microbatches, 1, force_no_clone=False, mp=m)
    ptrs = rec["fwd_input_ptrs"]
    assert len(ptrs) == num_microbatches
    collisions = [i for i in range(len(ptrs) - 1) if ptrs[i] == ptrs[i + 1]]
    assert not collisions, (
        f"pp={pp_size} nmb={num_microbatches}: consecutive live inputs share "
        f"storage at indices {collisions}; clone_recv should keep them distinct. ptrs={ptrs}"
    )


@pytest.mark.parametrize("pp_size,num_microbatches", [(3, 2), (4, 3), (8, 4)])
def test_real_schedule_without_clone_aliases_retained_inputs(pp_size, num_microbatches, monkeypatch):
    """Positive control: strip the clone from the transport and the real schedule
    aliases its retained inputs to the one shared buffer -- the corruption the
    ``clone_recv=True`` fix prevents. Proves the harness actually detects it."""
    with monkeypatch.context() as m:
        rec = _run_real_schedule_for_rank(pp_size, num_microbatches, 1, force_no_clone=True, mp=m)
    ptrs = rec["fwd_input_ptrs"]
    assert len(set(ptrs)) == 1, (
        f"pp={pp_size} nmb={num_microbatches}: without clone every retained input "
        f"should alias the one shared buffer, got data_ptrs {ptrs}"
    )

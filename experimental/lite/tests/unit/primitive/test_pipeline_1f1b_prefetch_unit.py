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

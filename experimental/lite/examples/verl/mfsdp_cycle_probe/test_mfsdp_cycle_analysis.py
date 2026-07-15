# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for the M-FSDP per-cycle retention analysis core.

No torch / CUDA / GPU: these exercise the arithmetic that turns raw per-cycle
samples and a torch memory snapshot into the task's answer, on synthetic inputs
with known ground truth. Run:

    python -m pytest experimental/lite/examples/verl/mfsdp_cycle_probe/test_mfsdp_cycle_analysis.py -q
"""
from __future__ import annotations

import math

from mfsdp_cycle_analysis import (
    StackRetention,
    combine_gold_standard,
    dedup_host_storage_entries,
    diff_live_stacks,
    per_cycle_retention_mib,
    retention_delta_mib,
    stack_retentions_from_summary,
    top_live_allocation_stacks,
)


def _samples(per_cycle_growth_mib, *, phase="woke", base=10000, cycles=20, warmup_jump=500):
    """Synthesize per-cycle rows: constant per-cycle growth after a warmup jump.

    Cycle 1 carries a one-off ``warmup_jump`` (CUDA context / autotune) on top of
    ``base``; from then on each cycle adds ``per_cycle_growth_mib``. A correct
    steady-state slope must recover ``per_cycle_growth_mib`` and ignore the jump.
    """
    rows = []
    for c in range(1, cycles + 1):
        jump = warmup_jump if c >= 1 else 0
        used = base + jump + per_cycle_growth_mib * c
        rows.append({"cycle": c, "phase": phase, "device_used_MiB": used})
    return rows


def test_slope_recovers_constant_per_cycle_growth():
    rows = _samples(37.0)
    res = per_cycle_retention_mib(rows, phase="woke", warmup_cycles=3)
    assert math.isclose(res["slope_mib_per_cycle"], 37.0, rel_tol=1e-6)
    assert res["n_points"] == 17  # cycles 4..20
    assert res["net_delta_mib"] > 0


def test_flat_arm_has_zero_slope_despite_warmup_jump():
    # A bounded (non-leaking) arm: big one-off jump at cycle 1, flat after.
    rows = _samples(0.0, warmup_jump=6000)
    res = per_cycle_retention_mib(rows, phase="woke", warmup_cycles=3)
    assert abs(res["slope_mib_per_cycle"]) < 1e-9
    assert abs(res["net_delta_mib"]) < 1e-9


def test_gold_standard_delta_is_mfsdp_minus_fsdp2():
    mfsdp = per_cycle_retention_mib(_samples(40.0), phase="woke")["slope_mib_per_cycle"]
    fsdp2 = per_cycle_retention_mib(_samples(2.0), phase="woke")["slope_mib_per_cycle"]
    assert math.isclose(retention_delta_mib(mfsdp, fsdp2), 38.0, rel_tol=1e-6)


def test_phase_filter_isolates_woke_from_asleep():
    rows = _samples(30.0, phase="woke") + _samples(0.0, phase="asleep")
    woke = per_cycle_retention_mib(rows, phase="woke")["slope_mib_per_cycle"]
    asleep = per_cycle_retention_mib(rows, phase="asleep")["slope_mib_per_cycle"]
    assert math.isclose(woke, 30.0, rel_tol=1e-6)
    assert abs(asleep) < 1e-9


def test_too_few_points_returns_zero_slope():
    rows = [{"cycle": 1, "phase": "woke", "device_used_MiB": 100}]
    res = per_cycle_retention_mib(rows, phase="woke", warmup_cycles=3)
    assert res["slope_mib_per_cycle"] == 0.0
    assert res["n_points"] == 0


# ── snapshot attribution ────────────────────────────────────────────────────


def _snapshot_new_style(export_bytes, other_bytes):
    """torch snapshot where blocks carry their own ``frames`` (recent torch)."""
    return {
        "segments": [
            {
                "blocks": [
                    {
                        "state": "active_allocated",
                        "size": export_bytes,
                        "frames": [
                            {"filename": "mfsdp/buffer.py", "line": 773, "name": "allocate_full"},
                            {"filename": "mfsdp/wrapper.py", "line": 210, "name": "materialize_all"},
                        ],
                    },
                    {
                        "state": "active_allocated",
                        "size": other_bytes,
                        "frames": [
                            {"filename": "adamw.py", "line": 42, "name": "step"},
                        ],
                    },
                    {
                        "state": "inactive",  # free block — must be ignored
                        "size": 999999,
                        "frames": [{"filename": "x.py", "line": 1, "name": "free"}],
                    },
                ]
            }
        ]
    }


def test_top_live_stacks_ranks_by_retained_bytes_and_skips_free():
    snap = _snapshot_new_style(export_bytes=8 * 1024 * 1024, other_bytes=2 * 1024 * 1024)
    stacks = top_live_allocation_stacks(snap, top_n=5)
    assert len(stacks) == 2  # the inactive block dropped
    assert stacks[0].retained_bytes == 8 * 1024 * 1024
    assert stacks[0].top_frame() == "mfsdp/buffer.py:773:allocate_full"
    assert math.isclose(stacks[0].retained_mib, 8.0)


def test_top_live_stacks_history_fallback_old_torch():
    # Older torch: block has no top-level frames, only a history[0].frames.
    snap = {
        "segments": [
            {
                "blocks": [
                    {
                        "state": "allocated",
                        "size": 4 * 1024 * 1024,
                        "history": [
                            {"frames": [{"filename": "mfsdp/buffer.py", "line": 773, "name": "alloc"}]}
                        ],
                    }
                ]
            }
        ]
    }
    stacks = top_live_allocation_stacks(snap, top_n=3)
    assert len(stacks) == 1
    assert stacks[0].top_frame() == "mfsdp/buffer.py:773:alloc"


def test_diff_live_stacks_surfaces_mfsdp_only_retention():
    mfsdp = [
        StackRetention(("mfsdp/buffer.py:773:alloc",), 8 * 1024 * 1024, 4),
        StackRetention(("adamw.py:42:step",), 2 * 1024 * 1024, 1),
    ]
    fsdp2 = [
        StackRetention(("adamw.py:42:step",), 2 * 1024 * 1024, 1),
    ]
    diff = diff_live_stacks(mfsdp, fsdp2)
    # the mfsdp-only buffer stack is the biggest positive delta
    assert diff[0][0] == "mfsdp/buffer.py:773:alloc"
    assert math.isclose(diff[0][1], 8.0)
    # the shared adam stack cancels to ~0
    shared = [d for d in diff if d[0] == "adamw.py:42:step"][0]
    assert abs(shared[1]) < 1e-9


# ── gold-standard A/B combine (mfsdp − fsdp2) ────────────────────────────────


def _arm_summary(optimizer, expandable, *, asleep_slope, top_frame, top_mib):
    """A minimal per-arm summary dict shaped like mfsdp_cycle_probe.py writes."""
    def _ret(slope):
        return {
            m: {"slope_mib_per_cycle": slope, "net_delta_mib": slope * 16, "n_points": 16}
            for m in ("device_used_MiB", "torch_reserved_MiB", "torch_alloc_MiB")
        }

    return {
        "tag": f"{optimizer}-exp{expandable}",
        "optimizer": optimizer,
        "expandable_segments": expandable,
        "retention": {"woke": _ret(slope=asleep_slope), "exported": _ret(slope=asleep_slope), "asleep": _ret(slope=asleep_slope)},
        "live_stacks": [
            {"frames": [top_frame], "retained_bytes": int(top_mib * 1024 * 1024), "num_blocks": 1},
        ],
    }


def test_combine_pairs_by_expandable_and_computes_mfsdp_minus_fsdp2():
    summaries = [
        _arm_summary("mfsdp", True, asleep_slope=40.0, top_frame="mfsdp/buffer.py:773:alloc", top_mib=8.0),
        _arm_summary("fsdp2", True, asleep_slope=3.0, top_frame="adamw.py:42:step", top_mib=2.0),
        _arm_summary("mfsdp", False, asleep_slope=35.0, top_frame="mfsdp/buffer.py:773:alloc", top_mib=6.0),
        _arm_summary("fsdp2", False, asleep_slope=2.0, top_frame="adamw.py:42:step", top_mib=2.0),
    ]
    combined = combine_gold_standard(summaries)
    pairs = {p["expandable_segments"]: p for p in combined["gold_standard_AB"]}
    assert set(pairs) == {True, False}
    # main axis: mfsdp − fsdp2 asleep slope
    assert math.isclose(pairs[True]["delta"]["asleep"]["device_used_MiB"]["delta_mib_per_cycle"], 37.0)
    assert math.isclose(pairs[False]["delta"]["asleep"]["device_used_MiB"]["delta_mib_per_cycle"], 33.0)
    # attribution: the mfsdp-only buffer stack tops the stack diff
    assert pairs[True]["stack_diff"][0][0] == "mfsdp/buffer.py:773:alloc"
    assert math.isclose(pairs[True]["stack_diff"][0][1], 8.0)
    assert combined["unpaired_arms"] == []


def test_combine_reports_unpaired_arm_instead_of_dropping_it():
    # fsdp2 only ran with expandable=True → the expandable=False mfsdp arm is unpaired.
    summaries = [
        _arm_summary("mfsdp", True, asleep_slope=40.0, top_frame="mfsdp/buffer.py:773:alloc", top_mib=8.0),
        _arm_summary("fsdp2", True, asleep_slope=3.0, top_frame="adamw.py:42:step", top_mib=2.0),
        _arm_summary("mfsdp", False, asleep_slope=35.0, top_frame="mfsdp/buffer.py:773:alloc", top_mib=6.0),
    ]
    combined = combine_gold_standard(summaries)
    assert [p["expandable_segments"] for p in combined["gold_standard_AB"]] == [True]
    assert combined["unpaired_arms"] == [
        {"optimizer": "mfsdp", "expandable_segments": False, "tag": "mfsdp-expFalse"}
    ]


def test_stack_retentions_roundtrip_from_summary():
    summary = _arm_summary("mfsdp", True, asleep_slope=40.0, top_frame="mfsdp/buffer.py:773:alloc", top_mib=8.0)
    stacks = stack_retentions_from_summary(summary)
    assert len(stacks) == 1
    assert stacks[0].top_frame() == "mfsdp/buffer.py:773:alloc"
    assert stacks[0].retained_bytes == 8 * 1024 * 1024


# ── host-RAM census (the TASK-1.13.8.6 host-leak axis) ──────────────────────


def test_dedup_host_storage_entries_counts_each_storage_once():
    # Two tensors alias storage ptr 100 (an offload buffer narrowed into views);
    # a distinct storage at 200. Aliases must collapse so host_tensor_MiB is real
    # resident bytes, not an aliasing-inflated sum.
    raw = [
        (100, 4096, (1024,)),
        (100, 4096, (512, 2)),  # same storage, different view shape
        (200, 8192, (2048,)),
    ]
    deduped = dedup_host_storage_entries(raw)
    assert len(deduped) == 2
    total = sum(nbytes for nbytes, _ in deduped)
    assert total == 4096 + 8192  # storage 100 counted once
    # first-seen shape retained for the aliased storage
    by_bytes = dict(deduped)
    assert by_bytes[4096] == (1024,)


def test_dedup_host_storage_entries_empty():
    assert dedup_host_storage_entries([]) == []


def test_host_rss_slope_recovers_monotone_leak():
    # The host-leak signature: rss_MiB climbs a constant amount every asleep
    # cycle. per_cycle_retention_mib is metric-generic, so the rss axis reports
    # the same steady-state slope the device axis does for GPU growth.
    rows = []
    for c in range(1, 21):
        rows.append({"cycle": c, "phase": "asleep", "rss_MiB": 160000 + 2500 * c})
    res = per_cycle_retention_mib(rows, phase="asleep", metric="rss_MiB", warmup_cycles=3)
    assert math.isclose(res["slope_mib_per_cycle"], 2500.0, rel_tol=1e-6)
    assert res["net_delta_mib"] > 0

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for the two-arm Qwen3 EP-overlap probe."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_LITE_ROOT = str(Path(__file__).resolve().parents[3])
sys.path = [path for path in sys.path if path != _LITE_ROOT]
sys.path.insert(0, _LITE_ROOT)


def test_overlap_probe_arms_differ_only_in_overlap_knob():
    from examples.bench import qwen3_ep_overlap_probe as probe

    baseline, overlap = probe.build_arm_configs()
    assert baseline["overlap_moe_expert_parallel_comm"] is False
    assert overlap["overlap_moe_expert_parallel_comm"] is True
    probe.assert_only_overlap_diff(baseline, overlap)


def test_overlap_probe_rejects_non_overlap_difference():
    from examples.bench.qwen3_ep_overlap_probe import assert_only_overlap_diff

    with pytest.raises(ValueError, match="use_deepep"):
        assert_only_overlap_diff(
            {"overlap_moe_expert_parallel_comm": False, "use_deepep": False},
            {"overlap_moe_expert_parallel_comm": True, "use_deepep": True},
        )


def test_overlap_probe_config_only_builds_both_runtime_configs():
    from examples.bench.qwen3_ep_overlap_probe import build_config_only_plans

    baseline, overlap = build_config_only_plans(hf_path="/tmp/hf")
    assert (
        baseline["runtime"]["backend_cfg"]["impl_cfg"][
            "overlap_moe_expert_parallel_comm"
        ]
        is False
    )
    assert (
        overlap["runtime"]["backend_cfg"]["impl_cfg"][
            "overlap_moe_expert_parallel_comm"
        ]
        is True
    )


def test_overlap_probe_reuses_shared_cycle_memory_harness():
    from examples.bench import cycle_memory_probe
    from examples.bench import qwen3_ep_overlap_probe as probe

    assert probe.sample_cuda_memory is cycle_memory_probe.sample_cuda_memory
    assert probe.live_allocation_stacks is cycle_memory_probe.live_allocation_stacks
    assert probe.per_cycle_retention is cycle_memory_probe.per_cycle_retention
    assert not hasattr(probe, "_sample_cuda_memory")
    assert not hasattr(probe, "_live_stacks")


def test_cycle_memory_retention_separates_fragmentation_from_inactive_split():
    from examples.bench.cycle_memory_probe import per_cycle_retention

    rows = [
        {
            "cycle": cycle,
            "phase": "after",
            "reserved_minus_allocated_bytes": 100 + 9 * cycle,
            "inactive_split_bytes": 10 + 7 * cycle,
        }
        for cycle in range(5)
    ]
    fragmentation = per_cycle_retention(
        rows, phase="after", metric="reserved_minus_allocated_bytes", warmup_cycles=1
    )
    inactive_split = per_cycle_retention(
        rows, phase="after", metric="inactive_split_bytes", warmup_cycles=1
    )
    assert fragmentation["slope_bytes_per_cycle"] == 9
    assert inactive_split["slope_bytes_per_cycle"] == 7

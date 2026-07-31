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
    from examples.bench.qwen3_ep_overlap_probe import (
        build_arm_configs,
        assert_only_overlap_diff,
    )

    baseline, overlap = build_arm_configs()
    assert baseline["overlap_moe_expert_parallel_comm"] is False
    assert overlap["overlap_moe_expert_parallel_comm"] is True
    assert_only_overlap_diff(baseline, overlap)


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

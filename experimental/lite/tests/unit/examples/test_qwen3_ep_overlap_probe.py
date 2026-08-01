# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for the two-arm Qwen3 EP-overlap probe."""

from __future__ import annotations

import json
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


def test_overlap_probe_selects_exactly_one_arm_per_process():
    from examples.bench import qwen3_ep_overlap_probe as probe

    baseline = probe.select_arm_config("baseline")
    overlap = probe.select_arm_config("overlap")
    assert baseline["overlap_moe_expert_parallel_comm"] is False
    assert overlap["overlap_moe_expert_parallel_comm"] is True
    assert baseline is not overlap


def test_overlap_probe_rejects_non_overlap_difference():
    from examples.bench.qwen3_ep_overlap_probe import assert_only_overlap_diff

    with pytest.raises(ValueError, match="use_deepep"):
        assert_only_overlap_diff(
            {"overlap_moe_expert_parallel_comm": False, "use_deepep": False},
            {"overlap_moe_expert_parallel_comm": True, "use_deepep": True},
        )


def test_overlap_probe_rejects_wrong_parallel_topology():
    from examples.bench import qwen3_ep_overlap_probe as probe

    with pytest.raises(ValueError, match="parallel topology"):
        probe.build_parallel_evidence(probe.BenchCliConfig(ep=4, tp=1, pp=1, cp=1))


def _write_qwen3_moe_config(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "num_hidden_layers": 48,
                "num_experts": 128,
                "num_experts_per_tok": 8,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_overlap_probe_config_only_builds_full_model_runtime_configs(tmp_path):
    from examples.bench.qwen3_ep_overlap_probe import build_config_only_plans

    baseline, overlap = build_config_only_plans(
        hf_path=str(_write_qwen3_moe_config(tmp_path / "hf"))
    )
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
    for plan in (baseline, overlap):
        assert plan["evidence_contract"]["model"] == {
            "num_hidden_layers": 48,
            "truncated": False,
            "num_experts": 128,
        }
        assert plan["evidence_contract"]["parallel"] == {
            "ep": 8,
            "tp": 1,
            "pp": 1,
            "cp": 1,
        }


def test_overlap_probe_rejects_truncated_model_evidence():
    from examples.bench.qwen3_ep_overlap_probe import validate_probe_summary

    with pytest.raises(ValueError, match="truncated"):
        validate_probe_summary(
            {
                "model": {
                    "num_hidden_layers": 2,
                    "truncated": True,
                    "num_experts": 128,
                },
                "parallel": {"ep": 8, "tp": 1, "pp": 1, "cp": 1},
                "rows": [],
            },
            warmup=1,
        )


def test_overlap_probe_rejects_missing_peak_memory_evidence():
    from examples.bench.qwen3_ep_overlap_probe import validate_probe_summary

    with pytest.raises(ValueError, match="peak memory"):
        validate_probe_summary(
            {
                "model": {
                    "num_hidden_layers": 48,
                    "truncated": False,
                    "num_experts": 128,
                },
                "parallel": {"ep": 8, "tp": 1, "pp": 1, "cp": 1},
                "rows": [
                    {
                        "cycle": cycle,
                        "phase": "after",
                        "step_ms": 10.0,
                        "peak_allocated_bytes": 100,
                        "peak_reserved_bytes": 200,
                    }
                    for cycle in (1, 2)
                ],
            },
            warmup=1,
        )


def test_overlap_probe_reuses_shared_cycle_memory_harness():
    from examples.bench import cycle_memory_probe
    from examples.bench import qwen3_ep_overlap_probe as probe

    assert probe.sample_cuda_memory is cycle_memory_probe.sample_cuda_memory
    assert probe.live_allocation_stacks is cycle_memory_probe.live_allocation_stacks
    assert probe.per_cycle_retention is cycle_memory_probe.per_cycle_retention
    assert not hasattr(probe, "_sample_cuda_memory")
    assert not hasattr(probe, "_live_stacks")


def test_overlap_probe_success_requires_all_rank_artifacts(tmp_path):
    from examples.bench.qwen3_ep_overlap_probe import require_probe_artifacts

    with pytest.raises(RuntimeError, match="probe evidence missing"):
        require_probe_artifacts(
            out_dir=tmp_path, arm="baseline", rank=3, cycles=2, warmup=1
        )

    expected = [
        tmp_path / "baseline-rank3.csv",
        tmp_path / "baseline-rank3-summary.json",
        tmp_path / "baseline-rank3-cycle1.pickle",
        tmp_path / "baseline-rank3-cycle2.pickle",
    ]
    for path in expected:
        path.write_bytes(b"evidence")

    assert (
        require_probe_artifacts(
            out_dir=tmp_path, arm="baseline", rank=3, cycles=2, warmup=1
        )
        == expected
    )


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

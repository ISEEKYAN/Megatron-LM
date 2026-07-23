"""Tests for the AC#4 three-arm experiment summarizer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/verl/scripts/summarize_three_arm_speed_memory.py"
)
_SPEC = importlib.util.spec_from_file_location("three_arm_speed_memory_summary", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SUMMARY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SUMMARY)


def _write_arm(root: Path, arm: str, filename: str, peak_memory: float) -> None:
    path = root / arm / filename
    path.parent.mkdir(parents=True)
    rows = [
        {
            "data": {
                "train/loss": 1.0 - 0.1 * step,
                "perf/max_memory_allocated_gb": peak_memory,
                "perf/max_memory_reserved_gb": peak_memory + 0.5,
                "train/mfu": 0.1,
            }
        }
        for step in range(2)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))


def test_main_fails_when_either_muon_arm_does_not_beat_adamw_memory(tmp_path, monkeypatch):
    _write_arm(tmp_path, "adamw", "q3moe_speed_mem_adamw.jsonl", 10.0)
    _write_arm(tmp_path, "muon", "q3moe_speed_mem_muon.jsonl", 11.0)
    _write_arm(tmp_path, "muon_fsdp2", "q3moe_speed_mem_muon_fsdp2.jsonl", 9.0)
    monkeypatch.setattr(sys, "argv", ["summary", "--run-root", str(tmp_path)])

    assert _SUMMARY.main() == 1


def test_main_passes_when_both_muon_arms_use_less_memory(tmp_path, monkeypatch):
    _write_arm(tmp_path, "adamw", "q3moe_speed_mem_adamw.jsonl", 10.0)
    _write_arm(tmp_path, "muon", "q3moe_speed_mem_muon.jsonl", 9.0)
    _write_arm(tmp_path, "muon_fsdp2", "q3moe_speed_mem_muon_fsdp2.jsonl", 8.0)
    monkeypatch.setattr(sys, "argv", ["summary", "--run-root", str(tmp_path)])

    assert _SUMMARY.main() == 0

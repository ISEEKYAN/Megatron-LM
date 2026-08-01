# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for shared per-cycle CUDA memory evidence helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_LITE_ROOT = str(Path(__file__).resolve().parents[3])
sys.path = [path for path in sys.path if path != _LITE_ROOT]
sys.path.insert(0, _LITE_ROOT)


def test_per_cycle_retention_rejects_a_single_post_warmup_sample():
    from examples.bench.cycle_memory_probe import per_cycle_retention

    rows = [{"cycle": 1, "phase": "after", "reserved_minus_allocated_bytes": 100}]

    with pytest.raises(ValueError, match="at least 2 samples"):
        per_cycle_retention(
            rows,
            phase="after",
            metric="reserved_minus_allocated_bytes",
            warmup_cycles=1,
        )


def test_per_cycle_retention_calculates_known_two_point_slope():
    from examples.bench.cycle_memory_probe import per_cycle_retention

    rows = [
        {"cycle": 2, "phase": "after", "allocated_bytes": 120},
        {"cycle": 5, "phase": "after", "allocated_bytes": 180},
    ]

    retention = per_cycle_retention(
        rows, phase="after", metric="allocated_bytes", warmup_cycles=1
    )

    assert retention["slope_bytes_per_cycle"] == 20
    assert retention["net_delta_bytes"] == 60
    assert retention["n_points"] == 2


def test_cuda_memory_sample_includes_allocated_and_reserved_peaks(monkeypatch):
    import torch
    from examples.bench.cycle_memory_probe import sample_cuda_memory

    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: {})
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 10)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 20)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 40)

    assert sample_cuda_memory() == {
        "allocated_bytes": 10,
        "reserved_bytes": 20,
        "peak_allocated_bytes": 30,
        "peak_reserved_bytes": 40,
        "reserved_minus_allocated_bytes": 10,
        "inactive_split_bytes": 0,
    }

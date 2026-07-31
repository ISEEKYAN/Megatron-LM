# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared per-cycle CUDA memory evidence helpers.

The sampling, snapshot fail-loud behavior, live-stack aggregation, and
least-squares retention calculation are ported from the validated per-cycle
memory harness. Workload adapters should only define their controlled arms and
lifecycle; they must not carry private copies of this evidence logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def record_memory_history(max_entries: int = 200000) -> None:
    import torch

    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", stacks="all", max_entries=max_entries
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"_record_memory_history failed: {exc}") from exc


def sample_cuda_memory() -> dict[str, int]:
    import torch

    stats = torch.cuda.memory_stats()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "reserved_minus_allocated_bytes": reserved - allocated,
        "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
    }


def dump_snapshot(path: Path) -> None:
    import torch

    try:
        torch.cuda.memory._dump_snapshot(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"_dump_snapshot failed for {path}: {exc}") from exc


def current_snapshot() -> dict[str, Any]:
    import torch

    try:
        return torch.cuda.memory._snapshot()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"_snapshot failed: {exc}") from exc


def live_allocation_stacks(
    snapshot: Mapping[str, object], top_n: int = 15
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[int]] = {}
    for segment in snapshot.get("segments", []) or []:
        for block in segment.get("blocks", []) or []:
            if block.get("state") not in {"active_allocated", "allocated"}:
                continue
            frames = block.get("frames") or []
            if not frames and (history := block.get("history") or []):
                frames = history[0].get("frames") or []
            key = tuple(
                f"{frame.get('filename', '?')}:{frame.get('line', '?')}:{frame.get('name', '?')}"
                for frame in frames
            )[:6]
            if not key:
                continue
            entry = grouped.setdefault(key, [0, 0])
            entry[0] += int(block.get("size", block.get("requested_size", 0)) or 0)
            entry[1] += 1
    return [
        {"frames": list(key), "retained_bytes": value[0], "num_blocks": value[1]}
        for key, value in sorted(
            grouped.items(), key=lambda item: item[1][0], reverse=True
        )[:top_n]
    ]


def _least_squares_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    return (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        / denominator
    )


def per_cycle_retention(
    samples: Iterable[Mapping[str, object]],
    *,
    phase: str,
    metric: str,
    warmup_cycles: int,
) -> dict[str, float | int | None]:
    rows = [
        row
        for row in samples
        if str(row.get("phase")) == phase and int(row["cycle"]) >= warmup_cycles
    ]
    rows.sort(key=lambda row: int(row["cycle"]))
    if len(rows) < 2:
        raise ValueError(
            "per-cycle retention requires at least 2 samples: "
            f"phase={phase!r}, metric={metric!r}, n_points={len(rows)}"
        )
    xs = [float(int(row["cycle"])) for row in rows]
    ys = [float(row[metric]) for row in rows]
    return {
        "slope_bytes_per_cycle": _least_squares_slope(xs, ys),
        "net_delta_bytes": ys[-1] - ys[0],
        "first_cycle": int(rows[0]["cycle"]),
        "last_cycle": int(rows[-1]["cycle"]),
        "n_points": len(rows),
    }


__all__ = [
    "current_snapshot",
    "dump_snapshot",
    "live_allocation_stacks",
    "per_cycle_retention",
    "record_memory_history",
    "sample_cuda_memory",
]

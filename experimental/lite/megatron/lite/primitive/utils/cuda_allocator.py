# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Env-gated CUDA workspace shape diagnostics."""

from __future__ import annotations

import os
from typing import Any

import torch

_WORKSPACE_SHAPE_STATS: dict[tuple[int, str, int], dict[str, Any]] = {}
_GIB = 1024**3


def cuda_allocator_metrics() -> dict[str, float | int]:
    """Snapshot allocator pressure so full-model runs expose fragmentation."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    stats = torch.cuda.memory_stats()
    return {
        "perf/cuda_memory_allocated_gb": allocated / _GIB,
        "perf/cuda_memory_reserved_gb": reserved / _GIB,
        "perf/cuda_reserved_minus_allocated_gb": (reserved - allocated) / _GIB,
        "perf/cuda_active_bytes_gb": stats.get("active_bytes.all.current", 0)
        / _GIB,
        "perf/cuda_inactive_split_bytes_gb": stats.get(
            "inactive_split_bytes.all.current", 0
        )
        / _GIB,
        "perf/cuda_inactive_split_peak_gb": stats.get(
            "inactive_split_bytes.all.peak", 0
        )
        / _GIB,
        "perf/cuda_segment_count": stats.get("segment.all.current", 0),
        "perf/cuda_active_block_count": stats.get("active.all.current", 0),
        "perf/cuda_inactive_split_block_count": stats.get(
            "inactive_split.all.current", 0
        ),
        "perf/cuda_num_alloc_retries": stats.get("num_alloc_retries", 0),
        "perf/cuda_num_ooms": stats.get("num_ooms", 0),
    }


def record_workspace_shape(
    *,
    device_index: int,
    scope: str,
    slot: int,
    dimensions: dict[str, int],
) -> None:
    if os.environ.get("MEGATRON_LITE_CUDA_WORKSPACE_SHAPE_METRICS") != "1":
        return
    key = (device_index, scope, slot)
    stats = _WORKSPACE_SHAPE_STATS.setdefault(
        key,
        {"calls": 0, "dimensions": {}},
    )
    stats["calls"] += 1
    for name, value in dimensions.items():
        dimension = stats["dimensions"].setdefault(
            name,
            {"min": value, "max": value, "unique": set()},
        )
        dimension["min"] = min(dimension["min"], value)
        dimension["max"] = max(dimension["max"], value)
        dimension["unique"].add(value)


def pop_workspace_shape_metrics() -> dict[str, int]:
    """Return and reset per-step shape-jitter diagnostics."""
    metrics = {}
    for (_device_idx, scope, slot), stats in _WORKSPACE_SHAPE_STATS.items():
        prefix = f"perf/workspace_{scope}_{slot}"
        metrics[f"{prefix}_calls"] = stats["calls"]
        for name, dimension in stats["dimensions"].items():
            metrics[f"{prefix}_{name}_min"] = dimension["min"]
            metrics[f"{prefix}_{name}_max"] = dimension["max"]
            metrics[f"{prefix}_{name}_span"] = dimension["max"] - dimension["min"]
            metrics[f"{prefix}_{name}_unique"] = len(dimension["unique"])
    _WORKSPACE_SHAPE_STATS.clear()
    return metrics


__all__ = [
    "cuda_allocator_metrics",
    "pop_workspace_shape_metrics",
    "record_workspace_shape",
]

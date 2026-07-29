# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Env-gated CUDA workspace shape diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

_WORKSPACE_SHAPE_STATS: dict[tuple[int, str, int], dict[str, Any]] = {}
_FIXED_CAPACITY_SCRATCH: dict[tuple[Any, ...], list["_FixedScratchSlot"]] = {}
_GIB = 1024**3


@dataclass
class _FixedScratchSlot:
    tensor: torch.Tensor
    in_use: bool = False
    event: Any | None = None
    stream_key: int | None = None


class _FixedScratchLease:
    def __init__(self, slot: _FixedScratchSlot, device: torch.device):
        self._slot = slot
        self._device = device
        self._active = True

    def release(self, *, stream: torch.cuda.Stream | None = None) -> None:
        if not self._active:
            return
        if self._slot.tensor.is_cuda:
            if stream is None:
                stream = torch.cuda.current_stream(self._device)
            event = torch.cuda.Event()
            event.record(stream)
            self._slot.event = event
            self._slot.stream_key = int(stream.cuda_stream)
        else:
            self._slot.event = None
            self._slot.stream_key = None
        self._slot.in_use = False
        self._active = False


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


def _slot_ready(slot: _FixedScratchSlot, device: torch.device) -> bool:
    if slot.event is None or not slot.tensor.is_cuda:
        return True
    stream = torch.cuda.current_stream(device)
    if slot.stream_key == int(stream.cuda_stream):
        return True
    return bool(slot.event.query())


def lease_fixed_capacity_scratch(
    *,
    scope: str,
    shape: tuple[int, ...] | torch.Size,
    capacity_rows: int,
    dtype: torch.dtype,
    device: torch.device,
    max_slots: int = 2,
) -> tuple[torch.Tensor, _FixedScratchLease]:
    """Lease one bounded scratch view whose cache key excludes dynamic rows."""
    shape = tuple(int(dim) for dim in shape)
    if not shape:
        raise ValueError("fixed-capacity scratch requires a non-empty shape")
    rows = shape[0]
    if capacity_rows <= 0:
        raise ValueError(f"fixed scratch capacity must be positive, got {capacity_rows}")
    if rows > capacity_rows:
        raise ValueError(
            f"fixed scratch rows {rows} exceed fixed capacity {capacity_rows}"
        )
    if max_slots <= 0:
        raise ValueError(f"fixed scratch max_slots must be positive, got {max_slots}")
    key = (
        _device_key(device),
        scope,
        dtype,
        shape[1:],
        int(capacity_rows),
    )
    slots = _FIXED_CAPACITY_SCRATCH.setdefault(key, [])
    wait_candidate = None
    for slot in slots:
        if slot.in_use:
            continue
        if _slot_ready(slot, device):
            slot.in_use = True
            slot.event = None
            view = slot.tensor[:rows].view(shape).detach()
            view.zero_()
            return view, _FixedScratchLease(slot, device)
        if wait_candidate is None:
            wait_candidate = slot
    if len(slots) < max_slots:
        slot = _FixedScratchSlot(
            torch.empty((capacity_rows, *shape[1:]), dtype=dtype, device=device),
            in_use=True,
        )
        slots.append(slot)
        view = slot.tensor[:rows].view(shape).detach()
        view.zero_()
        return view, _FixedScratchLease(slot, device)
    if wait_candidate is None:
        raise RuntimeError(
            f"fixed scratch {scope!r}: all {max_slots} slots are in use"
        )
    assert wait_candidate.event is not None
    torch.cuda.current_stream(device).wait_event(wait_candidate.event)
    wait_candidate.in_use = True
    wait_candidate.event = None
    wait_candidate.stream_key = None
    view = wait_candidate.tensor[:rows].view(shape).detach()
    view.zero_()
    return view, _FixedScratchLease(wait_candidate, device)


def fixed_capacity_scratch_metrics() -> dict[str, int | float]:
    """Snapshot retained fixed scratch bytes and finite-slot state."""
    bytes_by_scope: dict[str, int] = {}
    total_bytes = 0
    total_slots = 0
    slots_in_use = 0
    slots_not_ready = 0
    for key, slots in _FIXED_CAPACITY_SCRATCH.items():
        scope = str(key[1])
        for slot in slots:
            bytes_ = int(slot.tensor.numel() * slot.tensor.element_size())
            total_bytes += bytes_
            total_slots += 1
            bytes_by_scope[scope] = bytes_by_scope.get(scope, 0) + bytes_
            if slot.in_use:
                slots_in_use += 1
            elif slot.event is not None and not _slot_ready(slot, slot.tensor.device):
                slots_not_ready += 1
    metrics: dict[str, int | float] = {
        "perf/scratch_bytes": total_bytes,
        "perf/scratch_gb": total_bytes / _GIB,
        "perf/scratch_slots": total_slots,
        "perf/scratch_slots_in_use": slots_in_use,
        "perf/scratch_slots_not_ready": slots_not_ready,
    }
    for scope, bytes_ in sorted(bytes_by_scope.items()):
        metrics[f"perf/scratch_{scope}_bytes"] = bytes_
        metrics[f"perf/scratch_{scope}_gb"] = bytes_ / _GIB
    return metrics


def _reset_fixed_capacity_scratch_for_tests() -> None:
    _FIXED_CAPACITY_SCRATCH.clear()


def cuda_allocator_metrics() -> dict[str, float | int]:
    """Snapshot allocator pressure so full-model runs expose fragmentation."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    stats = torch.cuda.memory_stats()
    metrics: dict[str, float | int] = {
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
    metrics.update(fixed_capacity_scratch_metrics())
    return metrics


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
    "fixed_capacity_scratch_metrics",
    "lease_fixed_capacity_scratch",
    "pop_workspace_shape_metrics",
    "record_workspace_shape",
]

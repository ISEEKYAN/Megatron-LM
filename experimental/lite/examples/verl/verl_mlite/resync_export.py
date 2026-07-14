# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resync export memory policy (colocated actor ↔ vLLM).

During actor→rollout weight resync the M-FSDP export all-gathers the full dense
params into a transient buffer (≈tens of GiB/rank) and releases it once the
generator is drained. ``release_all()`` only returns that memory to torch's
caching allocator; the colocated vLLM ``wake_up`` uses a *separate* (cumem)
allocator, so unless the freed segments are handed back to the driver first,
vLLM cannot reclaim them and OOMs (``cumem_allocator.cpp`` on wake_up). We drain
the export with a threshold-batched ``empty_cache`` — flushing per ≥N GiB of
exported material and once more after the buffer releases — so the gather buffer
is returned to the driver *before* vLLM wakes.

This is an RL-colocation concern and deliberately lives in the verl integration
layer, never in the vLLM-agnostic megatron_lite export primitive (layering).
Kept dependency-free (stdlib only) so it is CPU unit-testable without a verl or
CUDA runtime. See TASK-1.13.8.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping

_RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV = "MLITE_RESYNC_EXPORT_EMPTY_CACHE_GIB"
_DEFAULT_RESYNC_EXPORT_EMPTY_CACHE_GIB = 4.0

# MEMCURVE instrumentation (default off; enabled only when either env is set).
# Records the true per-cycle resync export peak on *every* training rank so the
# transient full-model all-gather peak — which escapes the 5 s nvidia-smi device
# sampler and need not land on the sampled head node — can be attributed. Ported
# from the DS4 resync memory protocol (MLITE_RESYNC_MEMCURVE); see TASK-1.13.8.
_RESYNC_MEMCURVE_ENV = "MLITE_RESYNC_MEMCURVE"
_RESYNC_MEMLOG_PATH_ENV = "MLITE_RESYNC_MEMLOG_PATH"


def resync_export_empty_cache_threshold_bytes() -> int:
    """Bytes of exported material per ``empty_cache`` flush; ``<=0`` disables.

    Batched at ≥4 GiB by default (per the DS4 resync recipe) so small tensors
    never trigger a device sync. Override / disable via
    ``MLITE_RESYNC_EXPORT_EMPTY_CACHE_GIB`` (``0`` = off).
    """
    raw = os.environ.get(_RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV)
    if raw is None or raw == "":
        gib = _DEFAULT_RESYNC_EXPORT_EMPTY_CACHE_GIB
    else:
        try:
            gib = float(raw)
        except ValueError:
            gib = _DEFAULT_RESYNC_EXPORT_EMPTY_CACHE_GIB
    if gib <= 0:
        return 0
    return int(gib * (1024**3))


def stream_export_with_empty_cache(
    generator: Iterator, threshold_bytes: int, empty_cache_fn: Callable[[], None]
) -> Iterator:
    """Yield from ``generator``, flushing the allocator per ``threshold_bytes``.

    Fires ``empty_cache_fn`` once per ``threshold_bytes`` of cumulative exported
    tensor bytes, and once more after the generator drains (its own ``finally``
    has by then released the M-FSDP all-gather buffer), returning the freed
    memory to the driver before the colocated vLLM wakes. ``threshold_bytes<=0``
    disables all flushing (pass-through).
    """
    if threshold_bytes <= 0:
        yield from generator
        return
    accumulated = 0
    try:
        for name, tensor in generator:
            yield name, tensor
            accumulated += tensor.element_size() * tensor.nelement()
            if accumulated >= threshold_bytes:
                empty_cache_fn()
                accumulated = 0
    finally:
        empty_cache_fn()


# ── MEMCURVE: per-cycle resync export peak instrumentation ──────────────────
#
# The torch/CUDA reads (reset_peak_memory_stats, max_memory_allocated, …) stay
# in the engine where torch is already imported; this module keeps only the
# pure gating/formatting so it remains stdlib-only and CPU unit-testable.


def resync_memcurve_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True when MEMCURVE emission is requested (default off).

    Enabled by either ``MLITE_RESYNC_MEMCURVE`` (any non-empty value) or by
    setting ``MLITE_RESYNC_MEMLOG_PATH`` (a JSONL sink implies you want the
    curve). Off by default so production resync is unaffected.
    """
    env = os.environ if env is None else env
    return bool(env.get(_RESYNC_MEMCURVE_ENV) or env.get(_RESYNC_MEMLOG_PATH_ENV))


def resync_memcurve_memlog_path(env: Mapping[str, str] | None = None) -> str | None:
    """Optional JSONL sink path for the per-rank MEMCURVE record (``None`` = off)."""
    env = os.environ if env is None else env
    path = env.get(_RESYNC_MEMLOG_PATH_ENV)
    return path or None


def resync_memcurve_peak_gib(curve: list[dict], worst: dict) -> float:
    """Reported export peak (GiB).

    Peak stats are reset per exported tensor, so the coarse curve snapshots
    understate the transient export peak; the worst single-tensor
    ``max_allocated`` is the real lower bound and must dominate. Falls back to a
    snapshot's ``allocated_gib`` when ``max_allocated_gib`` is absent.
    """
    curve_peak = max(
        (s.get("max_allocated_gib", s.get("allocated_gib", 0.0)) for s in curve),
        default=0.0,
    )
    worst_gib = worst.get("peak_bytes", 0) / (1024**3)
    return max(curve_peak, worst_gib)


def format_resync_memcurve_line(rank: int, curve: list[dict], worst: dict) -> str:
    """Single-line stdout marker grep-able as ``MLITE_RESYNC_MEMCURVE``."""
    worst_gib = worst.get("peak_bytes", 0) / (1024**3)
    peak = resync_memcurve_peak_gib(curve, worst)
    summary = " ".join(f"{s['tag']}={s['allocated_gib']:.3f}" for s in curve)
    return (
        f"MLITE_RESYNC_MEMCURVE rank={rank} {summary} "
        f"worst_tensor={worst.get('name')} worst_tensor_peak_gib={worst_gib:.3f} "
        f"export_peak_max_alloc_gib={peak:.3f}"
    )


def resync_memcurve_record(rank: int, curve: list[dict], worst: dict) -> dict:
    """JSONL-serialisable per-rank MEMCURVE record for ``MLITE_RESYNC_MEMLOG_PATH``."""
    return {
        "rank": rank,
        "curve": curve,
        "worst_tensor": worst.get("name"),
        "worst_tensor_peak_gib": worst.get("peak_bytes", 0) / (1024**3),
        "export_peak_max_alloc_gib": resync_memcurve_peak_gib(curve, worst),
    }

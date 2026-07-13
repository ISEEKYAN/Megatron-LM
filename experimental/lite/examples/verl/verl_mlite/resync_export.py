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
from collections.abc import Callable, Iterator

_RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV = "MLITE_RESYNC_EXPORT_EMPTY_CACHE_GIB"
_DEFAULT_RESYNC_EXPORT_EMPTY_CACHE_GIB = 4.0


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

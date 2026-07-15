# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resync export release ordering (colocated actor ↔ vLLM).

During an actor→rollout weight resync the M-FSDP export all-gathers the full
dense params into a transient buffer and, in the param-offload config, is
reloaded to GPU for the walk. Once the generator drains, that memory must be
returned to the *driver* before the colocated vLLM ``wake_up`` runs: vLLM's
cumem allocator shares the physical device but is separate from torch's caching
allocator, so anything torch merely *cached* (the released all-gather buffer) or
left *resident* (the reloaded model) starves ``create_and_map`` and OOMs
(``cumem_allocator.cpp`` on wake_up).

Two thin generator wrappers enforce that ordering around the export stream:

* ``stream_export_with_empty_cache`` — hand the released all-gather buffer back
  to the driver by firing ``empty_cache`` per ≥N GiB of exported material and
  once more on drain.
* ``offload_params_after_export`` — reuse the engine's existing ``self.to(cpu)``
  offload lifecycle (the same call ``save_checkpoint`` / ``load_checkpoint``
  use), deferred to the generator's ``finally`` so the model returns to CPU
  exactly once the export is consumed — right before the vLLM wake_up. Doing it
  in ``finally`` keeps the ordering even if the consumer stops early or raises.

This is an RL-colocation concern and deliberately lives in the verl integration
layer, never in the vLLM-agnostic megatron_lite export primitive (layering).
Kept stdlib-only so the release-ordering invariant is CPU unit-testable without
a verl or CUDA runtime. See TASK-1.13.8.
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


def offload_params_after_export(
    streamed: Iterator, offload_fn: Callable[[], None], drain_fn: Callable[[], None]
) -> Iterator:
    """Yield from ``streamed``, then offload params + drain once it is consumed.

    ``get_per_tensor_param`` reloads the model to GPU for the export; in the
    param-offload config the steady state is model-on-CPU, so the reload must be
    undone once the caller finishes iterating — which is right before the
    colocated vLLM ``wake_up``. vLLM's separate (cumem) allocator shares the
    physical device, so a model left resident (plus any export-peak residue)
    starves ``create_and_map`` and OOMs. Undoing it in the generator's
    ``finally`` guarantees the ordering even if the consumer stops early or
    raises. ``offload_fn`` is the engine's existing ``self.to(cpu, model=True)``
    lifecycle (shared with save/load); ``drain_fn`` is ``empty_cache``.
    """
    try:
        yield from streamed
    finally:
        offload_fn()
        drain_fn()

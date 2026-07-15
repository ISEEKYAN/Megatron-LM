# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DoubleBufferAllocator.release_cached busy-buffer guard (exception-path fix).

The guard exists to catch genuine misuse on the normal path — releasing the
cache while a collective is legitimately in flight. But on an exception-driven
teardown (``full_parameter_context`` unwinding because the export consumer
raised, e.g. a downstream OOM) an aborted collective can leave a slot busy;
raising there would replace the primary error with a misleading "active buffers"
RuntimeError. ``force=True`` bypasses the guard so the cache is dropped and the
primary exception survives. CPU-only — see TASK-1.13.8.5.
"""
import pytest

from megatron.lite.primitive.optimizers.mfsdp.buffer import (
    DoubleBufferAllocator,
    NCCLUserBuffer,
)


def _allocator() -> DoubleBufferAllocator:
    # enabled=False → CPU-safe (no CUDA / Apex NCCL pool needed).
    ub = NCCLUserBuffer(enabled=False, groups=(), symmetric=True)
    return DoubleBufferAllocator(ub)


def test_release_cached_raises_when_a_buffer_is_busy_on_the_normal_path():
    alloc = _allocator()
    key = ("cuda", "bf16", 0)
    alloc._slots[key] = [object(), None]
    alloc._busy[key] = {0}  # simulate an in-flight collective
    with pytest.raises(RuntimeError, match="active M-FSDP communication buffers"):
        alloc.release_cached()
    # Guard fired before clearing: the cache is left intact for the caller.
    assert alloc._busy[key] == {0}


def test_force_release_cached_drops_the_cache_despite_a_busy_buffer():
    alloc = _allocator()
    key = ("cuda", "bf16", 0)
    alloc._slots[key] = [object(), None]
    alloc._busy[key] = {0}
    alloc._reuse_events[key] = [object(), None]
    alloc.release_cached(force=True)  # must not raise
    assert alloc._slots == {}
    assert alloc._busy == {}
    assert alloc._reuse_events == {}


def test_release_cached_clears_when_no_buffer_is_busy():
    alloc = _allocator()
    key = ("cuda", "bf16", 0)
    alloc._slots[key] = [object(), None]
    alloc._busy[key] = set()  # nothing in flight
    alloc.release_cached()  # force not needed; guard passes
    assert alloc._slots == {}
    assert alloc._busy == {}

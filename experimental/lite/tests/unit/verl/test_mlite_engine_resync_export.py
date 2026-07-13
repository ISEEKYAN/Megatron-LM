# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Resync export memory policy: threshold-batched empty_cache before vLLM wake.

Covers the pure helpers wired into ``get_per_tensor_param`` so the M-FSDP
all-gather buffer is returned to the driver before the colocated vLLM wakes
(the resync ``wake_up`` OOM fix). GPU/verl-free — see TASK-1.13.8.
"""
import torch

from verl_mlite.resync_export import (
    _DEFAULT_RESYNC_EXPORT_EMPTY_CACHE_GIB,
    _RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV,
    resync_export_empty_cache_threshold_bytes,
    stream_export_with_empty_cache,
)


def _pairs(sizes):
    # float32 → 4 bytes/element, so a 1-D tensor of n elems is 4n bytes.
    return [(f"w{i}", torch.zeros(n, dtype=torch.float32)) for i, n in enumerate(sizes)]


def test_stream_export_is_transparent_to_the_consumer():
    """Every (name, tensor) pair is yielded unchanged (export correctness)."""
    src = _pairs([1, 2, 3])
    out = list(stream_export_with_empty_cache(iter(src), 4, lambda: None))
    assert [n for n, _ in out] == ["w0", "w1", "w2"]
    for (_, got), (_, want) in zip(out, src):
        assert got is want


def test_flushes_once_per_threshold_plus_a_final_release():
    """empty_cache fires per ≥threshold bytes of exported material, and once more
    after the generator drains (the buffer-release-before-wake flush)."""
    calls = {"n": 0}

    def flush():
        calls["n"] += 1

    # Two 4-byte tensors per 8-byte threshold: flush after the 2nd tensor, then
    # again after the 4th, then a final drain flush → 3 total.
    src = _pairs([1, 1, 1, 1])
    list(stream_export_with_empty_cache(iter(src), 8, flush))
    assert calls["n"] == 3


def test_final_flush_happens_even_without_a_mid_stream_flush():
    """Below-threshold total still gets the buffer-release flush on drain."""
    calls = {"n": 0}
    src = _pairs([1])
    list(stream_export_with_empty_cache(iter(src), 1 << 30, lambda: calls.__setitem__("n", calls["n"] + 1)))
    assert calls["n"] == 1


def test_disabled_threshold_is_pass_through_without_flushing():
    calls = {"n": 0}
    src = _pairs([1, 2])
    out = list(stream_export_with_empty_cache(iter(src), 0, lambda: calls.__setitem__("n", calls["n"] + 1)))
    assert [n for n, _ in out] == ["w0", "w1"]
    assert calls["n"] == 0


def test_final_flush_fires_when_consumer_aborts_early():
    """Closing the wrapper mid-drain still returns the buffer to the driver."""
    calls = {"n": 0}
    src = _pairs([1, 1, 1, 1])
    gen = stream_export_with_empty_cache(iter(src), 1 << 30, lambda: calls.__setitem__("n", calls["n"] + 1))
    next(gen)  # consume one pair, then abandon
    gen.close()
    assert calls["n"] == 1


def test_threshold_defaults_to_four_gib(monkeypatch):
    monkeypatch.delenv(_RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV, raising=False)
    assert resync_export_empty_cache_threshold_bytes() == int(
        _DEFAULT_RESYNC_EXPORT_EMPTY_CACHE_GIB * (1024**3)
    )


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv(_RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV, "2")
    assert resync_export_empty_cache_threshold_bytes() == 2 * (1024**3)


def test_threshold_zero_disables(monkeypatch):
    monkeypatch.setenv(_RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV, "0")
    assert resync_export_empty_cache_threshold_bytes() == 0


def test_threshold_negative_disables(monkeypatch):
    monkeypatch.setenv(_RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV, "-1")
    assert resync_export_empty_cache_threshold_bytes() == 0


def test_threshold_bad_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(_RESYNC_EXPORT_EMPTY_CACHE_GIB_ENV, "not-a-number")
    assert resync_export_empty_cache_threshold_bytes() == int(
        _DEFAULT_RESYNC_EXPORT_EMPTY_CACHE_GIB * (1024**3)
    )

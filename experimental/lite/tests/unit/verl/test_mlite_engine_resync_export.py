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
    _RESYNC_MEMCURVE_ENV,
    _RESYNC_MEMLOG_PATH_ENV,
    format_resync_memcurve_line,
    resync_export_empty_cache_threshold_bytes,
    resync_memcurve_enabled,
    resync_memcurve_memlog_path,
    resync_memcurve_peak_gib,
    resync_memcurve_record,
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


# ── MEMCURVE instrumentation (pure gating/formatting) ──────────────────────


def test_memcurve_disabled_by_default():
    assert resync_memcurve_enabled({}) is False


def test_memcurve_enabled_by_flag_or_memlog_path():
    assert resync_memcurve_enabled({_RESYNC_MEMCURVE_ENV: "1"}) is True
    assert resync_memcurve_enabled({_RESYNC_MEMLOG_PATH_ENV: "/tmp/x.jsonl"}) is True
    # An empty value must not enable (matches the empty-cache env's semantics).
    assert resync_memcurve_enabled({_RESYNC_MEMCURVE_ENV: ""}) is False


def test_memlog_path_returns_none_when_unset_or_empty():
    assert resync_memcurve_memlog_path({}) is None
    assert resync_memcurve_memlog_path({_RESYNC_MEMLOG_PATH_ENV: ""}) is None
    assert resync_memcurve_memlog_path({_RESYNC_MEMLOG_PATH_ENV: "/a/b"}) == "/a/b"


def _curve():
    # export_begin low, a mid snapshot high, export_end back low — the coarse
    # snapshots miss the true transient peak captured per-tensor in `worst`.
    return [
        {"tag": "resync/enter", "allocated_gib": 9.6, "max_allocated_gib": 9.6},
        {"tag": "resync/export_begin", "allocated_gib": 9.6, "max_allocated_gib": 9.6},
        {"tag": "resync/export_end", "allocated_gib": 9.7, "max_allocated_gib": 40.0},
    ]


def test_peak_is_dominated_by_worst_single_tensor():
    # worst tensor peak (58 GiB) exceeds any coarse snapshot (40 GiB).
    worst = {"name": "layers.0.mlp", "peak_bytes": 58 * (1024**3)}
    assert resync_memcurve_peak_gib(_curve(), worst) == 58.0


def test_peak_falls_back_to_snapshot_when_worst_is_smaller():
    worst = {"name": None, "peak_bytes": 0}
    assert resync_memcurve_peak_gib(_curve(), worst) == 40.0


def test_format_line_is_grepable_and_reports_peak():
    worst = {"name": "layers.0.mlp", "peak_bytes": 58 * (1024**3)}
    line = format_resync_memcurve_line(3, _curve(), worst)
    assert line.startswith("MLITE_RESYNC_MEMCURVE rank=3 ")
    assert "resync/export_begin=9.600" in line
    assert "worst_tensor=layers.0.mlp" in line
    assert "export_peak_max_alloc_gib=58.000" in line


def test_record_is_jsonl_serialisable():
    import json

    worst = {"name": "layers.0.mlp", "peak_bytes": 58 * (1024**3)}
    rec = resync_memcurve_record(7, _curve(), worst)
    assert rec["rank"] == 7
    assert rec["worst_tensor"] == "layers.0.mlp"
    assert rec["export_peak_max_alloc_gib"] == 58.0
    json.dumps(rec)  # must not raise

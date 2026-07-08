# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_weight_update_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", types.ModuleType("ray"))
    path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "miles"
        / "miles_mlite"
        / "weight_update.py"
    )
    spec = importlib.util.spec_from_file_location("miles_mlite_weight_update", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_weight_export_requests_gpu_resident_fast_path(monkeypatch) -> None:
    weight_update = _load_weight_update_module(monkeypatch)
    captured = {}

    class NoPayloadUpdater(weight_update.RawHFWeightUpdater):
        @property
        def _needs_weight_payload(self) -> bool:
            return False

        def _iter_local_hf_weights(self, **export_kwargs):
            captured.update(export_kwargs)
            yield "weight", torch.ones(1)

    updater = object.__new__(NoPayloadUpdater)
    updater.args = SimpleNamespace(
        mlite_export_dtype="bfloat16",
        update_weight_buffer_size=4 * 1024**3,
    )

    assert list(updater._export_weight_chunks()) == []
    assert captured == {"cpu": False, "export_dtype": "bfloat16"}

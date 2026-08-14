# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit coverage for external prerequisites of the GLM5 DSA smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_dsa_smoke_module():
    path = (
        Path(__file__).parents[3]
        / "tests/smoke/model/test_glm5_dsa_acceptance_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("glm5_dsa_acceptance_smoke_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glm5_dsa_smoke_declares_flashmla_before_constructing_dsa(monkeypatch):
    module = _load_dsa_smoke_module()
    calls = []

    class FlashMLAMissing(Exception):
        pass

    def importorskip(name, *, reason):
        calls.append((name, reason))
        raise FlashMLAMissing

    monkeypatch.setattr(module.pytest, "importorskip", importorskip)

    with pytest.raises(FlashMLAMissing):
        module._make_dsa()

    assert calls == [
        (
            "flash_mla",
            "GLM5 DSA accept-with-proof needs FlashMLA sparse attention kernels.",
        )
    ]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from experimental.lite.tools.check_model_structure_fallbacks import check, scan


def test_model_structure_guard_requires_exact_interface_allowlist(tmp_path):
    source_root = tmp_path / "model"
    source_root.mkdir()
    (source_root / "sample.py").write_text(
        "def broken(model):\n"
        "    layers = getattr(model, 'layers', [])\n"
        "    return layers if hasattr(model, 'head') else []\n",
        encoding="utf-8",
    )

    callsites = scan(source_root, tmp_path)
    errors = check(callsites, {})

    assert len(callsites) == 2
    assert len(errors) == 2
    assert all("MLITE001" in error for error in errors)

    allowed = {
        site.signature: (
            1,
            "Test-only interface boundary with an exact stable signature.",
        )
        for site in callsites
    }
    assert check(callsites, allowed) == []

    allowed["model/sample.py:broken:hasattr:model:stale"] = (
        1,
        "Test-only stale interface boundary entry.",
    )
    assert check(callsites, allowed)[-1].startswith("MLITE002 stale allowlist entry")

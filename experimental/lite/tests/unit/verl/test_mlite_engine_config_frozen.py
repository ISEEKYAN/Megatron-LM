# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from types import MappingProxyType


CONFIG = (
    Path(__file__).resolve().parents[3] / "examples/verl/verl_mlite/engine/config.py"
)


def test_resync_fields_normalize_after_verl_base_freezes_config(monkeypatch) -> None:
    @dataclass
    class FrozenEngineConfig:
        _target_: str = ""

        def __post_init__(self) -> None:
            object.__setattr__(self, "_frozen_fields", frozenset(self.__dict__))

        def __setattr__(self, name, value) -> None:
            if name in getattr(self, "_frozen_fields", ()):
                raise FrozenInstanceError(f"Field {name!r} is frozen")
            object.__setattr__(self, name, value)

    verl = types.ModuleType("verl")
    workers = types.ModuleType("verl.workers")
    config = types.ModuleType("verl.workers.config")
    engine = types.ModuleType("verl.workers.config.engine")
    engine.EngineConfig = FrozenEngineConfig
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.workers", workers)
    monkeypatch.setitem(sys.modules, "verl.workers.config", config)
    monkeypatch.setitem(sys.modules, "verl.workers.config.engine", engine)

    spec = importlib.util.spec_from_file_location("_frozen_mlite_config", CONFIG)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    result = module.MegatronLiteEngineConfig(
        custom_backend_module=None,
        resync_format="vllm_checkpoint",
        resync_config=MappingProxyType({"expert_dtype": "fp8"}),
    )

    assert result.resync_format == "vllm_checkpoint"
    assert result.resync_config == {"expert_dtype": "fp8"}
    assert isinstance(result.resync_config, dict)

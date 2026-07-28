# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU behavior contracts for QAT and R3 across every MLite MoE model."""

from __future__ import annotations

import copy
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from megatron.lite.primitive.modules.router_replay import (
    attach_router_replay,
    detach_router_replay,
)
from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    apply_qat_to_chunks,
    normalize_qat_spec,
)

pytestmark = pytest.mark.mlite

MODEL_NAMES = ("qwen3_5", "qwen3_moe", "deepseek_v4", "glm5", "kimi_k2")


class _TinyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)
        self.router_replay = None


class _TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = nn.Module()
        self.moe.router = _TinyRouter()
        self.mlp = nn.Module()
        self.mlp.gate_up = nn.Linear(8, 16, bias=False)
        self.mlp.down = nn.Linear(16, 8, bias=False)


class _TinyModel(nn.Module):
    def __init__(self, model_name: str, *, mtp_enabled: bool):
        super().__init__()
        layers = [_TinyDecoderLayer(), _TinyDecoderLayer()]
        if model_name == "deepseek_v4":
            self.layers = nn.ModuleDict(
                {str(i): layer for i, layer in enumerate(layers)}
            )
        else:
            self.layers = nn.ModuleList(layers)
        if mtp_enabled:
            self.mtp = nn.ModuleList([_TinyDecoderLayer()])


class _TinyChunk(nn.Module):
    def __init__(self, model_name: str, *, mtp_enabled: bool):
        super().__init__()
        self.model = _TinyModel(model_name, mtp_enabled=mtp_enabled)


@dataclass(frozen=True)
class _ModelCase:
    name: str
    protocol: object
    chunk: _TinyChunk


def _protocol(model_name: str, transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    if model_name == "deepseek_v4":
        # The package __init__ eagerly imports the fused CSA model, which is not
        # needed for this CPU protocol contract. Load the real protocol and
        # checkpoint modules without importing a GPU-only model implementation.
        package_name = "megatron.lite.model.deepseek_v4.lite"
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(
                Path(__file__).parents[3]
                / "megatron"
                / "lite"
                / "model"
                / "deepseek_v4"
                / "lite"
            )
        ]
        monkeypatch.setitem(sys.modules, package_name, package)
    return importlib.import_module(f"megatron.lite.model.{model_name}.lite.protocol")


def _layers(chunk: _TinyChunk) -> list[nn.Module]:
    layers = chunk.model.layers
    return list(layers.values()) if isinstance(layers, nn.ModuleDict) else list(layers)


def _case(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
    *,
    mtp_enabled: bool,
) -> _ModelCase:
    return _ModelCase(
        name=model_name,
        protocol=_protocol(model_name, transformer_engine_import_stub, monkeypatch),
        chunk=_TinyChunk(model_name, mtp_enabled=mtp_enabled),
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("mtp_enabled", [False, True], ids=["mtp-off", "mtp-on"])
def test_every_model_replay_roots_are_exact_decoder_layers(
    model_name: str,
    mtp_enabled: bool,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=mtp_enabled,
    )
    expected = _layers(case.chunk)

    roots = case.protocol.router_replay_roots(case.chunk)

    assert roots == expected
    assert len(roots) == len(case.chunk.model.layers)
    assert roots != [case.chunk], (
        "falling back to the whole chunk would include MTP routers"
    )
    if mtp_enabled:
        mtp_modules = set(case.chunk.model.mtp.modules())
        assert all(root not in mtp_modules for root in roots)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_attaches_replay_only_to_decoder_router_count(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=True,
    )
    roots = case.protocol.router_replay_roots(case.chunk)

    count = sum(attach_router_replay(root, reset=False) for root in roots)
    try:
        assert count == len(case.chunk.model.layers)
        assert all(
            layer.moe.router.router_replay is not None for layer in _layers(case.chunk)
        )
        assert all(
            layer.moe.router.router_replay is None for layer in case.chunk.model.mtp
        )
    finally:
        for root in roots:
            detach_router_replay(root)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_qat_none_is_bitwise_inert(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=False,
    )
    implicit = copy.deepcopy(case.chunk)
    explicit = copy.deepcopy(case.chunk)

    implicit_cfg = case.protocol.ImplConfig()
    explicit_cfg = case.protocol.ImplConfig(qat=None)
    implicit_stats = apply_qat_to_chunks(
        [implicit], normalize_qat_spec(implicit_cfg.qat)
    )
    explicit_stats = apply_qat_to_chunks(
        [explicit], normalize_qat_spec(explicit_cfg.qat)
    )

    assert implicit_stats["quantized_modules"] == 0
    assert explicit_stats["quantized_modules"] == 0
    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    assert all(
        torch.equal(implicit_state[key], explicit_state[key]) for key in implicit_state
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_qat_quantizes_gate_up_but_not_router_gate(
    model_name: str,
    transformer_engine_import_stub,
    monkeypatch,
):
    case = _case(
        model_name,
        transformer_engine_import_stub,
        monkeypatch,
        mtp_enabled=False,
    )

    stats = apply_qat_to_chunks(
        [case.chunk],
        QATSpec(enabled=True, format="int8", group_size=-1),
    )

    assert stats["quantized_modules"] > 0
    assert stats["skipped_ignored"] > 0
    for layer in _layers(case.chunk):
        assert not parametrize.is_parametrized(layer.moe.router.gate, "weight")
        assert parametrize.is_parametrized(layer.mlp.gate_up, "weight")

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cheap protocol conformance checks shared by every registered Lite model."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.model.registry import TRAIN_RUNTIME_MODULES
from megatron.lite.primitive.recompute import apply_offload, apply_recompute


pytestmark = pytest.mark.mlite


@pytest.fixture(params=sorted(TRAIN_RUNTIME_MODULES.items()), ids=lambda item: item[0])
def protocol(request, transformer_engine_import_stub):
    """Load every registered protocol after installing the CPU-only TE stub."""
    transformer_engine_import_stub()
    runtime_name, module_name = request.param
    return runtime_name, importlib.import_module(module_name)


class _CountingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x.square()


def _core_attn_layer(runtime_name: str, target: nn.Module) -> nn.Module:
    """Provide the smallest real module path selected by each protocol map."""
    layer = nn.Module()
    if runtime_name == "deepseek_v4":
        layer.self_attn = target
    elif runtime_name == "glm5":
        layer.self_attention = SimpleNamespace(
            self_attention=target,
            core_attn=target,
            linear_proj=target,
        )
    elif runtime_name == "kimi_k2":
        layer.self_attention = SimpleNamespace(core_attn=target, linear_proj=target)
    elif runtime_name == "qwen3_5":
        layer.full_attn = SimpleNamespace(core_attn=target, proj=target)
    else:  # qwen3 and qwen3_moe share the same registered protocol.
        layer.attn = SimpleNamespace(core_attn=target, proj=target)
    return layer


def test_registered_protocol_wires_both_memory_features(protocol):
    runtime_name, module = protocol
    source = inspect.getsource(module.build_model)

    assert "apply_recompute(" in source, f"{runtime_name} omits recompute wiring"
    assert "apply_offload(" in source, f"{runtime_name} omits activation-offload wiring"
    assert "ModelBundle(" in source, f"{runtime_name} bypasses the startup feature audit"


def test_registered_protocol_core_attention_recompute_and_offload_are_observable(protocol):
    runtime_name, module = protocol
    recompute_target = _CountingModule()
    recompute_layer = _core_attn_layer(runtime_name, recompute_target)

    assert apply_recompute(nn.ModuleList([recompute_layer]), ["core_attn"], module.MODULE_MAP) == 1
    recompute_target(torch.tensor([2.0], requires_grad=True)).sum().backward()
    assert recompute_target.calls == 2
    assert recompute_target._megatron_lite_recompute_wrapped is True

    offload_target = _CountingModule()
    offload_layer = _core_attn_layer(runtime_name, offload_target)
    assert apply_offload(nn.ModuleList([offload_layer]), ["core_attn"], module.MODULE_MAP) == 1
    offload_target(torch.tensor([2.0], requires_grad=True)).sum().backward()
    assert offload_target._megatron_lite_offload_wrapped is True


def test_registered_protocol_expert_placement_is_sharded(protocol):
    runtime_name, module = protocol
    expert_name = "layers.0.moe.experts.0.fc1.weight"
    placements = module.PLACEMENT_FN(expert_name)

    assert module.EXPERT_CLASSIFIER(expert_name), f"{runtime_name} does not classify routed experts"
    assert len(placements) == 4
    assert any(type(placement).__name__ == "Shard" for placement in placements), (
        f"{runtime_name} replicates routed expert weights instead of sharding them"
    )

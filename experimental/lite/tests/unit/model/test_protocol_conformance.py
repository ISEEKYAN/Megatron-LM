# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cheap protocol conformance checks shared by every registered Lite model."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.model.registry import TRAIN_RUNTIME_MODULES
from megatron.lite.primitive.recompute import apply_offload


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


def test_activation_offload_is_explicitly_unsupported(protocol):
    runtime_name, module = protocol
    target = _CountingModule()
    layer = _core_attn_layer(runtime_name, target)

    with pytest.raises(NotImplementedError, match=r"no activation-offload backend"):
        apply_offload(nn.ModuleList([layer]), ["core_attn"], module.MODULE_MAP)

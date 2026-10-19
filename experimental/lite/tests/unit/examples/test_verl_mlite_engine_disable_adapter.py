# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""VERL computes LoRA reference log-probs with ``engine.disable_adapter()``."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

pytest.importorskip("verl", reason="VERL is required for this optional example test.")

from verl_mlite.engine.mlite_engine import MegatronLiteEngine

from megatron.lite.primitive.modules.lora import LinearLoRA, lora_active

pytestmark = [pytest.mark.mlite, pytest.mark.optional]


class _Chunk(nn.Module):
    """Base projection plus an optional adapter, wired like the GQA/expert call sites."""

    def __init__(self, seed: int, with_adapter: bool = True):
        super().__init__()
        torch.manual_seed(seed)
        self.base = nn.Linear(4, 4)
        self.lora = LinearLoRA(4, 4, rank=2, alpha=4, dropout=0.0) if with_adapter else None
        if self.lora is not None:
            with torch.no_grad():
                self.lora.lora_b.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if lora_active(self.lora):
            out = out + self.lora(x)
        return out


def _forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    for chunk in module:
        x = chunk(x)
    return x


def _engine(module: nn.Module) -> MegatronLiteEngine:
    engine = object.__new__(MegatronLiteEngine)
    engine.runtime = object()
    engine.handle = object()
    engine.module = module
    return engine


def _base_copy(policy: nn.ModuleList) -> nn.ModuleList:
    base = copy.deepcopy(policy)
    for chunk in base:
        chunk.lora = None
    return base


def test_disable_adapter_forward_equals_base_and_policy_returns_after_exit():
    policy = nn.ModuleList([_Chunk(0), _Chunk(1)])
    base = _base_copy(policy)
    engine = _engine(policy)
    x = torch.randn(3, 4)

    with torch.no_grad():
        policy_out = _forward(policy, x)
        base_out = _forward(base, x)
        assert not torch.equal(policy_out, base_out)

        with engine.disable_adapter():
            assert torch.equal(_forward(engine.module, x), base_out)
        assert torch.equal(_forward(engine.module, x), policy_out)

        with pytest.raises(ValueError, match="ref step failed"):
            with engine.disable_adapter():
                raise ValueError("ref step failed")
        assert torch.equal(_forward(engine.module, x), policy_out)


def test_disable_adapter_rejects_models_without_adapters():
    engine = _engine(nn.ModuleList([_Chunk(0, with_adapter=False)]))

    with pytest.raises(RuntimeError, match="no LoRA adapters"):
        with engine.disable_adapter():
            pass


def test_disable_adapter_requires_initialized_engine():
    engine = object.__new__(MegatronLiteEngine)
    engine.runtime = None
    engine.handle = None

    with pytest.raises(RuntimeError, match="not initialized"):
        with engine.disable_adapter():
            pass

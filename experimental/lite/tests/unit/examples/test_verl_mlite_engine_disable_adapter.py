# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""VERL reference forward bypasses adapters and restores policy state."""
import copy

import pytest
import torch
from torch import nn

from megatron.lite.primitive.modules.lora import LinearLoRA, lora_active

pytestmark = [pytest.mark.mlite, pytest.mark.optional]


class _Chunk(nn.Module):
    def __init__(self, adapter=True):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora = LinearLoRA(4, 4, rank=2) if adapter else None
        if adapter:
            nn.init.normal_(self.lora.lora_b)

    def forward(self, x):
        out = self.base(x)
        return out + self.lora(x) if lora_active(self.lora) else out


@pytest.fixture
def engine_type():
    return pytest.importorskip("verl_mlite.engine.mlite_engine").MegatronLiteEngine


def _engine(module, engine_type):
    engine = object.__new__(engine_type)
    engine.runtime, engine.handle, engine.module = object(), object(), module
    return engine


def test_disable_adapter_forward_equals_base_and_policy_returns_after_exit(engine_type):
    policy = nn.Sequential(_Chunk(), _Chunk())
    base = copy.deepcopy(policy)
    for chunk in base:
        chunk.lora = None
    engine, x = _engine(policy, engine_type), torch.randn(3, 4)
    with torch.no_grad():
        policy_out, base_out = policy(x), base(x)
        assert not torch.equal(policy_out, base_out)
        with engine.disable_adapter():
            assert torch.equal(engine.module(x), base_out)
        assert torch.equal(engine.module(x), policy_out)
        with pytest.raises(ValueError, match="ref step failed"):
            with engine.disable_adapter():
                raise ValueError("ref step failed")
        assert torch.equal(engine.module(x), policy_out)


def test_disable_adapter_rejects_models_without_adapters(engine_type):
    with pytest.raises(RuntimeError, match="no LoRA adapters"):
        with _engine(nn.Sequential(_Chunk(False)), engine_type).disable_adapter():
            pass


def test_disable_adapter_requires_initialized_engine(engine_type):
    engine = object.__new__(engine_type)
    engine.runtime = engine.handle = None
    with pytest.raises(RuntimeError, match="not initialized"):
        with engine.disable_adapter():
            pass

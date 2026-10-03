# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3-MoE as the consumer of the opt-in W4A8 routed-expert forward (CPU)."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization.qat import QATSpec

pytestmark = pytest.mark.mlite

LITE_ROOT = Path(__file__).resolve().parents[3]
W4A8_MODULE = "megatron.lite.primitive.quantization.w4a8_experts"
_CONFIG = SimpleNamespace(num_experts=3, hidden_size=128, moe_intermediate_size=128)


class CpuGroupedLinear(nn.Module):
    """TE GroupedLinear parameter layout (``weight0..weightN-1``) with a CPU forward."""

    def __init__(self, num_gemms, in_features, out_features, *, bias=False, **_):
        super().__init__()
        for index in range(num_gemms):
            weight = torch.randn(out_features, in_features) * 0.05
            self.register_parameter(f"weight{index}", nn.Parameter(weight))

    def forward(self, x, m_splits):
        outputs, start = [], 0
        for index, count in enumerate(m_splits):
            weight = getattr(self, f"weight{index}")
            outputs.append(x[start : start + count] @ weight.t())
            start += count
        return torch.cat(outputs)


@pytest.fixture
def experts_cls(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    from megatron.lite.primitive import transformer_engine as lite_te

    # Patch the TE module the wrapper bound at import, which may be an earlier stub.
    monkeypatch.setattr(lite_te._TE, "GroupedLinear", CpuGroupedLinear, raising=False)
    # The default path's SwiGLU is torch.compile'd; eager is enough on CPU here.
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    from megatron.lite.primitive.modules.experts import Experts

    return Experts


def _experts(experts_cls, seed=0, **kwargs):
    torch.manual_seed(seed)
    return experts_cls(_CONFIG, ParallelState(), **kwargs).to(torch.bfloat16)


def _routed_tokens(seed=1):
    torch.manual_seed(seed)
    m_splits = [4, 0, 9]
    x = torch.randn(sum(m_splits), _CONFIG.hidden_size).to(torch.bfloat16)
    return x, torch.tensor(m_splits), torch.rand(sum(m_splits))


def _spec(**overrides):
    return QATSpec(enabled=True, format="mxfp4", activation_bits=8, **overrides)


def test_enabled_experts_run_the_w4a8_primitive_on_master_weights(experts_cls):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts
    from megatron.lite.primitive.quantization.w4a8_experts import w4a8_expert_mlp

    experts = _experts(experts_cls)
    x, tokens_per_expert, probs = _routed_tokens()
    default_out = experts(x, tokens_per_expert, probs)

    assert enable_w4a8_experts([experts], _spec()) == 1
    out = experts(x, tokens_per_expert, probs)

    fc1 = [getattr(experts.fc1, f"weight{i}") for i in range(3)]
    fc2 = [getattr(experts.fc2, f"weight{i}") for i in range(3)]
    expected = w4a8_expert_mlp(x, fc1, fc2, [4, 0, 9], probs.unsqueeze(-1))
    assert torch.equal(out, expected)
    assert not torch.equal(out, default_out)


def test_w4a8_experts_train_the_bf16_master_weights(experts_cls):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts

    experts = _experts(experts_cls)
    enable_w4a8_experts([experts], _spec())
    x, tokens_per_expert, probs = _routed_tokens()
    experts(x, tokens_per_expert, probs).float().square().sum().backward()

    for index, has_tokens in enumerate((True, False, True)):
        for linear in (experts.fc1, experts.fc2):
            weight = getattr(linear, f"weight{index}")
            assert weight.dtype == torch.bfloat16 and weight.grad is not None
            assert bool(torch.count_nonzero(weight.grad)) == has_tokens


def test_enable_leaves_state_dict_bytes_unchanged(experts_cls):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts

    baseline = _experts(experts_cls).state_dict()
    experts = _experts(experts_cls)
    enable_w4a8_experts([experts], _spec())
    state = experts.state_dict()

    assert list(state) == list(baseline)
    assert all(
        torch.equal(state[k].view(torch.uint8), baseline[k].view(torch.uint8))
        for k in state
    )


def test_enable_respects_ignore_patterns_and_rejects_unreproduced_modes(experts_cls):
    from megatron.lite.primitive.modules.experts import enable_w4a8_experts

    root = nn.Module()
    root.experts = _experts(experts_cls)
    assert enable_w4a8_experts([root], _spec(ignore_patterns=("experts",))) == 0
    assert root.experts.w4a8 is False

    for kwargs, match in (
        ({"fp8": True}, "fp8"),
        ({"moe_act_recompute": True}, "recompute"),
        ({"lora_config": {"enabled": True, "rank": 2}}, "LoRA"),
    ):
        with pytest.raises(ValueError, match=match):
            enable_w4a8_experts([_experts(experts_cls, **kwargs)], _spec())


def _build(monkeypatch, experts_cls, qat):
    from megatron.lite.model.qwen3_moe.lite import protocol

    class Model(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = experts_cls(_CONFIG, ParallelState())

    monkeypatch.setattr(protocol, "Qwen3MoEModel", Model)
    monkeypatch.setattr(protocol, "init_parallel", lambda _p: ParallelState())
    monkeypatch.setattr(nn.Module, "cuda", lambda self: self)
    impl_cfg = protocol.ImplConfig(optimizer=None, qat=qat)
    bundle = protocol.build_model(
        SimpleNamespace(num_nextn_predict_layers=0), impl_cfg=impl_cfg
    )
    return bundle.chunks[0].layers[0].moe.experts


def test_qwen3_moe_build_model_opts_experts_into_w4a8(monkeypatch, experts_cls):
    a8 = {"enabled": True, "format": "mxfp4", "activation_bits": 8}
    assert _build(monkeypatch, experts_cls, a8).w4a8 is True
    assert _build(monkeypatch, experts_cls, {**a8, "enabled": False}).w4a8 is False
    weight_only = {"enabled": True, "format": "mxfp4"}
    assert _build(monkeypatch, experts_cls, weight_only).w4a8 is False
    assert _build(monkeypatch, experts_cls, None).w4a8 is False

    with pytest.raises(ValueError, match="matched no routed-expert module"):
        _build(monkeypatch, experts_cls, {**a8, "ignore_patterns": ["experts"]})


def test_default_experts_forward_never_imports_w4a8_module():
    script = "\n".join(
        [
            "import sys, types",
            "import torch",
            "import torch.nn as nn",
            "te_root = types.ModuleType('transformer_engine')",
            "te_pt = types.ModuleType('transformer_engine.pytorch')",
            "te_root.pytorch = te_pt",
            "sys.modules['transformer_engine'] = te_root",
            "sys.modules['transformer_engine.pytorch'] = te_pt",
            inspect.getsource(CpuGroupedLinear),
            "te_pt.GroupedLinear = CpuGroupedLinear",
            "from types import SimpleNamespace",
            "import megatron.lite.primitive.quantization",
            "from megatron.lite.primitive.modules.experts import Experts",
            "from megatron.lite.primitive.parallel import ParallelState",
            "cfg = SimpleNamespace(num_experts=2, hidden_size=128, moe_intermediate_size=128)",
            "experts = Experts(cfg, ParallelState()).to(torch.bfloat16)",
            "x = torch.randn(5, 128).to(torch.bfloat16)",
            "experts(x, torch.tensor([2, 3]), torch.rand(5))",
            f"assert {W4A8_MODULE!r} not in sys.modules",
            "experts.w4a8 = True",
            "experts(x, torch.tensor([2, 3]), torch.rand(5))",
            f"assert {W4A8_MODULE!r} in sys.modules",
        ]
    )
    env = {**os.environ, "MEGATRON_LITE_DISABLE_JIT_FUSER": "1"}
    subprocess.run([sys.executable, "-c", script], cwd=LITE_ROOT, env=env, check=True)

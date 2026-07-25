# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for apply_lora_to_chunks post-build applicator."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from megatron.lite.primitive.modules.lora import (
    LinearLoRA,
    LoraSpec,
    normalize_lora_spec,
    _weight_owner,
)
from megatron.lite.primitive.modules.lora_apply import (
    LoRAWrappedLinear,
    apply_lora_to_chunks,
)

pytestmark = pytest.mark.mlite


@pytest.fixture(autouse=True)
def _te_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


@pytest.fixture(autouse=True)
def _disable_torch_compile():
    import torch._dynamo

    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    yield
    torch._dynamo.config.disable = prev


def _swiglu_mlp():
    from megatron.lite.primitive.modules.mlp import SwiGLUMLP

    return SwiGLUMLP


def test_lora_spec_enabled_is_authoritative_not_rank():
    assert not normalize_lora_spec({"rank": 8}).enabled
    assert normalize_lora_spec({"enabled": True, "rank": 4}).enabled
    assert not normalize_lora_spec({"enabled": False, "rank": 8}).enabled


def test_disabled_apply_is_bit_identical():
    SwiGLUMLP = _swiglu_mlp()
    mlp = SwiGLUMLP(8, 16)
    chunk = nn.Module()
    chunk.mlp = mlp
    before = copy.deepcopy(chunk.state_dict())
    stats = apply_lora_to_chunks([chunk], LoraSpec(enabled=False, rank=8))
    assert stats["attached_modules"] == 0
    torch.testing.assert_close(chunk.state_dict(), before)


def test_apply_wraps_swiglu_and_freezes_base():
    SwiGLUMLP = _swiglu_mlp()
    mlp = SwiGLUMLP(8, 16)
    chunk = nn.Module()
    chunk.mlp = mlp
    spec = LoraSpec(enabled=True, rank=2, alpha=4)
    stats = apply_lora_to_chunks([chunk], spec)
    assert stats["attached_modules"] == 2
    assert isinstance(mlp.gate_up, LoRAWrappedLinear)
    assert isinstance(mlp.down, LoRAWrappedLinear)
    trainable = [n for n, p in chunk.named_parameters() if p.requires_grad]
    assert all("lora" in n.lower() for n in trainable)
    assert stats["trainable_tensors"] > 0
    assert stats["frozen_tensors"] > 0


class _IdentityWeightTransform(parametrize.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight * 1.0


def _apply_fake_qat_to_linears(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LoRAWrappedLinear):
            owner = _weight_owner(module.base)
        else:
            owner = _weight_owner(module)
        if owner is None or parametrize.is_parametrized(owner, "weight"):
            continue
        parametrize.register_parametrization(
            owner, "weight", _IdentityWeightTransform(), unsafe=True
        )


def test_qat_and_lora_four_combos():
    SwiGLUMLP = _swiglu_mlp()
    combos = [(False, False), (True, False), (False, True), (True, True)]
    outputs = []
    torch.manual_seed(42)
    base_state = SwiGLUMLP(8, 16).state_dict()
    x = torch.randn(4, 8)
    for qat_on, lora_on in combos:
        mlp = SwiGLUMLP(8, 16)
        mlp.load_state_dict(base_state)
        chunk = nn.Module()
        chunk.mlp = mlp
        if qat_on:
            _apply_fake_qat_to_linears(chunk)
        apply_lora_to_chunks([chunk], LoraSpec(enabled=lora_on, rank=2))
        if lora_on:
            for name, param in chunk.named_parameters():
                if "lora_b" in name:
                    param.data.fill_(0.25)
        with torch.no_grad():
            outputs.append(mlp(x).clone())

    # Identity QAT parametrization must not change the base forward.
    torch.testing.assert_close(outputs[0], outputs[1])
    assert not torch.allclose(outputs[0], outputs[2])
    assert outputs[3].shape == outputs[0].shape


def test_lora_wrapped_linear_forwards_base_attrs():
    base = nn.Linear(4, 3, bias=False)
    base.weight.data.fill_(0.5)
    adapter = LinearLoRA(4, 3, rank=2, alpha=2, dropout=0.0)
    wrapped = LoRAWrappedLinear(base, adapter)
    assert wrapped.weight is base.weight
    x = torch.randn(2, 4)
    expected = base(x) + adapter(x)
    torch.testing.assert_close(wrapped(x), expected)


def test_apply_lora_tp_layout_on_gqa_adapters():
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.lora_apply import _attach_gqa_proj, _attach_gqa_qkv

    class _MockLinearSurface:
        def __init__(self, local_in: int, local_out: int):
            self.local_in = local_in
            self.local_out = local_out
            self.use_sp = False
            self.linear = SimpleNamespace()

    class _MockAttn:
        def __init__(self):
            self.ps = SimpleNamespace(tp_group=object(), tp_size=2, tp_rank=0)
            self.qkv = _MockLinearSurface(16, 24)
            self.proj = _MockLinearSurface(24, 16)

    attn = _MockAttn()
    spec = LoraSpec(enabled=True, rank=4, target_modules=("linear_qkv", "linear_proj"))
    assert _attach_gqa_qkv(attn, spec)
    assert _attach_gqa_proj(attn, spec)
    qkv_adapter = attn.qkv.adapter
    proj_adapter = attn.proj.adapter
    assert qkv_adapter.rank_partitioned_a is True
    assert qkv_adapter.lora_a.tensor_model_parallel is True
    assert qkv_adapter.lora_b.tensor_model_parallel is True
    assert proj_adapter.input_parallel_reduce is True
    assert proj_adapter.output_partitioned_b is True

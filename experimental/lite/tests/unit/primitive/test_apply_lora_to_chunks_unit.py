# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU unit tests for apply_lora_to_chunks post-build applicator."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from megatron.lite.primitive.modules.lora import (
    LinearLoRA,
    LoraSpec,
    _weight_owner,
    normalize_lora_spec,
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


def test_qwen_protocol_loads_canonical_weights_before_lora_attach(monkeypatch):
    """Regression for HF load seeing ``.base.`` names after early LoRA attach."""
    from megatron.lite.model.qwen3_moe.lite import protocol

    SwiGLUMLP = _swiglu_mlp()

    class _CpuQwenProxy(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].mlp = SwiGLUMLP(8, 16)

        def cuda(self):
            return self

    monkeypatch.setattr(protocol, "Qwen3MoEModel", _CpuQwenProxy)
    monkeypatch.setattr(
        protocol,
        "init_parallel",
        lambda _cfg: SimpleNamespace(
            tp_size=1, ep_size=1, etp_size=1, pp_size=1, cp_size=1
        ),
    )
    monkeypatch.setattr(protocol, "set_cross_entropy_fusion", lambda *_args: None)

    cfg = SimpleNamespace(
        router_aux_loss_coef=0.0,
        num_nextn_predict_layers=0,
        mtp_loss_scaling_factor=0.1,
    )
    bundle = protocol.build_model(
        cfg,
        impl_cfg=protocol.ImplConfig(
            optimizer=None,
            lora={
                "enabled": True,
                "rank": 2,
                "target_modules": ("linear_fc1", "linear_fc2"),
            },
        ),
    )
    chunk = bundle.chunks[0]
    mlp = chunk.layers[0].mlp

    # Canonical checkpoint names must still exist at load time.
    assert not isinstance(mlp.gate_up, LoRAWrappedLinear)
    checkpoint = {
        name: torch.full_like(param, 0.25) for name, param in chunk.state_dict().items()
    }
    incompatible = chunk.load_state_dict(checkpoint, strict=False)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        base_t0 = mlp(x).clone()

    updates = bundle.extras["post_model_load_hook"]()
    assert updates["extras"]["lora_stats"]["attached_modules"] == 2
    assert isinstance(mlp.gate_up, LoRAWrappedLinear)
    assert isinstance(mlp.down, LoRAWrappedLinear)
    assert torch.count_nonzero(mlp.gate_up.adapter.lora_b) == 0
    assert torch.count_nonzero(mlp.down.adapter.lora_b) == 0
    with torch.no_grad():
        lora_t0 = mlp(x)
    assert torch.equal(lora_t0, base_t0)


def test_qwen3_30b_lora_attachment_and_parameter_contract():
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig

    cfg = Qwen3MoEConfig()
    rank = 16
    target_ep = 8
    targets = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")

    assert cfg.num_hidden_layers * len(targets) == 192
    qkv_out = (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim
    attention_lora = (
        cfg.num_hidden_layers
        * rank
        * (cfg.hidden_size + qkv_out + cfg.hidden_size + cfg.hidden_size)
    )
    expert_lora = (
        cfg.num_hidden_layers
        * target_ep
        * rank
        * (
            (cfg.hidden_size + 2 * cfg.moe_intermediate_size)
            + (cfg.moe_intermediate_size + cfg.hidden_size)
        )
    )
    trainable_numel = attention_lora + expert_lora
    total_numel = 30_532_122_624

    assert trainable_numel == 47_972_352
    assert trainable_numel / total_numel == pytest.approx(0.0015712092012329002)


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
    from megatron.lite.primitive.modules.lora_apply import (
        _attach_gqa_proj,
        _attach_gqa_qkv,
    )
    from megatron.lite.primitive.parallel.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    def _column_surface():
        surface = ColumnParallelLinear.__new__(ColumnParallelLinear)
        nn.Module.__init__(surface)
        surface.tp_size = 2
        surface.tp_rank = 0
        surface.tp_group = object()
        surface.local_out = 12
        surface.use_sp = True
        surface.linear = nn.Linear(16, 12, bias=False, dtype=torch.bfloat16)
        surface.gather_output = False
        return surface

    def _row_surface():
        surface = RowParallelLinear.__new__(RowParallelLinear)
        nn.Module.__init__(surface)
        surface.tp_size = 2
        surface.tp_rank = 0
        surface.tp_group = object()
        surface.local_in = 12
        surface.use_sp = True
        surface.linear = nn.Linear(12, 16, bias=False, dtype=torch.bfloat16)
        return surface

    class _MockAttn:
        def __init__(self):
            from types import SimpleNamespace

            self.ps = SimpleNamespace(tp_group=object(), tp_size=2, tp_rank=0)
            self.qkv = _column_surface()
            self.proj = _row_surface()

    attn = _MockAttn()
    spec = LoraSpec(enabled=True, rank=4, target_modules=("linear_qkv", "linear_proj"))
    assert _attach_gqa_qkv(attn, spec)
    assert _attach_gqa_proj(attn, spec)
    assert attn.qkv.weight is attn.qkv.base.linear.weight
    assert attn.proj.weight is attn.proj.base.linear.weight
    qkv_adapter = attn.qkv.adapter
    proj_adapter = attn.proj.adapter
    assert qkv_adapter.lora_a.dtype == torch.bfloat16
    assert proj_adapter.lora_a.dtype == torch.bfloat16
    assert qkv_adapter.lora_a.shape == (2, 16)
    assert qkv_adapter.lora_b.shape == (12, 4)
    assert qkv_adapter.rank_partitioned_a is True
    assert qkv_adapter.lora_a.tensor_model_parallel is True
    assert qkv_adapter.lora_b.tensor_model_parallel is True
    assert proj_adapter.lora_a.shape == (4, 12)
    assert proj_adapter.lora_b.shape == (8, 4)
    assert proj_adapter.input_parallel_reduce is True
    assert proj_adapter.output_partitioned_b is True

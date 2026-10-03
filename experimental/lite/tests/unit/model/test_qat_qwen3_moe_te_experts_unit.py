# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Weight-only MXFP4 QAT must cover TE ``GroupedLinear`` expert weights.

TE ``GroupedLinear`` registers one ``weight{i}`` parameter per local expert
instead of a single ``weight``. These tests build the real ``Qwen3MoEModel``
with CPU stand-ins that keep TE's parameter names and per-GEMM weight access,
then check coverage, forward numerics against a W4A16 reference and pairing
with the MXFP4 rollout export.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

pytestmark = pytest.mark.mlite

_LAYERS = 2
_EXPERTS = 4
_HIDDEN = 64
_INTERMEDIATE = 32


class _TELinear(nn.Linear):
    def __init__(self, in_features, out_features, *, bias=True, return_bias=False, **_):
        super().__init__(in_features, out_features, bias=bias)
        self.return_bias = return_bias


class _TELayerNormLinear(nn.Module):
    def __init__(self, in_features, out_features, *, bias=True, **_):
        super().__init__()
        self.layer_norm_weight = nn.Parameter(torch.ones(in_features))
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None


class _TERMSNorm(nn.Module):
    def __init__(self, hidden_size, **_):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))


class _TEDotProductAttention(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()


class _TEGroupedLinear(nn.Module):
    """TE ``GroupedLinear`` layout: ``weight0..N-1`` read per GEMM in forward."""

    def __init__(self, num_gemms, in_features, out_features, *, bias=False, **_):
        super().__init__()
        self.num_gemms = num_gemms
        for index in range(num_gemms):
            self.register_parameter(
                f"weight{index}",
                nn.Parameter(torch.randn(out_features, in_features) * 0.05),
            )

    def forward(self, x, m_splits):
        weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        chunks = torch.split(x, list(m_splits))
        return torch.cat([c @ w.t() for c, w in zip(chunks, weights, strict=True)])


@pytest.fixture
def tiny_qwen3_moe(transformer_engine_import_stub, monkeypatch):
    transformer_engine_import_stub()
    # Run the torch.compile'd SwiGLU helpers eagerly; numerics are unchanged.
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    from megatron.lite.primitive import transformer_engine as mlite_te

    for name, cls in {
        "Linear": _TELinear,
        "LayerNormLinear": _TELayerNormLinear,
        "RMSNorm": _TERMSNorm,
        "DotProductAttention": _TEDotProductAttention,
        "GroupedLinear": _TEGroupedLinear,
    }.items():
        monkeypatch.setattr(mlite_te, name, cls, raising=False)

    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite.model import Qwen3MoEModel
    from megatron.lite.primitive.parallel import ParallelState

    config = Qwen3MoEConfig(
        num_hidden_layers=_LAYERS,
        hidden_size=_HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=64,
        num_experts=_EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=_INTERMEDIATE,
        max_position_embeddings=16,
        layer_types=["full_attention"] * _LAYERS,
    )
    ps = ParallelState()

    def build():
        torch.manual_seed(0)
        return Qwen3MoEModel(config, ps, use_deepep=False).to(torch.bfloat16)

    return build, config, ps


def _grouped_linears(model):
    for layer in model.layers:
        yield layer.moe.experts.fc1
        yield layer.moe.experts.fc2


def _expert_weight_names(fc):
    return [f"weight{i}" for i in range(fc.num_gemms)]


def test_mxfp4_qat_fake_quantizes_every_te_expert_weight(tiny_qwen3_moe):
    from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks

    build, _config, _ps = tiny_qwen3_moe
    model = build()
    stats = apply_qat_to_chunks([model], QATSpec(enabled=True, format="mxfp4"))

    covered = sum(
        parametrize.is_parametrized(fc, name)
        for fc in _grouped_linears(model)
        for name in _expert_weight_names(fc)
    )
    assert covered == _LAYERS * _EXPERTS * 2
    # qkv + proj + fc1 + fc2 per layer.
    assert stats["quantized_modules"] == _LAYERS * 4
    for layer in model.layers:
        assert not parametrize.is_parametrized(layer.moe.router.gate, "weight")


def test_mxfp4_qat_expert_load_targets_resolve_to_master(tiny_qwen3_moe):
    from megatron.lite.primitive.ckpt.hf_weights import _resolve_param_name
    from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks

    build, _config, _ps = tiny_qwen3_moe
    model = build()
    apply_qat_to_chunks([model], QATSpec(enabled=True, format="mxfp4"))
    state = model.state_dict()
    for layer_idx in range(_LAYERS):
        for fc in ("fc1", "fc2"):
            for expert in range(_EXPERTS):
                logical = f"layers.{layer_idx}.moe.experts.{fc}.weight{expert}"
                actual = _resolve_param_name(logical, state)
                assert actual.endswith(f"parametrizations.weight{expert}.original")


def test_mxfp4_qat_expert_forward_matches_w4a16_reference(tiny_qwen3_moe):
    from megatron.lite.primitive.quantization.mxfp4 import (
        dequantize_mxfp4,
        quantize_mxfp4,
    )
    from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks

    build, _config, _ps = tiny_qwen3_moe
    model = build()
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for fc in _grouped_linears(reference):
            for name in _expert_weight_names(fc):
                weight = getattr(fc, name)
                weight.copy_(dequantize_mxfp4(*quantize_mxfp4(weight)).to(weight.dtype))
    apply_qat_to_chunks([model], QATSpec(enabled=True, format="mxfp4"))

    tokens_per_expert = torch.tensor([3, 0, 5, 2])
    torch.manual_seed(1)
    x = torch.randn(int(tokens_per_expert.sum()), _HIDDEN, dtype=torch.bfloat16)
    probs = torch.rand(x.shape[0])
    for layer, ref_layer in zip(model.layers, reference.layers, strict=True):
        out = layer.moe.experts(x, tokens_per_expert, probs)
        ref = ref_layer.moe.experts(x, tokens_per_expert, probs)
        assert torch.equal(out, ref)

    model.layers[0].moe.experts(x, tokens_per_expert, probs).float().sum().backward()
    fc1 = model.layers[0].moe.experts.fc1
    master = fc1.parametrizations.weight0.original
    assert isinstance(master, nn.Parameter)
    assert master.grad is not None and master.grad.abs().sum() > 0


@pytest.mark.parametrize("exporter", ["model_resync", "verl_qat_export"])
def test_mxfp4_qat_expert_weights_pair_with_mxfp4_export(tiny_qwen3_moe, exporter):
    from megatron.lite.model.qwen3_moe.lite.checkpoint import export_hf_weights
    from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4
    from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks

    build, config, ps = tiny_qwen3_moe
    model = build()
    apply_qat_to_chunks([model], QATSpec(enabled=True, format="mxfp4"))
    if exporter == "model_resync":
        exported = dict(export_hf_weights(model, config, ps, target="mxfp4"))
    else:
        from verl_mlite.qat_export import export_qat_weights

        qat_config = {"apply_modelopt_fake_quant": False, "mode": "mxfp4"}
        exported = dict(
            export_qat_weights(export_hf_weights(model, config, ps), qat_config)
        )

    checked = 0
    for layer_idx, layer in enumerate(model.layers):
        experts = layer.moe.experts
        for expert in range(_EXPERTS):
            prefix = f"model.layers.{layer_idx}.mlp.experts.{expert}"
            with torch.no_grad():
                fc1 = getattr(experts.fc1, f"weight{expert}")
                fc2 = getattr(experts.fc2, f"weight{expert}")
            trained = {
                "gate_proj": fc1[:_INTERMEDIATE],
                "up_proj": fc1[_INTERMEDIATE:],
                "down_proj": fc2,
            }
            for proj, w_hat in trained.items():
                packed = exported[f"{prefix}.{proj}.weight"]
                scale = exported[f"{prefix}.{proj}.weight_scale"]
                assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8
                deployed = dequantize_mxfp4(
                    packed.view(torch.int8), scale.view(torch.float8_e8m0fnu)
                )
                deployed = deployed.to(w_hat.dtype)
                assert torch.equal(deployed, w_hat), (prefix, proj)
                checked += 1
    assert checked == _LAYERS * _EXPERTS * 3


def test_qat_rejects_declared_expert_weight_that_is_not_a_parameter():
    from megatron.lite.primitive.quantization.qat import (
        QATSpec,
        apply_qat_to_chunks,
        declare_qat_weights,
    )

    fc = _TEGroupedLinear(2, 32, 32)
    declare_qat_weights(fc, ("weight0", "weight1", "weight2"))
    with pytest.raises(ValueError, match="weight2"):
        apply_qat_to_chunks([fc], QATSpec(enabled=True, format="mxfp4"))
    assert not parametrize.is_parametrized(fc)

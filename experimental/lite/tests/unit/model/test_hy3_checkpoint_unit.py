# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace

from megatron.lite.model.hy3.config import Hy3Config
from megatron.lite.model.hy3.lite.checkpoint import Hy3WeightSpec


def _config() -> Hy3Config:
    return Hy3Config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=2,
        vocab_size=16,
        intermediate_size=12,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        num_nextn_predict_layers=1,
    )


def test_hy3_weight_spec_maps_dense_sparse_shared_bias_and_mtp_names():
    spec = Hy3WeightSpec(_config())
    weight_map = spec.weight_map()

    assert weight_map["layers.0.mlp.gate_up.linear.weight"] == [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    ]
    assert weight_map["layers.1.moe.router.expert_bias"] == [
        "model.layers.1.mlp.expert_bias"
    ]
    assert weight_map["layers.1.moe.shared_mlp.down.linear.weight"] == [
        "model.layers.1.mlp.shared_mlp.down_proj.weight"
    ]
    assert weight_map["mtp.layers.0.enorm.weight"] == ["model.layers.2.enorm.weight"]
    assert weight_map["mtp.layers.0.final_layernorm.weight"] == [
        "model.layers.2.final_layernorm.weight"
    ]


def test_hy3_weight_spec_round_trips_qkv_dense_and_shared_swiglu():
    config = _config()
    spec = Hy3WeightSpec(config)
    hidden = config.hidden_size

    q = torch.arange(8 * hidden).reshape(8, hidden)
    k = torch.arange(4 * hidden).reshape(4, hidden) + 1000
    v = torch.arange(4 * hidden).reshape(4, hidden) + 2000
    packed = spec.hf_to_native("layers.0.attn.qkv.linear.weight", [q, k, v])
    qkv = dict(spec.native_to_hf("layers.0.attn.qkv.linear.weight", packed))
    assert torch.equal(qkv["model.layers.0.self_attn.q_proj.weight"], q)
    assert torch.equal(qkv["model.layers.0.self_attn.k_proj.weight"], k)
    assert torch.equal(qkv["model.layers.0.self_attn.v_proj.weight"], v)

    gate = torch.arange(12 * hidden).reshape(12, hidden)
    up = gate + 1000
    for native_name in (
        "layers.0.mlp.gate_up.linear.weight",
        "layers.1.moe.shared_mlp.gate_up.linear.weight",
    ):
        packed = spec.hf_to_native(native_name, [gate, up])
        exported = dict(spec.native_to_hf(native_name, packed))
        assert any(torch.equal(value, gate) for name, value in exported.items() if "gate_proj" in name)
        assert any(torch.equal(value, up) for name, value in exported.items() if "up_proj" in name)


def test_checkpoint_tensor_iterator_includes_only_mapped_persistent_buffers():
    from megatron.lite.primitive.checkpoint_transforms import iter_checkpoint_tensors

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("expert_bias", torch.zeros(1), persistent=True)
            self.register_buffer("scratch", torch.zeros(1), persistent=False)

    weight_map = {"weight": ["weight"], "expert_bias": ["expert_bias"]}
    tensors = dict(iter_checkpoint_tensors(Module(), weight_map))
    assert set(tensors) == {"weight", "expert_bias"}


def test_hy3_export_translates_grouped_linear_local_expert_names():
    pytest.importorskip("safetensors")
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Module()
            self.fc1.register_parameter("weight0", nn.Parameter(torch.arange(64).reshape(8, 8).float()))

    class Sparse(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = nn.Module()
            self.moe.experts = Experts()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity(), Sparse()])
            self.layer_indices = [0, 1]

    ps = SimpleNamespace(
        pp_size=1,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
    )
    exported = dict(export_hf_weights(Model(), Hy3WeightSpec(_config()), ps))
    assert set(exported) == {
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
    }

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib

import pytest
import torch.nn as nn
from megatron.lite.primitive import recompute as recompute_primitive

pytestmark = pytest.mark.mlite


class _TinyLayer(nn.Module):
    def forward(self, value):
        return value


class _TinyDeepseekV4(nn.Module):
    """Match DeepseekV4Model's direct ``layers`` / ``mtp`` ownership."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _TinyLayer()})
        self.mtp = nn.ModuleList()


class _TinyDirectLayersModel(nn.Module):
    """Match Qwen-family protocols that consume ``chunk.layers`` directly."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer()])


@pytest.mark.parametrize(
    "apply_fn", [recompute_primitive.apply_recompute, recompute_primitive.apply_offload]
)
def test_nonempty_activation_policy_rejects_empty_layers(apply_fn):
    with pytest.raises(
        ValueError, match="non-empty activation policy.*no transformer layers"
    ):
        apply_fn(nn.ModuleList(), ["full"], {})


def test_full_recompute_config_wraps_ds4_and_direct_layer_models(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    ds4_protocol = importlib.import_module(
        "megatron.lite.model.deepseek_v4.lite.protocol"
    )
    qwen_protocol = importlib.import_module(
        "megatron.lite.model.qwen3_moe.lite.protocol"
    )

    ds4 = _TinyDeepseekV4()
    direct = _TinyDirectLayersModel()
    ds4_forward = ds4.layers["0"].forward
    direct_forward = direct.layers[0].forward

    ds4_spec = recompute_primitive.parse_recompute_spec(
        ds4_protocol.ImplConfig(recompute=["full"]).recompute
    )
    qwen_spec = recompute_primitive.parse_recompute_spec(
        qwen_protocol.ImplConfig(recompute=["full"]).recompute
    )
    recompute_primitive.apply_recompute(
        ds4_protocol._iter_transformer_units(ds4), ds4_spec, ds4_protocol.MODULE_MAP
    )
    recompute_primitive.apply_recompute(
        direct.layers, qwen_spec, qwen_protocol.MODULE_MAP
    )

    assert ds4.layers["0"].forward != ds4_forward
    assert direct.layers[0].forward != direct_forward


@pytest.mark.parametrize(
    "protocol_name", ["deepseek_v4", "glm5", "kimi_k2", "qwen3_5", "qwen3_moe"]
)
def test_unknown_model_config_override_fails_loud(
    protocol_name, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    protocol = importlib.import_module(
        f"megatron.lite.model.{protocol_name}.lite.protocol"
    )

    with pytest.raises(
        ValueError, match="Unknown .*Config override: misspelled_option"
    ):
        protocol.build_model_config({}, misspelled_option=True)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contracts for the MLite-owned QAT rollout exporter."""

from __future__ import annotations

import sys

import pytest
import torch
from megatron.lite.primitive.quantization.mxfp4 import MXFP4_BLOCK_SIZE, quantize_mxfp4


def _qat_config(**overrides):
    config = {
        "enable": True,
        "apply_modelopt_fake_quant": False,
        "mode": "mxfp4",
        "group_size": MXFP4_BLOCK_SIZE,
        "ignore_patterns": ["lm_head", "embed_tokens", "re:.*mlp.gate$"],
    }
    config.update(overrides)
    return config


def test_mxfp4_export_is_independent_of_verl_modelopt(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "verl.utils.modelopt", None)
    from verl_mlite.qat_export import export_qat_weights

    weight = torch.linspace(-4, 4, MXFP4_BLOCK_SIZE * 2).reshape(2, -1)
    source = iter(
        [
            ("model.layers.0.mlp.experts.0.up_proj.weight", weight),
            ("model.embed_tokens.weight", weight.clone()),
            ("model.layers.0.mlp.gate.weight", weight.clone()),
        ]
    )

    exported = dict(export_qat_weights(source, _qat_config()))

    packed, scale = quantize_mxfp4(weight)
    prefix = "model.layers.0.mlp.experts.0.up_proj"
    assert exported[f"{prefix}.weight"].dtype == torch.uint8
    assert exported[f"{prefix}.weight_scale"].dtype == torch.uint8
    assert torch.equal(exported[f"{prefix}.weight"], packed.view(torch.uint8))
    assert torch.equal(exported[f"{prefix}.weight_scale"], scale.view(torch.uint8))
    assert torch.equal(exported["model.embed_tokens.weight"], weight)
    assert torch.equal(exported["model.layers.0.mlp.gate.weight"], weight)


def test_qat_export_stays_lazy() -> None:
    from verl_mlite.qat_export import export_qat_weights

    consumed = []

    def source():
        consumed.append(True)
        yield "model.layers.0.mlp.down_proj.weight", torch.ones(2, MXFP4_BLOCK_SIZE)

    exported = export_qat_weights(source(), _qat_config())

    assert consumed == []
    next(exported)
    assert consumed == [True]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"apply_modelopt_fake_quant": True}, "apply_modelopt_fake_quant=False"),
        ({"mode": "w4a16", "group_size": 16}, "only supports mode='mxfp4'"),
        ({"group_size": 16}, "group_size=32"),
    ],
)
def test_qat_export_rejects_unsupported_contract(overrides, message) -> None:
    from verl_mlite.qat_export import export_qat_weights

    with pytest.raises(ValueError, match=message):
        export_qat_weights(iter(()), _qat_config(**overrides))

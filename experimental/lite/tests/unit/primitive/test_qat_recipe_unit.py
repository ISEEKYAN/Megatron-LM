# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for the ModelOpt/CT QAT recipe boundary."""

import sys
from types import ModuleType

import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.quantization.recipe import (
    QATRecipe,
    QuantizerB,
    recipe_contract,
)

pytestmark = pytest.mark.mlite


def test_recipe_contracts_reject_int4_g16_marlin_alias():
    int4 = recipe_contract(QATRecipe.INT4_W4A16_G128)
    assert (int4.weight_format, int4.group_size, int4.compressed_tensors_scheme) == (
        "int4",
        128,
        "W4A16",
    )
    nvfp4 = recipe_contract("nvfp4_w4a16")
    assert (nvfp4.group_size, nvfp4.scale_format, nvfp4.activation_quantized) == (
        16,
        "float8_e4m3fn",
        False,
    )
    with pytest.raises(ValueError, match="Unsupported QAT recipe"):
        recipe_contract("int4_w4a16_g16")


def test_quantizer_b_requires_authoritative_optional_backends():
    quantizer = QuantizerB(QATRecipe.NVFP4_W4A16)
    with pytest.raises(RuntimeError, match="ModelOpt"):
        quantizer.prepare(nn.Linear(16, 16, bias=False))
    with pytest.raises(RuntimeError, match="export_hf_checkpoint"):
        list(quantizer.export([("weight", torch.ones(1))]))


def test_nvfp4_hf_export_delegates_to_modelopt_under_inference_mode(monkeypatch, tmp_path):
    """NVFP4 deployment must use ModelOpt's artifact-aware HF exporter."""
    calls = []

    def export_hf_checkpoint(model, *, export_dir):
        calls.append((model, export_dir, torch.is_inference_mode_enabled()))

    modelopt = ModuleType("modelopt")
    modelopt.__path__ = []
    modelopt_torch = ModuleType("modelopt.torch")
    modelopt_torch.__path__ = []
    modelopt_export = ModuleType("modelopt.torch.export")
    modelopt_export.export_hf_checkpoint = export_hf_checkpoint
    monkeypatch.setitem(sys.modules, "modelopt", modelopt)
    monkeypatch.setitem(sys.modules, "modelopt.torch", modelopt_torch)
    monkeypatch.setitem(sys.modules, "modelopt.torch.export", modelopt_export)

    model = nn.Linear(16, 16, bias=False)
    QuantizerB("nvfp4_w4a16").export_hf_checkpoint(model, tmp_path)

    assert calls == [(model, str(tmp_path), True)]

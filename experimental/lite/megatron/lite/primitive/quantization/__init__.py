# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization primitives for Megatron Lite (weight-only QAT: fp8/mxfp4 + int8/int4)."""

from __future__ import annotations

from megatron.lite.primitive.quantization.qat import (
    QATSpec,
    WeightFakeQuant,
    apply_qat_to_chunks,
    apply_qat_to_module,
    compute_amax,
    dequantize_weight,
    fake_quantize_weight,
    normalize_qat_spec,
    pack_int4,
    qat_state_dict,
    quantize_weight,
    unpack_int4,
)
from megatron.lite.primitive.quantization.recipe import (
    QATRecipe,
    QuantizerB,
    RecipeContract,
    recipe_contract,
)

__all__ = [
    "QATSpec",
    "WeightFakeQuant",
    "apply_qat_to_chunks",
    "apply_qat_to_module",
    "compute_amax",
    "dequantize_weight",
    "fake_quantize_weight",
    "normalize_qat_spec",
    "pack_int4",
    "qat_state_dict",
    "quantize_weight",
    "unpack_int4",
    "QATRecipe",
    "QuantizerB",
    "RecipeContract",
    "recipe_contract",
]

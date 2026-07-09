# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import torch


def test_select_proxy_weights_requires_dense_and_routed_expert_scale_pairs() -> None:
    from examples.verl.ds4_hopper_resync_proxy import select_proxy_weights

    weight_map = {
        "embed.weight": "global.safetensors",
        "layers.0.attn.wq_b.weight": "dense.safetensors",
        "layers.0.attn.wq_b.scale": "dense.safetensors",
        "layers.1.ffn.experts.7.w1.weight": "expert.safetensors",
        "layers.1.ffn.experts.7.w1.scale": "expert.safetensors",
    }

    assert select_proxy_weights(weight_map) == {
        "dense": "layers.0.attn.wq_b.weight",
        "routed_expert": "layers.1.ffn.experts.7.w1.weight",
    }


def test_crop_source_pair_preserves_block_aligned_dense_and_mxfp4_values() -> None:
    from examples.verl.ds4_hopper_resync_proxy import crop_source_pair
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    values = torch.linspace(-3.0, 3.0, 256 * 256).reshape(256, 256)
    dense_weight, dense_scale = quantize_block_fp8(values, scale_format="float32")
    expert_weight, expert_scale = quantize_mxfp4(values)

    dense = crop_source_pair(
        dense_weight,
        dense_scale,
        source_kind="block_fp8",
        block_shape=(128, 128),
        proxy_shape=(128, 128),
    )
    expert = crop_source_pair(
        expert_weight,
        expert_scale,
        source_kind="mxfp4",
        block_shape=(128, 128),
        proxy_shape=(128, 128),
    )

    assert dense.shape == expert.shape == (128, 128)
    assert dense.dtype == expert.dtype == torch.bfloat16
    assert torch.isfinite(dense).all()
    assert torch.isfinite(expert).all()


def test_export_trained_proxy_forces_dense_and_expert_to_pure_block_fp8() -> None:
    from examples.verl.ds4_hopper_resync_proxy import export_trained_proxy

    weights = {
        "layers.0.attn.wq_b.weight": torch.randn(128, 128, dtype=torch.bfloat16),
        "layers.1.ffn.experts.7.w1.weight": torch.randn(128, 128, dtype=torch.bfloat16),
    }
    config = SimpleNamespace(
        expert_dtype="fp4",
        quantization_config={"weight_block_size": [128, 128]},
    )

    exported, restored = export_trained_proxy(weights, config)

    for name in weights:
        assert exported[name].dtype == torch.float8_e4m3fn
        assert exported[name.replace(".weight", ".scale")].dtype == torch.float32
        assert restored[name].shape == weights[name].shape
        assert restored[name].dtype == torch.bfloat16
    assert not any(tensor.dtype == torch.int8 for tensor in exported.values())


def test_proxy_gate_uses_the_frozen_dapo_thresholds() -> None:
    from examples.verl.ds4_hopper_resync_proxy import evaluate_proxy_gate

    passing = {
        "fp32": {
            "p99_ratio_deviation": 0.009,
            "max_ratio_deviation": 0.049,
            "p99_kl": 9e-5,
            "clipping_boundary_crossings": 0,
        }
    }
    failing = {
        "fp32": {
            **passing["fp32"],
            "clipping_boundary_crossings": 1,
        }
    }

    assert evaluate_proxy_gate(passing)["acceptable"] is True
    rejected = evaluate_proxy_gate(failing)
    assert rejected["acceptable"] is False
    assert rejected["failures"] == ["clipping boundary crossings"]


def test_proxy_token_sequences_are_fixed_and_token_aligned() -> None:
    from examples.verl.ds4_hopper_resync_proxy import fixed_token_sequences

    sequences = fixed_token_sequences(vocab_size=128)

    assert len(sequences) == 8
    assert {len(sequence) for sequence in sequences} == {12}
    assert all(0 <= token < 128 for sequence in sequences for token in sequence)

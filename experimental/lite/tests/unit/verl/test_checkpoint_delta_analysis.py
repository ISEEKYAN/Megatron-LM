# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib.util
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file


SCRIPT = (
    Path(__file__).parents[3]
    / "examples"
    / "verl"
    / "scripts"
    / "analyze_checkpoint_delta.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_checkpoint_delta", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
delta_analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delta_analysis)


def test_classify_parameter_family_uses_exported_names() -> None:
    classify = delta_analysis.classify_parameter_family

    assert classify("model.embed_tokens.weight") == "embedding"
    assert classify("model.layers.0.self_attn.q_proj.weight") == "attention"
    assert classify("model.layers.0.linear_attn.in_proj_qkv.weight") == "attention"
    assert classify("model.layers.0.mlp.experts.3.down_proj.weight") == "expert"
    assert classify("model.layers.0.mlp.gate.weight") == "router"
    assert classify("model.layers.0.mlp.shared_expert_gate.weight") == "router"
    assert classify("model.layers.0.mlp.shared_expert.gate_proj.weight") == "expert"
    assert classify("model.layers.0.input_layernorm.weight") == "norm"
    assert classify("model.layers.0.self_attn.q_norm.weight") == "norm"
    assert classify("model.vision_model.blocks.0.mlp.linear_fc1.weight") == "vision"
    assert classify("lm_head.weight") == "head"
    assert classify("model.module.output_layer.weight") == "head"


def test_tensor_delta_statistics_counts_exact_and_thresholded_changes() -> None:
    before = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    after = torch.tensor([0.0, 1.005, 2.02, 5.0], dtype=torch.float32)

    stats = delta_analysis.tensor_delta_statistics(
        "model.layers.0.self_attn.q_proj.weight",
        before,
        after,
        thresholds=(0.0, 0.01, 1.0),
        magnitude_edges=(0.0, 0.01, 0.1, 1.0, math.inf),
        chunk_elements=2,
    )

    assert stats["family"] == "attention"
    assert stats["numel"] == 4
    assert stats["changed_counts"] == {"0": 3, "0.01": 2, "1": 1}
    assert stats["changed_fractions"] == {"0": 0.75, "0.01": 0.5, "1": 0.25}
    assert stats["l_inf"] == 2.0
    assert stats["l2"] == pytest_approx(math.sqrt(2.0**2 + 0.02**2 + 0.005**2))
    assert stats["magnitude_histogram"] == [1, 1, 0, 1]


def test_tensor_delta_statistics_measures_lossless_xor_compression() -> None:
    before = torch.zeros(4096, dtype=torch.bfloat16)
    after = before.clone()
    after[17] = 1.0

    stats = delta_analysis.tensor_delta_statistics(
        "model.layers.0.self_attn.q_proj.weight", before, after, chunk_elements=257
    )

    assert stats["xor_nonzero_bytes"] > 0
    assert stats["xor_nonzero_bytes"] < stats["dense_bytes"]
    if delta_analysis.zstandard is None:
        assert stats["xor_zstd_bytes"] is None
    else:
        assert stats["xor_zstd_bytes"] < stats["dense_bytes"]


def test_tensor_delta_statistics_measures_real_block_fp8_target_format() -> None:
    before = torch.tensor([[0.5, 1.0], [0.25, -1.0]], dtype=torch.bfloat16)
    after = before.clone()
    after[0, 0] = torch.nextafter(
        before[0, 0], torch.tensor(float("inf"), dtype=torch.bfloat16)
    )

    stats = delta_analysis.tensor_delta_statistics(
        "model.layers.0.self_attn.q_proj.weight",
        before,
        after,
        fp8_block_shape=(2, 2),
    )

    assert stats["changed_fractions"]["0"] == 0.25
    fp8 = stats["target_formats"]["block_fp8"]
    assert fp8["kind"] == "quantized"
    assert fp8["weight_dtype"] == "float8_e4m3fn"
    assert fp8["scale_dtype"] == "float32"
    assert fp8["weight_changed_count"] == 0
    assert fp8["scale_changed_count"] == 0
    assert fp8["changed_value_fraction"] == 0.0


def test_tensor_delta_statistics_keeps_unquantized_head_in_target_format() -> None:
    before = torch.zeros((2, 2), dtype=torch.bfloat16)
    after = before.clone()
    after[0, 0] = 1.0

    stats = delta_analysis.tensor_delta_statistics(
        "lm_head.weight", before, after, fp8_block_shape=(2, 2)
    )

    fp8 = stats["target_formats"]["block_fp8"]
    assert fp8["kind"] == "passthrough_bf16"
    assert fp8["changed_value_fraction"] == 0.25
    assert fp8["serialized_bytes"] == stats["dense_bytes"]


def test_tensor_delta_statistics_quantizes_dcp_local_expert_weight_name() -> None:
    before = torch.zeros((2, 2), dtype=torch.bfloat16)
    after = before.clone()
    after[0, 0] = 1.0

    stats = delta_analysis.tensor_delta_statistics(
        "model.module.layers.0.moe.experts.fc1.weight4",
        before,
        after,
        fp8_block_shape=(2, 2),
    )

    assert stats["family"] == "expert"
    assert stats["target_formats"]["block_fp8"]["kind"] == "quantized"


def test_analyze_safetensor_checkpoints_aggregates_families_and_bytes(tmp_path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    save_file(
        {
            "model.embed_tokens.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                [1.0, 2.0], dtype=torch.bfloat16
            ),
        },
        before / "model-00001-of-00001.safetensors",
    )
    save_file(
        {
            "model.embed_tokens.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                [1.0, 3.0], dtype=torch.bfloat16
            ),
        },
        after / "model-00001-of-00001.safetensors",
    )

    report = delta_analysis.analyze_checkpoints(
        before,
        after,
        thresholds=(0.0, 0.5),
        magnitude_edges=(0.0, 0.5, math.inf),
        chunk_elements=1,
    )

    assert report["summary"]["tensor_count"] == 2
    assert report["summary"]["numel"] == 4
    assert report["summary"]["changed_counts"] == {"0": 1, "0.5": 1}
    assert report["summary"]["dense_bytes"] == 8
    assert report["summary"]["bitmap_value_bytes"]["0"] == 3
    assert report["summary"]["xor_nonzero_bytes"] == 1
    assert report["summary"]["xor_nonzero_byte_fraction"] == 0.125
    assert report["families"]["embedding"]["changed_counts"]["0"] == 0
    assert report["families"]["attention"]["changed_counts"]["0"] == 1


def test_analyze_dcp_checkpoints_streams_tensor_state(tmp_path) -> None:
    before = tmp_path / "before-dcp"
    after = tmp_path / "after-dcp"
    name = "model.module.layers.0.attn.qkv.linear.weight"
    dcp.save({name: torch.ones((2, 2), dtype=torch.bfloat16)}, checkpoint_id=before)
    changed = torch.ones((2, 2), dtype=torch.bfloat16)
    changed[0, 0] = 2.0
    dcp.save({name: changed, "step": 2}, checkpoint_id=after)

    report = delta_analysis.analyze_checkpoints(
        before,
        after,
        thresholds=(0.0,),
        magnitude_edges=(0.0, math.inf),
        fp8_block_shape=(2, 2),
    )

    assert report["checkpoint_format"] == "torch_dcp"
    assert report["summary"]["tensor_count"] == 1
    assert report["summary"]["changed_counts"]["0"] == 1
    assert report["families"]["attention"]["tensor_count"] == 1


def test_analyze_aggregates_depth_layers_and_change_concentration(tmp_path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    zeros = torch.zeros((2, 2), dtype=torch.bfloat16)
    shallow = torch.ones((2, 2), dtype=torch.bfloat16)
    middle = zeros.clone()
    middle[0, 0] = 1.0
    names = {
        "model.layers.0.self_attn.q_proj.weight": shallow,
        "model.layers.1.mlp.experts.0.down_proj.weight": middle,
        "model.layers.2.input_layernorm.weight": zeros,
        "lm_head.weight": zeros,
    }
    save_file({name: zeros.clone() for name in names}, before / "model.safetensors")
    save_file(
        {name: value.clone() for name, value in names.items()},
        after / "model.safetensors",
    )

    report = delta_analysis.analyze_checkpoints(
        before,
        after,
        thresholds=(0.0,),
        magnitude_edges=(0.0, math.inf),
        fp8_block_shape=(2, 2),
    )

    assert report["layers"]["layer.0"]["changed_counts"]["0"] == 4
    assert report["layers"]["layer.1"]["changed_counts"]["0"] == 1
    assert report["depths"]["shallow"]["changed_counts"]["0"] == 4
    assert report["depths"]["middle"]["changed_counts"]["0"] == 1
    assert report["depths"]["deep"]["changed_counts"]["0"] == 0
    assert report["family_depths"]["attention"]["shallow"]["tensor_count"] == 1
    concentration = report["layer_concentration"]
    assert concentration["layer_count"] == 3
    assert concentration["layers_for_80pct_changes"] == 1
    assert concentration["top_10pct_change_share"] == pytest_approx(0.8)


def test_analyze_rejects_the_same_checkpoint_directory(tmp_path) -> None:
    import pytest

    save_file(
        {"model.embed_tokens.weight": torch.ones(1, dtype=torch.bfloat16)},
        tmp_path / "model.safetensors",
    )

    with pytest.raises(ValueError, match="must be different"):
        delta_analysis.analyze_checkpoints(tmp_path, tmp_path)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, rel=1e-6, abs=1e-8)

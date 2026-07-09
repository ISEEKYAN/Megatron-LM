# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib.util
import math
from pathlib import Path

import torch
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

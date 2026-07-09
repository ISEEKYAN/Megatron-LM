import json

import pytest
import torch
from safetensors.torch import save_file


def _write_checkpoint(path) -> None:
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
    from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

    dense = torch.linspace(-3.0, 3.0, 128 * 128).reshape(128, 128)
    expert = torch.linspace(-6.0, 6.0, 2 * 64).reshape(2, 64)
    dense_weight, dense_scale = quantize_block_fp8(dense, scale_format="e8m0")
    expert_weight, expert_scale = quantize_mxfp4(expert)
    tensors = {
        "layers.2.attn.wo.weight": dense_weight,
        "layers.2.attn.wo.scale": dense_scale,
        "layers.2.ffn.experts.0.w1.weight": expert_weight,
        "layers.2.ffn.experts.0.w1.scale": expert_scale,
        "layers.2.attn.indexer.weights_proj.weight": torch.ones(
            128, 128, dtype=torch.bfloat16
        ),
        "layers.2.ffn.gate.weight": torch.ones(128, 128, dtype=torch.bfloat16),
        "layers.2.hc_attn_fn": torch.ones(4),
    }
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(
        json.dumps(
            {
                "expert_dtype": "fp4",
                "quantization_config": {
                    "quant_method": "fp8",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )


def test_roundtrip_reports_fp8_mxfp4_and_effective_ignored_layers(tmp_path) -> None:
    from examples.verl.ds4_checkpoint_roundtrip import run_roundtrip

    _write_checkpoint(tmp_path)
    report = run_roundtrip(tmp_path, device="cpu")

    assert report["summary"]["block_fp8"]["tensor_count"] == 1
    assert report["summary"]["mxfp4"]["tensor_count"] == 1
    assert report["audit"]["source"] == "inferred_from_scale_pairs"
    assert report["audit"]["violations"] == []
    assert report["audit"]["special_layers"]["indexer"]["unscaled_weights"] == [
        "layers.2.attn.indexer.weights_proj.weight"
    ]
    assert report["audit"]["special_layers"]["router"]["unscaled_weights"] == [
        "layers.2.ffn.gate.weight"
    ]
    assert report["audit"]["special_layers"]["mhc"]["direct_tensors"] == [
        "layers.2.hc_attn_fn"
    ]
    assert report["audit"]["special_layers"]["o_lora"] == {
        "direct_tensors": [],
        "scaled_weights": [],
        "unscaled_weights": [],
    }
    assert report["layers"]["layers.2"]["block_fp8"]["tensor_count"] == 1
    assert report["layers"]["layers.2"]["mxfp4"]["tensor_count"] == 1


def test_roundtrip_separates_scale_bytes_from_numeric_error(tmp_path) -> None:
    from examples.verl.ds4_checkpoint_roundtrip import run_roundtrip

    _write_checkpoint(tmp_path)
    report = run_roundtrip(tmp_path, device="cpu")

    for kind in ("block_fp8", "mxfp4"):
        metrics = report["summary"][kind]
        assert metrics["scale_byte_mismatch"]["total"] > 0
        assert metrics["scale_byte_mismatch"]["mismatched"] == 0
        assert metrics["relative_l2"]["max"] >= 0.0
        assert metrics["max_abs"]["max"] >= 0.0


def test_roundtrip_fails_closed_on_unknown_unscaled_matrix(tmp_path) -> None:
    from examples.verl.ds4_checkpoint_roundtrip import run_roundtrip

    save_file(
        {"layers.2.attn.unknown.weight": torch.ones(128, 128)},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "expert_dtype": "fp4",
                "quantization_config": {"weight_block_size": [128, 128]},
            }
        )
    )

    with pytest.raises(ValueError, match="unrecognized unscaled checkpoint weights"):
        run_roundtrip(tmp_path, device="cpu")


def test_roundtrip_preserves_pure_block_fp8_float32_scale_contract() -> None:
    from examples.verl.ds4_checkpoint_roundtrip import _measure_pair
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

    source = torch.linspace(-3.0, 3.0, 128 * 128).reshape(128, 128)
    weight, scale = quantize_block_fp8(source, scale_format="float32")

    record = _measure_pair(
        "layers.2.ffn.experts.0.w1.weight",
        weight,
        scale,
        expert_dtype="fp8",
        block_shape=(128, 128),
        device=torch.device("cpu"),
    )

    assert record["kind"] == "block_fp8"
    assert record["scale_byte_mismatch"]["mismatched"] == 0
    assert (
        record["scale_byte_mismatch"]["total"] == scale.numel() * scale.element_size()
    )

from types import SimpleNamespace

import pytest
import torch


def _config(*, expert_dtype: str = "fp4") -> SimpleNamespace:
    return SimpleNamespace(
        expert_dtype=expert_dtype,
        quantization_config={
            "quant_method": "deepseek_v4_fp8",
            "weight_block_size": [128, 128],
            "ignored_layers": ["layers.0.attn.indexer.wq_a", "layers.0.ffn.gate"],
        },
    )


def test_flash_resync_stream_separates_mxfp4_experts_and_block_fp8_dense() -> None:
    from megatron.lite.model.deepseek_v4.lite.resync import export_resync_weights

    dense = torch.linspace(-1.0, 1.0, 128 * 128).reshape(128, 128)
    expert = torch.tensor([0.5, -1.0, 1.5, -6.0] * 8).reshape(1, 32)
    ignored = torch.ones(128, 128, dtype=torch.bfloat16)
    stream = [
        ("layers.0.attn.wq.weight", dense),
        ("layers.0.ffn.experts.0.w1.weight", expert),
        ("layers.0.attn.indexer.wq_a.weight", ignored),
    ]

    exported = dict(export_resync_weights(stream, _config()))

    assert exported["layers.0.attn.wq.weight"].dtype == torch.float8_e4m3fn
    assert exported["layers.0.attn.wq.scale"].dtype == torch.float8_e8m0fnu
    assert exported["layers.0.attn.wq.scale"].shape == (1, 1)
    assert exported["layers.0.ffn.experts.0.w1.weight"].dtype == torch.int8
    assert exported["layers.0.ffn.experts.0.w1.weight"].shape == (1, 16)
    assert exported["layers.0.ffn.experts.0.w1.scale"].dtype == torch.float8_e8m0fnu
    assert exported["layers.0.attn.indexer.wq_a.weight"] is ignored
    assert "layers.0.attn.indexer.wq_a.scale" not in exported


def test_flash_base_resync_keeps_w1_w3_separate_with_128x128_scales() -> None:
    from megatron.lite.model.deepseek_v4.lite.resync import export_resync_weights

    stream = [
        ("layers.0.ffn.experts.0.w1.weight", torch.ones(256, 128)),
        ("layers.0.ffn.experts.0.w3.weight", torch.full((256, 128), 2.0)),
        ("layers.0.ffn.experts.0.w2.weight", torch.ones(128, 256)),
    ]

    exported = dict(export_resync_weights(stream, _config(expert_dtype="fp8")))

    assert exported["layers.0.ffn.experts.0.w1.scale"].shape == (2, 1)
    assert exported["layers.0.ffn.experts.0.w3.scale"].shape == (2, 1)
    assert exported["layers.0.ffn.experts.0.w2.scale"].shape == (1, 2)
    assert exported["layers.0.ffn.experts.0.w1.scale"].dtype == torch.float32
    assert not any("w13" in name for name in exported)


def test_production_expert_shapes_match_flash_and_base_checkpoint_layouts() -> None:
    from megatron.lite.model.deepseek_v4.lite.resync import export_resync_weights

    source = torch.zeros(2048, 4096, dtype=torch.bfloat16)
    name = "layers.0.ffn.experts.0.w1.weight"

    flash = dict(export_resync_weights([(name, source)], _config(expert_dtype="fp4")))
    assert flash[name].shape == (2048, 2048)
    assert flash[name].dtype == torch.int8
    assert flash["layers.0.ffn.experts.0.w1.scale"].shape == (2048, 128)
    assert flash["layers.0.ffn.experts.0.w1.scale"].dtype == torch.float8_e8m0fnu

    base = dict(export_resync_weights([(name, source)], _config(expert_dtype="fp8")))
    assert base[name].shape == (2048, 4096)
    assert base[name].dtype == torch.float8_e4m3fn
    assert base["layers.0.ffn.experts.0.w1.scale"].shape == (16, 32)
    assert base["layers.0.ffn.experts.0.w1.scale"].dtype == torch.float32


def test_resync_stream_fails_closed_without_ds4_quantization_contract() -> None:
    from megatron.lite.model.deepseek_v4.lite.resync import export_resync_weights

    with pytest.raises(ValueError, match="quantization_config"):
        dict(
            export_resync_weights(
                [("layers.0.attn.wq.weight", torch.ones(128, 128))], SimpleNamespace()
            )
        )


def test_resync_stream_rejects_unknown_expert_dtype() -> None:
    from megatron.lite.model.deepseek_v4.lite.resync import export_resync_weights

    with pytest.raises(ValueError, match="expert_dtype"):
        dict(
            export_resync_weights(
                [("layers.0.ffn.experts.0.w1.weight", torch.ones(128, 128))],
                _config(expert_dtype="int4"),
            )
        )


def test_deepseek_config_preserves_checkpoint_quantization_contract() -> None:
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    quantization_config = {
        "quant_method": "deepseek_v4_fp8",
        "weight_block_size": [128, 128],
        "ignored_layers": ["head"],
    }
    config = DeepseekV4Config._from_hf_dict(
        {"expert_dtype": "fp4", "quantization_config": quantization_config}
    )

    assert config.expert_dtype == "fp4"
    assert config.quantization_config == quantization_config


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("layers.0.mlp.gate.tid2eid", True),
        ("layers.0.mlp.gate.expert_bias", False),
        ("layers.1.mlp.gate.tid2eid", False),
        ("layers.1.mlp.gate.expert_bias", True),
        ("mtp.0.mlp.gate.tid2eid", False),
        ("mtp.0.mlp.gate.expert_bias", True),
    ],
)
def test_router_export_keeps_only_the_buffer_for_each_layer_kind(
    name: str, expected: bool
) -> None:
    from megatron.lite.model.deepseek_v4.lite.checkpoint import (
        _router_buffer_matches_layer_kind,
    )

    config = SimpleNamespace(num_hash_layers=1)
    assert _router_buffer_matches_layer_kind(name, config) is expected


@pytest.mark.parametrize(
    "name",
    [
        "embed.weight",
        "head.weight",
        "norm.weight",
        "layers.0.attn_norm.weight",
        "layers.0.attn.q_norm.weight",
        "layers.0.attn.kv_norm.weight",
        "layers.0.attn.compressor.wgate.weight",
        "layers.0.attn.compressor.wkv.weight",
        "layers.0.attn.indexer.compressor.wgate.weight",
        "layers.0.attn.indexer.weights_proj.weight",
        "layers.0.ffn.gate.weight",
        "mtp.0.enorm.weight",
    ],
)
def test_official_flash_unscaled_weight_families_are_not_quantized(name: str) -> None:
    from megatron.lite.model.deepseek_v4.lite.resync import export_resync_weights

    config = SimpleNamespace(
        expert_dtype="fp4",
        quantization_config={
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    )
    source = torch.ones(128, 128, dtype=torch.bfloat16)

    exported = dict(export_resync_weights([(name, source)], config))

    assert exported == {name: source}


def test_official_flash_indexer_wq_b_remains_block_fp8() -> None:
    from megatron.lite.model.deepseek_v4.lite.resync import export_resync_weights

    config = SimpleNamespace(
        expert_dtype="fp4",
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    exported = dict(
        export_resync_weights(
            [("layers.2.attn.indexer.wq_b.weight", torch.ones(128, 128))], config
        )
    )

    assert exported["layers.2.attn.indexer.wq_b.weight"].dtype == torch.float8_e4m3fn
    assert exported["layers.2.attn.indexer.wq_b.scale"].dtype == torch.float8_e8m0fnu

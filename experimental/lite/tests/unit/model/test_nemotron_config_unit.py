"""Guard independent attention/SSM dimensions and HF layer order, without CUDA."""

import pytest
from megatron.lite.model.nemotron_h.config import NemotronHConfig


@pytest.fixture
def lightning_config():
    return dict(
        model_type="nemotron_h",
        hidden_size=2688,
        vocab_size=131072,
        layers_block_type=["linear_attention", "moe", "full_attention"],
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        mamba_num_heads=64,
        mamba_head_dim=64,
        n_groups=8,
        ssm_state_size=128,
        conv_kernel=4,
        chunk_size=128,
        n_routed_experts=128,
        num_experts_per_tok=6,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
        n_shared_experts=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        layer_norm_epsilon=1e-5,
        residual_in_fp32=False,
        mlp_hidden_act="relu2",
        mamba_hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        mamba_proj_bias=False,
        use_bias=False,
        use_conv_bias=True,
        tie_word_embeddings=False,
        n_group=1,
        topk_group=1,
        expand=2,
    )


def test_independent_projection_dimensions_and_layer_order(lightning_config):
    cfg = NemotronHConfig._from_hf_dict(lightning_config)
    assert cfg.mamba_inner_size == 4096  # Not hidden_size * expand.
    assert cfg.mamba_conv_dim == 6144
    assert cfg.mamba_in_proj_size == 10304
    assert cfg.attention_qkv_size == 4608
    assert cfg.layers_block_type == ["linear_attention", "moe", "full_attention"]
    assert cfg.num_hidden_layers == 3
    assert cfg.mlp_hidden_act == "relu2"
    assert cfg.routed_scaling_factor == 2.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_hidden_layers", 52),
        ("n_groups", 3),
        ("num_key_value_heads", 3),
        ("num_experts_per_tok", 129),
        ("layers_block_type", ["gdn"]),
    ],
)
def test_reject_inconsistent_architecture(lightning_config, field, value):
    lightning_config[field] = value
    with pytest.raises(ValueError):
        NemotronHConfig._from_hf_dict(lightning_config)


def test_original_hf_layer_names_preserve_architecture(lightning_config):
    lightning_config["layers_block_type"] = ["mamba", "moe", "attention"]
    cfg = NemotronHConfig._from_hf_dict(lightning_config)
    assert cfg.layers_block_type == ["linear_attention", "moe", "full_attention"]
    assert lightning_config["layers_block_type"] == ["mamba", "moe", "attention"]


def test_missing_mamba_heads_cannot_fall_back_to_expand(lightning_config):
    del lightning_config["mamba_num_heads"]
    with pytest.raises(ValueError, match="mamba_num_heads"):
        NemotronHConfig._from_hf_dict(lightning_config)

from __future__ import annotations

import torch

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm import checkpoint
from megatron.lite.model.deepseek_v4.vllm.checkpoint import DeepseekV4WeightSpec
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization import deployment_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import (
    dequantize_mxfp4,
    quantize_mxfp4,
)


def test_fused_attention_loads_weight_scale_pairs_with_unequal_rows() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    spec = DeepseekV4WeightSpec(config)
    native = "layers.0.self_attn.fused_wqa_wkv"
    assert spec._load_names(native) == [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wkv.scale",
    ]
    target = torch.Size((384, 128))
    assert spec.hf_target_shape(native, 0, target) == torch.Size((256, 128))
    assert spec.hf_target_shape(native, 1, target) == torch.Size((2, 1))
    assert spec.hf_target_shape(native, 2, target) == torch.Size((128, 128))
    assert spec.hf_target_shape(native, 3, target) == torch.Size((1, 1))
    assert spec.read_hf_source_raw(native, 0, spec._load_names(native)[0])
    assert not spec.read_hf_source_raw(
        "layers.0.self_attn.q_norm", 0, "layers.0.attn.q_norm.weight"
    )

    q = torch.ones(256, 128, dtype=torch.float8_e4m3fn)
    kv = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
    master = spec.hf_to_native(
        native,
        [
            q,
            torch.full((2, 1), 2.0),
            kv,
            torch.full((1, 1), 4.0),
        ],
    )
    assert master.dtype == torch.bfloat16
    assert master.shape == target
    assert torch.all(master[:256] == 2)
    assert torch.all(master[256:] == 4)


def test_mixed_release_mxfp4_expert_loads_to_bf16_master() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        moe_intermediate_size=64,
        n_routed_experts=2,
    )
    spec = DeepseekV4WeightSpec(config)
    native = "layers.0.mlp.experts.w2.0"
    source = torch.linspace(-3, 3, 64 * 128, dtype=torch.bfloat16).reshape(64, 128)
    packed, scale = quantize_mxfp4(source)

    master = spec.hf_to_native(native, [packed, scale])

    assert master.dtype == torch.bfloat16
    torch.testing.assert_close(
        master,
        dequantize_mxfp4(packed, scale).to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert native not in spec.source_block_scales


def test_mhc_checkpoint_parameters_preserve_fp32_release_values() -> None:
    spec = DeepseekV4WeightSpec(DeepseekV4Config(hidden_size=128))
    source = torch.tensor([0.6337993144989014], dtype=torch.float32)

    master = spec.hf_to_native("layers.0.attn_hc.hc_fn", [source])
    sink = spec.hf_to_native("layers.0.self_attn.attn_sink", [source])

    assert master.dtype == torch.float32
    torch.testing.assert_close(master, source, rtol=0, atol=0)
    assert sink.dtype == torch.float32
    torch.testing.assert_close(sink, source, rtol=0, atol=0)


def test_mhc_export_preserves_fp32_values() -> None:
    model = torch.nn.Module()
    model.hc_head = torch.nn.Module()
    source = torch.tensor([0.6337993144989014], dtype=torch.float32)
    model.hc_head.hc_fn = torch.nn.Parameter(source.bfloat16())

    exported = dict(
        checkpoint.export_hf_weights(
            model,
            DeepseekV4Config(hidden_size=128),
            ParallelState(),
        )
    )

    assert exported["hc_head_fn"].dtype == torch.float32
    torch.testing.assert_close(
        exported["hc_head_fn"], source.bfloat16().float(), rtol=0, atol=0
    )


def test_bf16_master_checkpoint_load_skips_fp8_scale_pairs() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    spec = DeepseekV4WeightSpec(config, source_block_fp8=False)
    native = "layers.0.self_attn.fused_wqa_wkv"
    names = spec._load_names(native)
    assert names == [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wkv.weight",
    ]
    target = torch.Size((384, 128))
    assert spec.hf_target_shape(native, 0, target) == torch.Size((256, 128))
    assert spec.hf_target_shape(native, 1, target) == torch.Size((128, 128))
    assert not spec.read_hf_source_raw(native, 0, names[0])

    master = spec.hf_to_native(
        native,
        [
            torch.ones(256, 128, dtype=torch.bfloat16),
            torch.full((128, 128), 2, dtype=torch.bfloat16),
        ],
    )
    assert master.dtype == torch.bfloat16
    assert master.shape == target
    assert torch.all(master[:256] == 1)
    assert torch.all(master[256:] == 2)


def test_layer2_compressor_and_indexer_release_names_and_dtypes() -> None:
    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=128,
        head_dim=128,
        index_head_dim=128,
        index_n_heads=2,
        num_attention_heads=2,
        num_hidden_layers=4,
        compress_ratios=[0, 0, 4, 128],
        num_hash_layers=3,
    )
    spec = DeepseekV4WeightSpec(config)
    assert spec._load_names(
        "layers.2.self_attn.compressor.fused_wkv_wgate"
    ) == [
        "layers.2.attn.compressor.wkv.weight",
        "layers.2.attn.compressor.wgate.weight",
    ]
    assert spec._load_names("layers.2.self_attn.indexer.weights_proj") == [
        "layers.2.attn.indexer.weights_proj.weight"
    ]
    assert spec._load_names("layers.2.self_attn.indexer.wq_b") == [
        "layers.2.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.wq_b.scale",
    ]
    assert spec._load_names(
        "layers.2.self_attn.indexer.compressor.fused_wkv_wgate"
    ) == [
        "layers.2.attn.indexer.compressor.wkv.weight",
        "layers.2.attn.indexer.compressor.wgate.weight",
    ]
    assert spec._load_names("layers.3.self_attn.compressor.ape") == [
        "layers.3.attn.compressor.ape"
    ]

    compressor = spec.hf_to_native(
        "layers.2.self_attn.compressor.fused_wkv_wgate",
        [
            torch.ones(256, 128, dtype=torch.bfloat16),
            torch.full((256, 128), 2, dtype=torch.bfloat16),
        ],
    )
    assert compressor.dtype == torch.bfloat16
    assert compressor.shape == (512, 128)

    indexer = spec.hf_to_native(
        "layers.2.self_attn.indexer.wq_b",
        [
            torch.ones(256, 128, dtype=torch.float8_e4m3fn),
            torch.full((2, 1), 3.0),
        ],
    )
    assert indexer.dtype == torch.bfloat16
    assert indexer.shape == (256, 128)
    assert torch.all(indexer == 3)


def test_fp8_loads_are_replica_local_to_preserve_source_scales() -> None:
    dense_group = object()
    ps = ParallelState(dp_cp_group=dense_group, ep_dp_group=object())

    assert (
        DeepseekV4WeightSpec.replica_group_for_load(
            "layers.0.mlp.experts.w13.0", ps
        )
        is None
    )
    assert (
        DeepseekV4WeightSpec.replica_group_for_load(
            "layers.0.self_attn.wq_b", ps
        )
        is None
    )
    assert (
        DeepseekV4WeightSpec.replica_group_for_load(
            "layers.0.input_layernorm.weight", ps
        )
        is dense_group
    )
    assert (
        DeepseekV4WeightSpec.expert_local_name(
            "layers.0.mlp.experts.w13.128", 0
        )
        == "layers.0.mlp.experts.w13.0"
    )


def test_router_checkpoint_names_follow_hash_prefix_semantics() -> None:
    config = DeepseekV4Config(
        num_hidden_layers=4,
        num_hash_layers=3,
        n_routed_experts=256,
    )
    spec = DeepseekV4WeightSpec(config)

    assert spec._load_names("layers.2.mlp.gate.tid2eid") == [
        "layers.2.ffn.gate.tid2eid"
    ]
    assert spec._load_names("layers.3.mlp.gate.expert_bias") == [
        "layers.3.ffn.gate.bias"
    ]
    assert spec._load_names("layers.3.mlp.gate.gate.weight") == [
        "layers.3.ffn.gate.weight"
    ]
    assert spec.hf_to_native(
        "layers.2.mlp.gate.tid2eid",
        [torch.zeros(8, 6, dtype=torch.int64)],
    ).dtype == torch.int32
    assert spec.hf_to_native(
        "layers.3.mlp.gate.expert_bias",
        [torch.zeros(256, dtype=torch.float32)],
    ).dtype == torch.float32


def test_export_and_forward_share_canonical_bf16_to_fp8_quantizer(
    monkeypatch,
) -> None:
    assert (
        checkpoint.quantize_block_fp8_weight
        is deployment_block_fp8.quantize_block_fp8_weight
    )
    calls: list[torch.Tensor] = []

    def cast(value, block_size, use_ue8m0):
        assert block_size == [128, 128]
        assert use_ue8m0 is False
        calls.append(value)
        return (
            value.float().clamp(-1, 1).to(torch.float8_e4m3fn),
            torch.ones(
                value.shape[0] // 128,
                value.shape[1] // 128,
                dtype=torch.float32,
            ),
        )

    def post_process(**kwargs):
        return kwargs["wq"], kwargs["ws"]

    entries = {
        ("vllm.utils.deep_gemm", "per_block_cast_to_fp8"): cast,
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "deepgemm_post_process_fp8_weight_block",
        ): post_process,
    }
    monkeypatch.setattr(
        deployment_block_fp8,
        "_import_attr",
        lambda module, name: entries[(module, name)],
    )

    config = DeepseekV4Config(
        hidden_size=128,
        q_lora_rank=256,
        head_dim=128,
        num_attention_heads=2,
    )
    spec = DeepseekV4WeightSpec(config)
    native = "layers.0.self_attn.fused_wqa_wkv"
    master = torch.randn(384, 128, dtype=torch.bfloat16)
    exported = spec.native_to_hf(native, master)
    assert [name for name, _ in exported] == [
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wq_a.scale",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wkv.scale",
    ]
    assert exported[0][1].dtype == torch.float8_e4m3fn
    assert exported[1][1].dtype == torch.float32

    deployment_block_fp8.pack_block_fp8_weight(
        torch.nn.Parameter(master[:256].clone())
    )
    assert len(calls) == 3
    assert torch.equal(calls[0], master[:256])
    assert torch.equal(calls[2], master[:256])

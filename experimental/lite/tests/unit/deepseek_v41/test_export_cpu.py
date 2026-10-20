# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release-format export: names, storage formats, and byte round trip of release codes."""
import pytest
import torch

from megatron.lite.model.deepseek_v41.lite.export import release_format, release_names, to_release
from megatron.lite.primitive.ckpt.hf_weights import _dequantize_block_scaled_tensor
from megatron.lite.primitive.quantization.block_fp8 import dequantize_block_fp8
from megatron.lite.primitive.quantization.mxfp4 import quantize_mxfp4

NAMES = [
    ("layers.3.ffn.experts.linear_fc1.weight17", ["layers.3.ffn.experts.17.w1.weight",
                                                  "layers.3.ffn.experts.17.w3.weight"], "mxfp4"),
    ("layers.3.ffn.experts.linear_fc2.weight17", ["layers.3.ffn.experts.17.w2.weight"], "mxfp4"),
    ("layers.3.ffn.shared_experts.linear_fc2.weight", ["layers.3.ffn.shared_experts.w2.weight"],
     "fp8_block"),
    ("layers.3.attention.linear_o_group_proj", ["layers.3.attn.wo_a.weight"], "fp8_block"),
    ("layers.3.attention.linear_q_up_proj.weight", ["layers.3.attn.wq_b.weight"], "fp8_block"),
    ("layers.3.attention.core_attention.native.indexer.linear_wq_b.linear.weight",
     ["layers.3.attn.indexer.wq_b.weight"], "fp8_block"),
    ("layers.3.attention.core_attention.native.compressor.linear_wkv.weight",
     ["layers.3.attn.compressor.wkv.weight"], torch.bfloat16),
    ("layers.3.attention.core_attention.native.attn_sink", ["layers.3.attn.attn_sink"],
     torch.float32),
    ("layers.3.ffn.router.expert_bias", ["layers.3.ffn.gate.bias"], torch.float32),
    ("layers.3.ffn.router.bias_vl", ["layers.3.ffn.gate.bias_vl"], torch.float32),
    ("layers.3.attn_mixes.mapping_proj.weight", ["layers.3.hc_attn_fn"], torch.float32),
    ("layers.3.ffn_mixes.alpha_post", ["layers.3.hc_ffn_scale#1"], torch.float32),
    ("engrams.14.wkv.weight", ["layers.14.engram.wkv.weight"], "fp8_block"),
    ("engrams.14.q_weight", ["layers.14.engram.q_weight"], torch.bfloat16),
    ("engrams.14.embed.embedding.weight", [], None),
    ("layers.3.attention.core_attention.native.compressor.ape", [], None),
    ("head.weight", ["head.weight"], torch.bfloat16),
]


@pytest.mark.parametrize("native,keys,fmt", NAMES)
def test_release_names_and_storage(native, keys, fmt):
    assert release_names(native) == keys
    for key in keys:
        assert release_format(key.split("#")[0]) == fmt


def test_fp4_negative_zero_survives_release_round_trip():
    # Release MXFP4 stores -0 as code 0x8; dequant->requant must keep the byte.
    codes = torch.tensor([[0x80, 0x08, 0x7F, 0xF1] * 8], dtype=torch.uint8).view(torch.int8)
    scale = torch.full((1, 2), 125, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    master = _dequantize_block_scaled_tensor(codes, scale.view(torch.uint8), torch.Size((1, 64)))
    again, again_scale = quantize_mxfp4(master.bfloat16())
    assert torch.equal(again.view(torch.uint8), codes.view(torch.uint8))
    assert torch.equal(again_scale.view(torch.uint8), scale.view(torch.uint8))


def test_block_fp8_export_decodes_to_master():
    master = (torch.randn(64, 96, generator=torch.Generator().manual_seed(0)) * 0.05).bfloat16()
    (key, codes), (scale_key, scale) = to_release("layers.0.attn.wq_a.weight", master)
    assert (key, scale_key) == ("layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale")
    assert codes.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
    decoded = dequantize_block_fp8(codes, scale, (32, 32))
    # Requantizing decoded codes may pick a 2x smaller UE8M0 scale (block max rounded
    # onto a power-of-two boundary): bytes differ, decoded values do not.
    (_, codes2), (_, scale2) = to_release("layers.0.attn.wq_a.weight", decoded.bfloat16())
    assert torch.equal(dequantize_block_fp8(codes2, scale2, (32, 32)), decoded)


def test_router_biases_are_the_exported_persistent_buffers():
    """bias_vl must reach the rollout on every resync: layerwise reload restores only what is sent."""
    from megatron.lite.model.deepseek_v41.lite.export import DeepseekV41WeightSpec

    keys = ("layers.3.ffn.router.expert_bias", "layers.3.ffn.router.bias_vl", "hash.x")
    plan = DeepseekV41WeightSpec(None).load_weight_map(None, None, keys)
    assert plan == {
        "layers.3.ffn.router.expert_bias": ["layers.3.ffn.gate.bias"],
        "layers.3.ffn.router.bias_vl": ["layers.3.ffn.gate.bias_vl"],
    }

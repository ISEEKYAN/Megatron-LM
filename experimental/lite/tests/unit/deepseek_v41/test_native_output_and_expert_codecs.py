# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""wo_a input and expert FC2 input use the rollout's own FP8 codec kernels."""
import os

import pytest
import torch


def _require_cuda(what):
    if os.environ.get("MEGATRON_LITE_REQUIRE_CUDA_TESTS") == "1":
        assert torch.cuda.is_available(), what
    elif not torch.cuda.is_available():
        pytest.skip(what)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("ratio", [0, 2])
def test_wo_a_input_is_native_inverse_rope_fp8(ratio):
    _require_cuda("native inverse RoPE FP8")
    from megatron.lite.primitive.kernels import mxfp8_block32
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.quantization import mxfp8
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    torch.manual_seed(7 + ratio)
    config = csa.CrossLayerAttentionConfig(heads=8, groups=2, o_rank=64)
    b, length, heads, dim = 1, 300, config.heads, config.head_dim
    output = torch.randn(b, length, heads, dim, device="cuda").bfloat16()
    output[..., :3, :] *= 1e-6  # blocks below the old 1e-4 amax floor
    positions = torch.arange(9000, 9000 + length, device="cuda")
    cache = csa.rope_cos_sin_cache(int(positions.max()) + 1, config, ratio, "cuda")
    values, scales = fused_inv_rope_fp8_quant(
        output.reshape(length, heads, dim),
        positions,
        cache,
        n_groups=config.groups,
        heads_per_group=heads // config.groups,
        quant_group_size=32,
        tma_aligned_scales=False,
    )
    act = csa._native_inverse_rope_activation(output, positions, config, ratio)
    width = heads * dim // config.groups
    assert torch.equal(
        act.values.view(torch.uint8).reshape(length, config.groups, width),
        values.contiguous().view(torch.uint8),
    )
    assert torch.equal(
        torch.exp2(act.scale.view(torch.uint8).float() - 127).reshape(
            length, config.groups, -1
        ),
        scales.contiguous(),
    )

    weight = torch.randn(config.groups, config.o_rank, width, device="cuda") / 64
    grouped = csa.rotate(
        output,
        positions,
        config,
        ratio,
        inverse=True,
        output_dtype=torch.float32,
        deployment_math=True,
    ).reshape(b, length, config.groups, width)
    actual = mxfp8.dynamic_fp8_grouped_linear(grouped, weight, activation=act)
    codes, wscale = quantize_block_fp8(weight, (32, 32), scale_format="e8m0")
    expected = (
        mxfp8_block32.batched_block32_mm(
            act.values.reshape(-1, config.groups, width),
            act.scale.reshape(-1, config.groups, width // 32),
            codes,
            wscale,
            torch.float32,
            grouped=True,
            weight_scale_group_rows=32,
        )
        .reshape(b, length, config.o_rank)
        .to(grouped.dtype)
    )
    assert torch.equal(actual.flatten(2), expected)


@pytest.mark.gpus(1, min_architecture="blackwell")
@pytest.mark.parametrize("limit", [None, 10.0])
def test_fc2_input_matches_rollout_fused_silu_quant(monkeypatch, limit):
    _require_cuda("native SiLU-mul FP8 quantization")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    from megatron.lite.primitive.quantization import w4a8_experts as w
    from vllm.model_executor.kernels.batch_invariant import (
        fused_silu_mul_per_token_group_quant_fp8,
    )

    torch.manual_seed(11)
    fc1_out = (torch.randn(4096, 2 * 2304, device="cuda") * 4).bfloat16()
    hidden = w._swiglu_bf16(fc1_out, limit)
    codes, scale = w.quantize_fp8_act(hidden, packed_ue8m0=True)
    ref_codes, ref_packed = fused_silu_mul_per_token_group_quant_fp8(
        fc1_out, use_ue8m0=True, clamp_limit=limit, masked_m=None
    )
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert torch.equal(scale, w._unpack_ue8m0_scales(ref_packed, scale.shape[1]))
    # Scales are the same powers of two as the scalar (non-packed) variant.
    _, scalar_scale = w.quantize_fp8_act(hidden)
    assert torch.equal(scale, scalar_scale)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in Q RoPE matches deployed CUDA rounding and has a finite analytic VJP."""
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.gpus(1)
@pytest.mark.parametrize('ratio', [0, 2])
def test_deployment_q_rope_native_and_vjp(v41_core_te, ratio):
    if os.environ.get('MEGATRON_LITE_REQUIRE_CUDA_TESTS') == '1':
        assert torch.cuda.is_available(), 'Q RoPE evidence requires CUDA'
    elif not torch.cuda.is_available():
        pytest.skip('CUDA native RoPE')
    from megatron.lite.primitive.modules.attention.csa import rotate
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers import rotary_embedding

    rope_module = rotary_embedding.deepseek_scaling_rope

    config = SimpleNamespace(
        rope_dim=64,
        rope_theta=10000,
        compress_rope_theta=160000,
        original_length=65536,
        beta_fast=32,
        beta_slow=1,
        factor=16,
    )
    with set_current_vllm_config(VllmConfig()), torch.device('cuda'):
        if ratio:
            rope = rope_module.DeepseekV4ScalingRotaryEmbedding(
                512,
                64,
                65536,
                160000,
                False,
                16,
                torch.bfloat16,
                beta_fast=32,
                beta_slow=1,
                mscale=0,
                mscale_all_dim=0,
            )
            cache = rope.cos_sin_cache
        else:
            frequency = 1 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
            angles = torch.arange(5).float()[:, None] * frequency
            cache = torch.cat((angles.cos(), angles.sin()), -1)
    torch.manual_seed(927)
    q = torch.randn(2, 5, 8, 512, device='cuda', dtype=torch.bfloat16)
    # Actual BF16 midpoint regression: complex multiply chooses the adjacent value.
    q[0, 4, 4, 470] = 0.61328125
    q[0, 4, 4, 471] = 0.14453125
    positions = torch.arange(5, device='cuda', dtype=torch.int64)
    pos = positions.repeat(2)
    kv = torch.zeros(10, 512, device='cuda', dtype=q.dtype)
    storage = torch.zeros(1, 128 * 2048, device='cuda', dtype=torch.uint8)
    native = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q.reshape(10, 8, 512),
        kv,
        storage,
        torch.arange(10, device='cuda'),
        pos,
        cache,
        64,
        1e-6,
        128,
        False,
        True,
        True,
        False,
    )[:, :8].reshape_as(q)
    actual = rotate(q, positions, config, ratio, deployment_math=True)
    assert torch.equal(actual, native)
    if ratio:
        assert float(actual[0, 4, 4, 471]) == 0.1845703125
        old = rotate(q, positions, config, ratio)
        assert float(old[0, 4, 4, 471]) == 0.18359375
    leaf = q.float().requires_grad_()
    result = rotate(leaf, positions, config, ratio, deployment_math=True)
    incoming = torch.randn_like(result)
    gradient = torch.autograd.grad(result, leaf, incoming)[0]
    cs, sn = cache[positions].chunk(2, -1)
    cs, sn = cs[None, :, None], sn[None, :, None]
    pair = incoming[..., -64:].unflatten(-1, (-1, 2))
    expected = torch.stack(
        (pair[..., 0] * cs + pair[..., 1] * sn, pair[..., 1] * cs - pair[..., 0] * sn),
        -1,
    ).flatten(-2)
    expected = torch.cat((incoming[..., :-64], expected), -1)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("num_tokens", [1, 32, 223])
def test_deployment_swa_kv_rope_native_codec(v41_core_te, num_tokens):
    """SWA KV must match the native BF16 RoPE then all-dimension FP8 codec."""
    if os.environ.get("MEGATRON_LITE_REQUIRE_CUDA_TESTS") == "1":
        assert torch.cuda.is_available(), "SWA KV RoPE evidence requires CUDA"
    elif not torch.cuda.is_available():
        pytest.skip("CUDA native SWA KV codec")
    from megatron.lite.primitive.modules.attention.csa import rotate
    from megatron.lite.primitive.quantization.mxfp8 import quantize_swa
    from vllm.models.deepseek_v41.common import ops as native_ops

    config = SimpleNamespace(rope_dim=64, rope_theta=10000)
    torch.manual_seed(93)
    kv = torch.randn(num_tokens, 512, device="cuda", dtype=torch.bfloat16)
    # Cancellation near an FP8 midpoint exposed the missing KV opt-in.
    kv[:, 488] = 0.259765625
    kv[:, 489] = -0.09716796875
    positions = torch.full((num_tokens,), 110, device="cuda", dtype=torch.int64)
    frequency = 1 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    angles = torch.arange(111, device="cuda").float()[:, None] * frequency
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1)
    block_size = 32
    num_blocks = (num_tokens + block_size - 1) // block_size
    storage = torch.zeros(
        num_blocks, block_size * 528, device="cuda", dtype=torch.uint8
    )
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        torch.zeros(num_tokens, 1, 512, device="cuda", dtype=kv.dtype),
        kv,
        storage,
        torch.arange(num_tokens, device="cuda"),
        positions,
        cos_sin,
        64,
        1e-6,
        block_size,
        False,
        True,
        False,
        False,
    )
    native = torch.empty(1, num_tokens, 512, device="cuda", dtype=kv.dtype)
    native_ops.dequantize_and_gather_k_cache(
        native,
        storage.view(num_blocks, block_size, 528),
        torch.tensor([num_tokens], device="cuda", dtype=torch.int32),
        None,
        torch.arange(num_blocks, device="cuda", dtype=torch.int32)[None],
        block_size,
        offset=0,
        use_fnuz=False,
    )
    actual = quantize_swa(
        rotate(kv[None], positions, config, 0, deployment_math=True)
    ).decoded
    assert torch.equal(actual, native)


def test_deployment_swa_kv_rope_vjp_and_default(v41_core_te):
    from megatron.lite.primitive.modules.attention.csa import rotate

    config = SimpleNamespace(rope_dim=64, rope_theta=10000)
    torch.manual_seed(91)
    leaf = torch.randn(1, 111, 512, requires_grad=True)
    positions = torch.arange(111)
    actual = rotate(leaf, positions, config, 0, deployment_math=True)
    incoming = torch.randn_like(actual)
    gradient = torch.autograd.grad(actual, leaf, incoming)[0]
    frequency = 1 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
    angles = positions.float()[:, None] * frequency
    cs, sn = angles.cos()[None], angles.sin()[None]
    pairs = incoming[..., -64:].unflatten(-1, (-1, 2))
    tail = torch.stack(
        (
            pairs[..., 0] * cs + pairs[..., 1] * sn,
            pairs[..., 1] * cs - pairs[..., 0] * sn,
        ),
        -1,
    ).flatten(-2)
    expected = torch.cat((incoming[..., :-64], tail), -1)
    torch.testing.assert_close(gradient, expected, atol=2e-6, rtol=2e-6)
    # Default recipe retains its complex multiply and FP32 output.
    default = rotate(leaf, positions, config, 0)
    phase = torch.polar(torch.ones_like(angles), angles)[None]
    original_tail = torch.view_as_complex(
        leaf[..., -64:].contiguous().unflatten(-1, (-1, 2))
    )
    expected_default = torch.cat(
        (leaf[..., :-64], torch.view_as_real(original_tail * phase).flatten(-2)), -1
    )
    assert torch.equal(default, expected_default)

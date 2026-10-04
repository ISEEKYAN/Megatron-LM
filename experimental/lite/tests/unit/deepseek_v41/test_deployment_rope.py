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
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekV4ScalingRotaryEmbedding,
    )

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
            rope = DeepseekV4ScalingRotaryEmbedding(
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

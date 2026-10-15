# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unequal Q/KV widths share native CUDA forwards and the declared VJP."""
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def native_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    os.environ.get("VLLM_BATCH_INVARIANT") != "1",
    reason="Native strict RMS tests require VLLM_BATCH_INVARIANT=1 before startup",
)
@pytest.mark.parametrize(
    "q_width,rows",
    [
        (1280, 1),
        (1280, 4),
        (1280, 32),
        (1280, 257),
        (1536, 1),
        (1536, 257),
        (128, 1),
        (128, 257),
    ],
)
def test_paired_norm_native_forward_and_fp32_master_vjp(
    v41_core_te, native_vllm_config, q_width, rows
):
    if os.environ.get("MEGATRON_LITE_REQUIRE_CUDA_TESTS") == "1":
        assert torch.cuda.is_available(), "Paired RMS evidence requires CUDA"
    elif not torch.cuda.is_available():
        pytest.skip("CUDA paired Q/KV norm")
    from megatron.lite.primitive.modules import deployment_math
    from megatron.lite.primitive.modules.attention import csa
    from vllm.model_executor.layers.layernorm import RMSNorm as InferenceRMSNorm

    torch.manual_seed(927)
    config = csa.CrossLayerAttentionConfig(
        dim=32, heads=4, groups=4, q_rank=q_width, o_rank=8
    )
    with torch.device("cuda"):
        attention = csa.CompressedSparseAttention(
            config,
            layer_idx=0,
            ps=SimpleNamespace(cp_size=1),
            candidate_mode="none",
            compress_ratio=0,
            kv_owner=0,
        )
    pair = (attention.q_norm, attention.kv_norm)
    values = [
        torch.randn(rows, width, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for width in (q_width, 512)
    ]
    for norm in pair:
        norm.deployment_math = True
        with torch.no_grad():
            # Distinct live FP32 master values that share a BF16 deployment value.
            norm.weight.copy_(0.75 + torch.rand_like(norm.weight) * 0.5 + 1e-6)
    native = []
    for norm, x in zip(pair, values):
        deployed = InferenceRMSNorm(x.shape[-1], config.eps).to(
            device="cuda", dtype=torch.bfloat16
        )
        deployed.weight.data.copy_(norm.weight.bfloat16())
        with torch.no_grad():
            native.append(deployed.forward_cuda(x.detach()))
    for norm, x, expected in zip(pair, values, native):
        actual = norm(x)
        assert torch.equal(actual, expected)
        incoming = torch.randn_like(actual)
        gradient = torch.autograd.grad(actual, (x, norm.weight), incoming)
        reference = torch.nn.functional.rms_norm(
            x,
            norm.normalized_shape,
            deployment_math.decoded_bf16_master(norm.weight),
            norm.eps,
        )
        required = torch.autograd.grad(
            reference, (x, norm.weight), incoming.to(reference.dtype)
        )
        assert gradient[1].dtype == torch.float32
        assert all(torch.isfinite(g).all() for g in gradient)
        assert all(torch.equal(g, r) for g, r in zip(gradient, required))
        # The native CUDA provider affects only the deployment forward.
        norm.deployment_math = False
        ordinary = norm(x)
        baseline = torch.nn.functional.rms_norm(
            x, norm.normalized_shape, norm.weight, norm.eps
        )
        assert torch.equal(ordinary, baseline)

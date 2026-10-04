# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Grouped output projection preserves the deployment-codec/FP32-master boundary."""
import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='deployment FP8 GEMM requires CUDA'
)
@pytest.mark.parametrize('groups', [1, 2])
def test_grouped_output_uses_fp8_with_fp32_masters(v41_core_te, groups):
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.quantization import mxfp8

    torch.manual_seed(731)
    cfg = csa.CrossLayerAttentionConfig(
        dim=32,
        heads=4,
        head_dim=32,
        rope_dim=4,
        q_rank=32,
        o_rank=32,
        groups=groups,
        index_heads=1,
        index_dim=32,
        linear_fp8=True,
        swa_fp8=False,
        main_qat=False,
        index_qat=False,
    )
    model = csa.CompressedSparseAttention(
        cfg,
        ps=None,
        layer_idx=0,
        kv_owner=None,
        index_owner=None,
        candidate_mode='none',
        compress_ratio=0,
    )
    model = model.cuda()
    model.wo_a.weight.data = model.wo_a.weight.data.float()
    model.wo_a.native_fp32 = True
    assert model.wo_a.fp8
    # A value which changes under the deployment codec makes bypass observable.
    with torch.no_grad():
        model.wo_a.weight[0, 0] = 0.7501
    calls = []

    def project(x, weight):
        calls.append((x.detach().clone(), weight.detach().clone()))
        return mxfp8.dynamic_fp8_linear(x, weight)

    model.wo_a.fp8_operator = project
    x = torch.randn(1, 3, 32, device='cuda').bfloat16().requires_grad_()
    y, _ = model(x, csa.AttentionState())
    assert len(calls) == groups
    for i, (operand, weight) in enumerate(calls):
        assert operand.shape == (1, 3, cfg.heads * cfg.head_dim // groups)
        assert operand.dtype == torch.float32
        assert weight.dtype == torch.float32
        assert torch.equal(
            weight, model.wo_a.weight[i * cfg.o_rank : (i + 1) * cfg.o_rank]
        )
    y.float().square().mean().backward()
    assert model.wo_a.weight.grad.dtype == torch.float32
    assert torch.isfinite(model.wo_a.weight.grad).all()
    assert torch.count_nonzero(model.wo_a.weight.grad) > 0
    assert x.grad.dtype == torch.bfloat16


def test_inverse_rope_fp8_midpoint_preserves_fp32(v41_core_te):
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.quantization import mxfp8

    cfg = csa.CrossLayerAttentionConfig(head_dim=32, rope_dim=4)
    x = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
    x[..., -4:] = torch.tensor([0.5, 0.53125, 0.0, 4.0])
    positions = torch.tensor([1])
    precise = csa.rotate(x, positions, cfg, 0, inverse=True, output_dtype=torch.float32)
    old = csa.rotate(x, positions, cfg, 0, inverse=True)
    # Inverse rotation yields 0.71718258, which rounds to 0.71875 in BF16.
    # With the row's 1/64 UE8M0 scale these straddle an E4M3 midpoint.
    assert precise.dtype == torch.float32
    assert old.dtype == torch.bfloat16
    assert (
        mxfp8.quantize_linear_activation(precise)
        .values.view(torch.uint8)[0, 0, -4]
        .item()
        == 99
    )
    assert (
        mxfp8.quantize_linear_activation(old).values.view(torch.uint8)[0, 0, -4].item()
        == 100
    )

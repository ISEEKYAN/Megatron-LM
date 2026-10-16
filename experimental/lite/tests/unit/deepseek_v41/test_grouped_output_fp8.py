# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Grouped output projection preserves the deployment-codec/FP32-master boundary."""
import pytest
import torch


@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('weight_dtype', [torch.float32, torch.bfloat16])
def test_grouped_output_uses_fp8_with_master_or_reference_weights(
    v41_core_te, groups, weight_dtype
):
    """CPU integration checks codec ownership, not native FP8 numerics.

    Other projections use CPU reference linears. The actual CSA forward must
    invoke the supplied grouped codec once and retain FP32 activations and the
    original master/reference dtype. Native packets/kernel/VJP have CUDA tests.
    """
    from types import MethodType

    from megatron.lite.model.deepseek_v41 import codecs
    from megatron.lite.primitive.modules import native_fp32_linear
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.quantization import mxfp8

    assert (
        codecs.attention_codecs()['linear_grouped'] is mxfp8.dynamic_fp8_grouped_linear
    )
    torch.manual_seed(731)
    cfg = csa.CrossLayerAttentionConfig(
        dim=32,
        heads=4,
        head_dim=512,
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
    model.deployment_math = True  # CPU attention uses the Torch reference branch.
    model.wo_a.weight.data = model.wo_a.weight.data.to(weight_dtype)
    calls = []

    def reference(self, value):
        return torch.nn.functional.linear(value.float(), self.weight.float())

    for module in model.modules():
        if isinstance(module, native_fp32_linear.Linear) and module is not model.wo_a:
            module.forward = MethodType(reference, module)

    def grouped_codec(operand, weight):
        calls.append((operand.detach().clone(), weight.detach().clone()))
        return torch.einsum('...gk,gnk->...gn', operand, weight.float())

    model.codecs = dict(codecs.attention_codecs(), linear_grouped=grouped_codec)
    value = torch.randn(1, 3, 32).bfloat16().requires_grad_()
    output, _ = model(value, csa.AttentionState())
    assert len(calls) == 1
    operand, weight = calls[0]
    assert operand.shape == (1, 3, groups, cfg.heads * cfg.head_dim // groups)
    assert operand.dtype == torch.float32
    assert weight.dtype == weight_dtype
    assert torch.equal(weight, model.wo_a.weight.view(groups, cfg.o_rank, -1))
    output.float().square().mean().backward()
    assert model.wo_a.weight.grad.dtype == weight_dtype
    assert torch.isfinite(model.wo_a.weight.grad).all()
    assert torch.count_nonzero(model.wo_a.weight.grad) > 0
    assert value.grad.dtype == torch.bfloat16


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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='deployment FP8 GEMM requires CUDA'
)
def test_fp32_activation_bf16_reference_matches_master_codec_and_vjp(v41_core_te):
    from megatron.lite.primitive.quantization import mxfp8

    torch.manual_seed(913)
    x = torch.randn(2, 3, 64, device='cuda', dtype=torch.float32).requires_grad_()
    weight = torch.randn(32, 64, device='cuda', dtype=torch.bfloat16).requires_grad_()
    output = mxfp8.dynamic_fp8_linear(x, weight, deployment_math=True)
    master_output = mxfp8.dynamic_fp8_linear(x.detach(), weight.detach().float())
    assert output.dtype == torch.float32
    assert torch.equal(output, master_output)
    cotangent = torch.randn_like(output)
    output.backward(cotangent)
    encoded, scale = mxfp8.quantize_block_fp8(
        weight.detach(), (32, 32), scale_format='e8m0'
    )
    decoded = mxfp8.dequantize_block_fp8(encoded, scale, (32, 32)).float()
    activation = mxfp8.quantize_linear_activation(x.detach()).decoded
    with torch.autocast(device_type='cuda', enabled=False):
        expected_x = (cotangent.reshape(-1, 32) @ decoded).reshape_as(x)
        expected_weight = (
            cotangent.reshape(-1, 32).T @ activation.reshape(-1, 64)
        ).bfloat16()
    assert torch.equal(x.grad, expected_x)
    assert torch.equal(weight.grad, expected_weight)
    assert weight.grad.dtype == torch.bfloat16


def test_mixed_activation_dtype_is_deployment_only():
    """Execute the public validation boundary; CPU never pretends to run GEMM."""
    from megatron.lite.primitive.quantization import mxfp8

    x = torch.randn(2, 64, dtype=torch.float32)
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    for kwargs in ({}, {'deployment_math': False}):
        with pytest.raises(ValueError, match='compute dtype'):
            mxfp8.dynamic_fp8_linear(x, weight, **kwargs)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        mxfp8.dynamic_fp8_linear(x, weight, deployment_math=True)
    # An FP32 master remains supported without deployment opt-in.
    with pytest.raises(RuntimeError, match='requires CUDA'):
        mxfp8.dynamic_fp8_linear(x.bfloat16(), weight.float())

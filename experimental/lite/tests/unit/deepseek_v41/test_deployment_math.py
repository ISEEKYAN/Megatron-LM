# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Visible CUDA math is explicit; its reference VJP owns FP32 master gradients."""
import pytest
import torch


def test_visible_vjp_does_not_truncate_master_gradient():
    from megatron.lite.primitive.modules.deployment_math import visible_forward

    torch.manual_seed(603)
    x = torch.randn(7, 32).bfloat16().requires_grad_()
    weight = torch.nn.Parameter(torch.randn(16, 32))
    incoming = torch.randn(7, 16).bfloat16()
    output = visible_forward(
        lambda a, b: torch.nn.functional.linear(a, b.bfloat16()),
        lambda a, b: torch.nn.functional.linear(a.float(), b),
        x,
        weight,
    )
    output.backward(incoming)
    expected = incoming.float().T @ x.detach().float()
    assert torch.equal(weight.grad, expected)
    assert weight.grad.dtype == torch.float32
    assert not torch.equal(weight.grad, weight.grad.bfloat16().float())
    assert x.grad.dtype == torch.bfloat16


def test_visible_multi_output_unused_gradient():
    from megatron.lite.primitive.modules.deployment_math import visible_forward

    x = torch.tensor([2.0, 3.0], requires_grad=True)
    first, _ = visible_forward(
        lambda a: (a.square(), a * 3), lambda a: (a.square(), a * 3), x
    )
    first.sum().backward()
    assert torch.equal(x.grad, torch.tensor([4.0, 6.0]))


def test_deployment_math_rejects_fp32_residuals(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from test_w4a8_fp32 import tiny_config

    with pytest.raises(ValueError, match='Deployment math requires'):
        protocol.build_model(
            tiny_config(),
            impl_cfg=protocol.ImplConfig(
                device='cpu', quantized=False, deployment_math=True, dtype=torch.float32
            ),
        )


def test_norm_decodes_updated_master_and_keeps_fp32_gradient():
    from megatron.lite.primitive.modules.attention.mhc import RMSNorm

    torch.manual_seed(611)
    norm = RMSNorm(32, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(1 + torch.randn(32) * 0.01)
    x = torch.randn(3, 32).bfloat16()
    incoming = torch.randn_like(x)
    norm.deployment_math = True
    y = norm(x)
    decoded = norm.weight.detach().bfloat16().float().requires_grad_()
    expected = torch.nn.functional.rms_norm(x, (32,), decoded, norm.eps)
    assert torch.equal(y, expected)
    y.backward(incoming)
    expected.backward(incoming)
    assert torch.equal(norm.weight.grad, decoded.grad)
    assert norm.weight.grad.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deployment router')
def test_router_native_slots_weights_and_fp32_reference_gradient():
    from megatron.lite.primitive.modules.deployment_math import router_topk
    from vllm import _custom_ops as ops

    torch.manual_seed(719)
    for experts in (4, 16):
        logits = torch.randn(7, experts, device='cuda', requires_grad=True)
        bias = torch.linspace(-0.03, 0.03, experts, device='cuda')
        weights, indices = router_topk(logits, bias, topk=2, scaling_factor=1.5)
        expected_weights = torch.empty_like(weights)
        expected_indices = torch.empty_like(indices, dtype=torch.int32)
        ops.topk_hash_softplus_sqrt(
            expected_weights,
            expected_indices,
            torch.empty_like(expected_indices),
            logits.detach(),
            renormalize=True,
            routed_scaling_factor=1.5,
            e_score_correction_bias=bias,
        )
        assert torch.equal(indices, expected_indices.long())
        assert torch.equal(weights, expected_weights)
        incoming = torch.randn_like(weights)
        weights.backward(incoming)
        leaf = logits.detach().clone().requires_grad_()
        selected = torch.nn.functional.softplus(leaf).sqrt().gather(1, indices)
        (selected / selected.sum(-1, keepdim=True) * 1.5).backward(incoming)
        assert torch.equal(logits.grad, leaf.grad)
        assert logits.grad.dtype == torch.float32
        assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA deployment QKV RMS')
@pytest.mark.parametrize('width', [128, 512, 1280, 2048])
def test_qkv_norm_native_forward_fp32_vjp_and_process_state(width):
    from megatron.lite.primitive.modules import deployment_math
    from megatron.lite.primitive.modules.attention.mhc import RMSNorm
    from vllm.model_executor.determinism import batch_invariant
    from vllm.models.common.ops import fused_q_kv_rmsnorm

    torch.manual_seed(942)
    norm = RMSNorm(width, eps=1e-6, device='cuda', dtype=torch.float32)
    norm.deployment_math = True
    with torch.no_grad():
        norm.weight.copy_(1 + torch.randn_like(norm.weight) * 0.03)
    x = torch.randn(2, 13, width, device='cuda').bfloat16().requires_grad_()
    incoming = torch.randn_like(x)
    before = norm(x)
    native, _ = fused_q_kv_rmsnorm(
        x.detach().reshape(-1, width),
        x.detach().reshape(-1, width),
        norm.weight.detach().bfloat16(),
        norm.weight.detach().bfloat16(),
        norm.eps,
    )
    assert torch.equal(before.detach(), native.reshape_as(x))
    batch_invariant.init_batch_invariance()
    after = norm(x)
    assert torch.equal(before, after)
    assert torch.equal(after[:1], norm(x[:1]))
    actual = torch.autograd.grad(after, (x, norm.weight), incoming)
    values = x.detach().requires_grad_()
    master = norm.weight.detach().requires_grad_()
    reference = torch.nn.functional.rms_norm(
        values, (width,), deployment_math.decoded_bf16_master(master), norm.eps
    )
    expected = torch.autograd.grad(reference, (values, master), incoming)
    for got, wanted in zip(actual, expected, strict=True):
        assert torch.isfinite(got).all()
        assert torch.equal(got, wanted)
    assert actual[1].dtype == torch.float32
    assert not torch.equal(actual[1], actual[1].bfloat16().float())

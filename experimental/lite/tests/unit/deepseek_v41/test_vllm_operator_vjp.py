# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU proofs for owned derivatives; deployment kernel forward is a GPU gate."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.linear import _BF16Linear, _MXFP8Linear, linear
from megatron.lite.model.deepseek_v41.vllm.norm import _RMSNorm


@pytest.mark.parametrize("shape", [(3, 32), (2, 3, 32), (0, 32)])
def test_bf16_linear_owned_vjp_uses_rounded_weight(shape):
    """Backward GEMMs take BF16 operands (the rounded weight, input and upstream gradient)
    with FP32 accumulation, as TE/Megatron dgrad/wgrad do."""
    x = torch.randn(shape).bfloat16().float().requires_grad_()
    weight = torch.randn(64, 32, requires_grad=True)
    compute = weight.bfloat16()
    surrogate = weight + (compute.float() - weight).detach()
    out = x @ surrogate.T
    grad = torch.randn_like(out).bfloat16().float()
    expected = torch.autograd.grad(out, (x, weight), grad)
    ctx = SimpleNamespace(saved_tensors=(x.detach(), compute.detach()), owners=None, like=weight)
    dx, dw, _ = _BF16Linear.backward(ctx, grad)
    torch.testing.assert_close(dx, expected[0].bfloat16().float())
    torch.testing.assert_close(dw, expected[1])


@pytest.mark.parametrize("parts", [1, 2])
def test_linear_weight_gradient_lands_in_fp32_main_grad(parts):
    """With DDP's FP32 main_grad on every stacked owner, the FP32 weight gradient is added
    there (no BF16 rounding) and the returned gradient is DDP's placeholder."""
    from megatron.lite.model.deepseek_v41.vllm.linear import accumulate_wgrad

    owners = [torch.nn.Parameter(torch.zeros(4 * (i + 1), 8).bfloat16()) for i in range(parts)]
    for owner in owners:
        owner.main_grad = torch.full(owner.shape, 0.5)
        owner.grad_added_to_main_grad = False
    like = owners[0] if parts == 1 else torch.cat(owners)
    dw = torch.randn(like.shape) * 1e-3 + 1
    placeholder = accumulate_wgrad(tuple(owners), dw, like)
    assert placeholder.shape == like.shape and placeholder.dtype == like.dtype
    start = 0
    for owner in owners:
        assert owner.grad_added_to_main_grad
        torch.testing.assert_close(owner.main_grad, 0.5 + dw[start : start + owner.shape[0]])
        start += owner.shape[0]
    assert not torch.equal(dw, dw.bfloat16().float())  # would not survive a BF16 grad


def test_mxfp8_owned_vjp_uses_actual_decoded_operands():
    x, weight = torch.randn(7, 32).bfloat16(), torch.randn(64, 32).bfloat16()
    grad = torch.randn(7, 64).bfloat16()
    ctx = SimpleNamespace(
        saved_tensors=(x, weight), input_shape=x.shape, weight_dtype=torch.float32
    )
    dx, dw, *ignored = _MXFP8Linear.backward(ctx, grad)
    assert ignored == [None, None]
    torch.testing.assert_close(dx, (grad.float() @ weight.float()).bfloat16())
    torch.testing.assert_close(dw, grad.float().T @ x.float())


def test_rms_owned_vjp_matches_fp32_reference_with_rounded_weight():
    x, weight = torch.randn(2, 3, 32, requires_grad=True), torch.randn(32, requires_grad=True)
    rounded = weight + (weight.bfloat16().float() - weight).detach()
    out = torch.nn.functional.rms_norm(x, (32,), rounded, 1e-6)
    grad = torch.randn_like(out)
    expected = torch.autograd.grad(out, (x, weight), grad)
    ctx = SimpleNamespace(saved_tensors=(x.detach(), weight.detach()), eps=1e-6)
    dx, dw, ignored = _RMSNorm.backward(ctx, grad)
    assert ignored is None
    for value, reference in zip((dx, dw), expected):
        torch.testing.assert_close(value, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fp8", [False, True])
def test_no_deployment_cpu_fallback(fp8):
    with pytest.raises(RuntimeError, match="CUDA"):
        linear(torch.zeros(2, 32), torch.zeros(64, 32), fp8=fp8)


def test_mhc_adapter_reads_actual_native_parameter_contract():
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.lite.model.deepseek_v41.vllm.mhc import joint

    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=1,
        num_residual_streams=4,
        use_fused_mhc=False,
    )
    mixes = HyperConnectionModule(config, 1)
    with pytest.raises(RuntimeError, match="CUDA"):
        joint(torch.zeros(2, 1, 4, 32), torch.zeros(2, 1, 4), torch.ones(32), mixes)


@pytest.mark.parametrize("fused", [True, False])
def test_wo_a_owned_vjp_matches_autograd_on_decoded_operands(fused):
    """wo_a backward: d(dequantized core output) into the carrier, d(master) through the
    column permutation of the fused layout (identity STE on both quantizations)."""
    from megatron.lite.model.deepseek_v41.vllm.attention import _WoA, dequantize_output

    tokens, groups, rank, width = 3, 8, 16, 4096
    data = torch.randn(tokens, groups, width).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 131, (tokens, groups, 128), dtype=torch.uint8).view(torch.int32)
    weight = torch.randn(groups * rank, width)
    perm = torch.randperm(width) if fused else None
    decoded = (weight if perm is None else weight[:, perm]).bfloat16()
    grad = torch.randn(tokens, groups, rank).bfloat16().float()  # backward GEMMs take BF16
    ctx = SimpleNamespace(
        saved_tensors=(data, scale, decoded, perm), weight_shape=weight.shape, weight=weight
    )
    _, _, d_out, d_weight, _ = _WoA.backward(ctx, grad)
    out = dequantize_output(data, scale).requires_grad_(True)
    master = weight.clone().requires_grad_(True)
    layout = master if perm is None else master[:, perm]
    layout = layout + (decoded.float() - layout).detach()
    z = torch.einsum("bhr,hdr->bhd", out, layout.view(groups, rank, width))
    expected_out, expected_weight = torch.autograd.grad(z, (out, master), grad)
    torch.testing.assert_close(d_out, expected_out.bfloat16())
    torch.testing.assert_close(d_weight, expected_weight, rtol=1e-5, atol=1e-4)


def test_engram_qk_grads_reach_fp32_masters_unrounded():
    """The gate saves BF16 q/k (what vLLM uses); their gradient is a reduction over every
    token and must reach the FP32 masters without a BF16 rounding."""
    from megatron.lite.model.deepseek_v41.vllm.engram import _Gate, reference

    torch.manual_seed(0)
    tokens, copies, dim = 64, 4, 128
    hidden = torch.randn(tokens, copies, dim).bfloat16()
    kv = torch.randn(tokens, (copies + 1) * dim).bfloat16()
    q, k = (torch.randn(dim).bfloat16() for _ in range(2))
    grad = torch.randn(tokens, copies, dim)
    ctx = SimpleNamespace(
        saved_tensors=(hidden, kv, q, k, None),
        needs_input_grad=(False, False, True, True, False, False),
        eps=1e-6,
        master_dtypes=(torch.float32, torch.float32),
    )
    _, _, dq, dk, _, _ = _Gate.backward(ctx, grad)
    assert dq.dtype == dk.dtype == torch.float32
    assert not torch.equal(dq, dq.bfloat16().float()) and not torch.equal(dk, dk.bfloat16().float())

    leaves = [t.double().requires_grad_() for t in (q, k)]
    out = reference(hidden.double(), kv.double(), *leaves, 1e-6, None)
    ref = torch.autograd.grad(out, leaves, grad.double())
    for got, want in zip((dq, dk), ref):
        error = (got.double() - want).norm() / want.norm()
        rounded = (got.bfloat16().double() - want).norm() / want.norm()
        assert error < 1e-5 < rounded, (error, rounded)

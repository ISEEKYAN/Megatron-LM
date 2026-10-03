# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical contract of block32 FP8 Linear and native FP32 wgrad.

CPU cases are bitwise (``torch.equal``). ``torch._scaled_mm`` is CUDA-only, so
the CPU GEMM cases replace it with a recording E4M3 emulation that checks the
operands and call count; the real kernel runs only in the GPU-marked case.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from megatron.lite.primitive.modules import native_fp32_linear as nfl
from megatron.lite.primitive.quantization import mxfp8
from megatron.lite.primitive.quantization.block_fp8 import (
    dequantize_block_fp8,
    quantize_block_fp8,
)


@pytest.fixture
def scaled_mm_calls(monkeypatch):
    calls = []

    def emulate(a, b, scale_a, scale_b, *, out_dtype):
        assert a.dtype == b.dtype == torch.float8_e4m3fn
        assert a.shape[1] == b.shape[0] == mxfp8.BLOCK and b.stride(0) == 1
        assert out_dtype == torch.float32
        assert scale_a.item() == scale_b.item() == 1.0
        calls.append((tuple(a.shape), tuple(b.shape)))
        return a.float() @ b.float()

    monkeypatch.setattr(torch, "_scaled_mm", emulate)
    return calls


def _integer_operands(rows=8, out=64, k=96, dtype=torch.bfloat16):
    # Integers in [-8, 8] are exact in E4M3 after a power-of-two block scale, and
    # every partial sum is exact in FP32, so all GEMM orders agree bitwise.
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(-8, 9, (2, rows, k), generator=gen).to(dtype)
    w = torch.randint(-8, 9, (out, k), generator=gen).float()
    return x, w


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_block32_codec_reuses_block_fp8_e8m0(dtype):
    x = torch.randn(4, 3, 64, generator=torch.Generator().manual_seed(1)).to(dtype)
    got = mxfp8.quantize_block32(x)
    values, scale = quantize_block_fp8(x, (1, 32), scale_format="e8m0")
    assert got.scale.dtype == torch.float8_e8m0fnu and got.scale.shape == (4, 3, 2)
    assert torch.equal(got.values.view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(got.scale.view(torch.uint8), scale.view(torch.uint8))
    decoded = dequantize_block_fp8(values, scale, (1, 32)).to(dtype)
    assert got.decoded.dtype == dtype and torch.equal(got.decoded, decoded)
    weight = mxfp8.quantize_block32(x[0, :, :].repeat(32, 1), mxfp8.WEIGHT_BLOCK)
    assert weight.scale.shape == (3, 2)


def test_fp8_gemm_calls_scaled_mm_per_k_block_with_fp32_scales(scaled_mm_calls):
    x, w = _integer_operands()
    flat = x.reshape(-1, x.shape[-1])
    a = mxfp8.quantize_block32(flat)
    b = mxfp8.quantize_block32(w, mxfp8.WEIGHT_BLOCK)
    out = mxfp8._fp8_gemm(a.values, a.scale, b.values, b.scale)
    assert scaled_mm_calls == [((16, 32), (32, 64))] * 3
    assert out.dtype == torch.float32
    assert torch.equal(out, flat.double().matmul(w.double().T).float())


def test_real_and_diagnostic_paths_share_codec_and_differ_from_floating(
    scaled_mm_calls,
):
    x, w = _integer_operands()
    real = mxfp8._DynamicFP8Linear.apply(x, w, False)  # bypasses the CUDA guard
    diagnostic = mxfp8.dynamic_fp8_linear(x, w, diagnostic=True)
    assert len(scaled_mm_calls) == 3 and real.dtype == torch.bfloat16
    assert torch.equal(real, diagnostic)

    # Random operands: diagnostic equals an FP32 GEMM on decoded operands
    # bitwise; the codec error versus unquantized floating is nonzero.
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(5, 64, generator=gen).bfloat16()
    w = torch.randn(32, 64, generator=gen)
    decoded_x = mxfp8.quantize_block32(x).decoded.float()
    decoded_w = mxfp8.quantize_block32(w, mxfp8.WEIGHT_BLOCK).decoded
    diagnostic = mxfp8.dynamic_fp8_linear(x, w, diagnostic=True)
    assert torch.equal(diagnostic, F.linear(decoded_x, decoded_w).bfloat16())
    assert not torch.equal(diagnostic, F.linear(x.float(), w).bfloat16())


def test_fp8_backward_is_native_fp32_wgrad():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 3, 64, generator=gen).bfloat16().requires_grad_()
    w = torch.randn(32, 64, generator=gen).requires_grad_()
    grad = torch.randn(2, 3, 32, generator=gen).bfloat16()
    mxfp8.dynamic_fp8_linear(x, w, diagnostic=True).backward(grad)
    decoded_x = mxfp8.quantize_block32(x.detach().reshape(-1, 64)).decoded
    decoded_w = mxfp8.quantize_block32(w.detach(), mxfp8.WEIGHT_BLOCK).decoded
    flat = grad.reshape(-1, 32).float()
    assert w.grad.dtype == torch.float32 and x.grad.dtype == torch.bfloat16
    assert torch.equal(w.grad, flat.T @ decoded_x.float())
    assert torch.equal(x.grad, (flat @ decoded_w).reshape(x.shape).bfloat16())


def test_native_fp32_wgrad_is_not_a_bf16_product_cast_up():
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(4, 7, 96, generator=gen).bfloat16().requires_grad_()
    w = torch.randn(64, 96, generator=gen).requires_grad_()
    grad = torch.randn(4, 7, 64, generator=gen).bfloat16()
    out = nfl.native_fp32_linear(x, w)
    assert torch.equal(out, F.linear(x.detach(), w.detach().bfloat16()))
    out.backward(grad)
    expected = grad.reshape(-1, 64).float().T @ x.detach().reshape(-1, 96).float()
    assert w.grad.dtype == torch.float32 and torch.equal(w.grad, expected)
    assert torch.equal(x.grad, grad @ w.detach().bfloat16())

    reference = w.detach().clone().requires_grad_()
    F.linear(x.detach(), reference.bfloat16()).backward(grad)
    assert not torch.equal(reference.grad, w.grad)  # BF16 dW, then cast to FP32


def test_provider_registry_and_fp32_master_restore():
    assert nfl.linear_provider("default") is torch.nn.Linear
    model = torch.nn.Sequential(
        nfl.linear_provider("block32_fp8")(64, 32, bias=False),
        nfl.linear_provider("native_fp32")(32, 48),
        torch.nn.Linear(48, 8, bias=False),
    ).to(torch.bfloat16)
    plain = model[2].weight
    before = plain.detach().clone()
    assert nfl.restore_fp32_masters(model) is model
    assert [m.weight.dtype for m in model] == [torch.float32] * 2 + [torch.bfloat16]
    assert model[2].weight is plain and torch.equal(plain, before)
    assert [m.mode for m in model[:2]] == ["block32_fp8", "native_fp32"]


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda: nfl.linear_provider("fp4"), ValueError),
        (lambda: nfl.Linear(64, 32, mode="default"), ValueError),
        (lambda: nfl.Linear(64, 32, mode="native_fp32", bias=True), ValueError),
        (lambda: nfl.Linear(48, 32, mode="block32_fp8"), ValueError),
        (lambda: nfl.Linear(64, 40, mode="block32_fp8"), ValueError),
        (
            lambda: nfl.native_fp32_linear(
                torch.ones(2, 4), torch.ones(4, 4).bfloat16()
            ),
            TypeError,
        ),
        (lambda: mxfp8.quantize_block32(torch.ones(2, 48)), ValueError),
        (
            lambda: mxfp8.quantize_block32(torch.ones(2, 32, dtype=torch.int32)),
            TypeError,
        ),
        (
            lambda: mxfp8.dynamic_fp8_linear(torch.ones(2, 32), torch.ones(48, 32)),
            ValueError,
        ),
        (
            lambda: mxfp8.dynamic_fp8_linear(torch.ones(2, 64), torch.ones(32, 32)),
            ValueError,
        ),
        (
            lambda: mxfp8.dynamic_fp8_linear(
                torch.ones(2, 32).half(), torch.ones(32, 32).bfloat16()
            ),
            TypeError,
        ),
        (
            lambda: mxfp8.dynamic_fp8_linear(torch.ones(2, 32), torch.ones(32, 32)),
            RuntimeError,
        ),
        (
            lambda: nfl.Linear(64, 32, mode="block32_fp8")(torch.ones(2, 64)),
            RuntimeError,
        ),
        (
            lambda: nfl.Linear(64, 32, mode="block32_fp8").bfloat16()(
                torch.ones(2, 64)
            ),
            TypeError,
        ),
    ],
)
def test_rejections(call, error):
    with pytest.raises(error):
        call()


@pytest.mark.gpus(1)
def test_cuda_scaled_mm_matches_diagnostic_on_exact_operands(monkeypatch):
    x, w = _integer_operands()
    x, w = x.cuda().requires_grad_(), w.cuda().requires_grad_()
    calls = []
    real_scaled_mm = torch._scaled_mm

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return real_scaled_mm(*args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", spy)
    layer = nfl.Linear(96, 64, mode="block32_fp8").cuda()
    with torch.no_grad():
        layer.weight.copy_(w)
    real = layer(x)
    assert len(calls) == 3
    assert torch.equal(real, mxfp8.dynamic_fp8_linear(x, w, diagnostic=True))
    real.float().sum().backward()
    assert layer.weight.grad.dtype == torch.float32
    decoded_x = mxfp8.quantize_block32(x.detach().reshape(-1, 96)).decoded.float()
    decoded_w = mxfp8.quantize_block32(w.detach(), mxfp8.WEIGHT_BLOCK).decoded
    grad = torch.ones_like(real).reshape(-1, 64).float()
    assert torch.equal(layer.weight.grad, grad.T @ decoded_x)
    assert torch.equal(x.grad, (grad @ decoded_w).reshape(x.shape).bfloat16())

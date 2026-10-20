# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Ratio-one compressor rows and the independently evaluated RMS derivative."""
import torch
from test_vllm_attention_math import compress_vjp


def test_ratio_one_rows_and_norm_vjp():
    torch.manual_seed(35)
    raw = torch.randn(7, 512, requires_grad=True)
    weight = torch.randn(512, requires_grad=True)
    grad = torch.randn_like(raw)
    # The kernel reads the BF16 norm weight; its rounding is an identity STE.
    compute = weight + (weight.bfloat16().float() - weight).detach()
    output = torch.nn.functional.rms_norm(raw, (512,), compute, 1e-6)
    expected = torch.autograd.grad(output, (raw, weight), grad)
    rows, *actual = compress_vjp(raw.detach(), weight.detach(), torch.arange(7), 1, grad, 1e-6)
    assert torch.equal(rows, torch.arange(7))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

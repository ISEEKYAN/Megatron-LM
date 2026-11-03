# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Existing-kernel adapter contracts; GPU performance is a separate proxy gate."""

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import mhc


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('rows,width', [(0, 16), (3, 16), (2, 512)])
def test_rms_math_adjoint_matches_autograd(weighted, rows, width):
    torch.manual_seed(814)
    x = torch.randn(rows, width, requires_grad=True)
    weight = torch.randn(width, requires_grad=True) if weighted else None
    g = torch.randn_like(x)
    inverse = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    y = x * inverse
    if weighted:
        y = y * weight
    expected = torch.autograd.grad(y, (x, weight) if weighted else (x,), g)
    dx, dw = mhc._rms_adjoint(g, x.detach(), inverse.detach(), weight, weighted)
    torch.testing.assert_close(dx, expected[0], rtol=1e-5, atol=1e-6)
    if weighted:
        torch.testing.assert_close(dw, expected[1], rtol=1e-5, atol=1e-6)
    else:
        assert dw is None


def test_cuda_rms_adapter_exact_inputs_and_output_mask(monkeypatch):
    calls = []
    x, g, inv, w = torch.randn(2, 16), torch.randn(2, 16), torch.ones(2, 1), torch.ones(16)
    outputs = (torch.empty_like(x), torch.empty_like(w))

    def native(*args):
        calls.append(args)
        return outputs

    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    monkeypatch.setattr(torch.ops.aten, '_fused_rms_norm_backward', native)
    dx, dw = mhc._rms_adjoint(g, x, inv, w, True)
    assert dx is outputs[0] and dw is outputs[1]
    assert calls[0][0] is g and calls[0][1] is x
    assert calls[0][2] == [16] and calls[0][3] is inv and calls[0][4] is w
    assert calls[0][5] == [True, True]


def test_core_aggregate_adapters_preserve_stream_and_pre_layout(monkeypatch):
    from megatron.core.fusions import fused_mhc_kernels as core

    calls = []
    x = torch.randn(3, 2, 4, 16).bfloat16()
    pre = torch.randn(3, 2, 4)
    grad = torch.randn(3, 2, 16).bfloat16()

    def fwd(v, p):
        calls.append(('fwd', v.shape, p.shape))
        return (v.float() * p.unsqueeze(-1)).sum(-2).bfloat16()

    def bwd(g, v, p):
        calls.append(('bwd', g.shape, v.shape, p.shape))
        return (g.float().unsqueeze(-2) * p.unsqueeze(-1)).bfloat16(), (
            g.float().unsqueeze(-2) * v.float()
        ).sum(-1)

    monkeypatch.setattr(core, '_triton_h_aggregate_fwd', fwd)
    monkeypatch.setattr(core, '_cutile_h_aggregate_bwd', bwd, raising=False)
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    out = mhc._aggregate_recompute(x, pre)
    dx, dp = mhc._aggregate_adjoint(grad, x, pre)
    assert out.shape == grad.shape and dx.shape == x.shape and dp.shape == pre.shape
    assert torch.equal(out, (x.float() * pre.unsqueeze(-1)).sum(-2).bfloat16())
    assert calls[0] == ('fwd', (6, 1, 4, 16), (6, 1, 4))
    assert calls[1] == ('bwd', (6, 1, 16), (6, 1, 4, 16), (6, 1, 4))

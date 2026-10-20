# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU adjoint checks for the owned rotary and compressor VJPs; no deployment forward claim."""
from types import SimpleNamespace

import torch

from megatron.lite.model.deepseek_v41.vllm.attention import _Compress
from megatron.lite.model.deepseek_v41.vllm.mega_attention import ROPE_DIM, rope


def compress_vjp(raw, weight, positions, ratio, grad, eps):
    """``_Compress.backward`` on the group-boundary rows its forward saves."""
    rows = ((positions + 1) % ratio == 0).nonzero().flatten()
    ctx = SimpleNamespace(
        saved_tensors=(raw, weight, rows),
        options=(eps, ratio),
        needs_input_grad=(True, False, True, False, False, False),
    )
    draw, _, dweight, *_ = _Compress.backward(ctx, grad)
    return rows, draw, dweight


def test_rotary_vjp_is_transpose_for_query_and_shared_kv():
    torch.manual_seed(413)
    positions = torch.tensor([2, 0, 3])
    angles = torch.randn(4, ROPE_DIM // 2)
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    for shape in ((3, 2, ROPE_DIM + 16), (3, ROPE_DIM + 16)):
        x = torch.randn(shape, requires_grad=True)
        grad = torch.randn(shape)
        y = rope(x, cache, positions)
        (actual,) = torch.autograd.grad(y, x, grad)
        expected = rope(grad, cache, positions, inverse=True)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(y[..., :16], x[..., :16], rtol=0, atol=0)


def test_ratio_two_geometry_and_open_group_adjoint():
    positions = torch.arange(147)
    rows, *_ = compress_vjp(
        torch.zeros(147, 1024), torch.ones(512), positions, 2, torch.zeros(147, 512), 1e-6
    )
    assert rows.tolist() == list(range(1, 147, 2))
    # Two packed documents: groups restart with the in-document positions.
    rows, *_ = compress_vjp(
        torch.zeros(5, 1024),
        torch.ones(512),
        torch.tensor([0, 1, 2, 0, 1]),
        2,
        torch.zeros(5, 512),
        1e-6,
    )
    assert rows.tolist() == [1, 4]
    raw = torch.arange(5 * 1024, dtype=torch.float32).reshape(5, 1024) / 4096
    _, dx, dw = compress_vjp(raw, torch.ones(512), torch.arange(5), 2, torch.ones(5, 512), 1e-20)
    assert torch.isfinite(dx).all() and torch.isfinite(dw).all()
    assert dx[-1].eq(0).all() and dx[:-1].count_nonzero()
    assert dw.count_nonzero()

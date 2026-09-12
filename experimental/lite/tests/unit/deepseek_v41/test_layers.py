# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import block as hc


@pytest.mark.parametrize('copies', [2, 3, 4])
def test_mhc_shift_and_orientation(copies):
    class Mix(torch.nn.Module):
        def __init__(self, pre):
            super().__init__()
            self.pre = pre

        def forward(self, h):
            return (
                self.pre,
                torch.full_like(self.pre, 0.25),
                torch.eye(copies).expand(*h.shape[:2], -1, -1),
            )

    h = torch.arange(1, 1 + copies * 2).reshape(1, 1, copies, 2).float()
    p = torch.arange(1, copies + 1).reshape(1, 1, copies).float() / copies
    expected = h.clone()
    for layer in range(2):
        ap, fp = p.roll(1, -1) * (layer + 2), p.flip(-1) * (layer + 3)
        block = hc.DeepseekV41Block(2, copies, torch.nn.Identity(), torch.nn.Identity())
        block.attn_norm = block.ffn_norm = torch.nn.Identity()
        block.attn_mixes, block.ffn_mixes = Mix(ap), Mix(fp)
        seen = []
        for module in (block.attn, block.ffn):
            module.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
        x = (expected * p.unsqueeze(-1)).sum(-2)
        expected = expected + 0.25 * x.unsqueeze(-2)
        y = (expected * ap.unsqueeze(-1)).sum(-2)
        expected = expected + 0.25 * y.unsqueeze(-2)
        h, p = block(h, p)
        for actual, reference in zip((seen[0], seen[1], h, p), (x, y, expected, fp)):
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.manual_seed(9)
    residual = torch.randn(1, 2, copies, 3, requires_grad=True)
    out = torch.randn(1, 2, 3, requires_grad=True)
    post = torch.randn(1, 2, copies, requires_grad=True)
    comb = torch.randn(1, 2, copies, copies, requires_grad=True)
    expected = torch.stack(
        [
            post[..., j, None] * out
            + sum(comb[..., i, j, None] * residual[..., i, :] for i in range(copies))
            for j in range(copies)
        ],
        -2,
    )
    actual = hc.mix_residual(out, residual, post, comb)
    torch.testing.assert_close(actual, expected)
    args = (out, residual, post, comb)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), args),
        torch.autograd.grad(expected.square().sum(), args),
    ):
        torch.testing.assert_close(a, b)

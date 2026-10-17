# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Owned deployment VJPs must depend on gradient values, not producer strides.

PP1 hands a stage an einsum-transposed gradient where PP2 hands it a carrier
slice of equal value; both must reach the reference VJP in one layout.
"""
import torch
from megatron.lite.primitive.modules.deployment_math.autograd import visible_forward


class _Strides(torch.autograd.Function):
    """Identity that records the stride of the gradient it receives."""

    @staticmethod
    def forward(ctx, x, seen):
        ctx.seen = seen
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        ctx.seen.append(grad.stride())
        return grad, None


def _run(grad, value):
    seen_grads, seen_leaves = [], []

    def reference(x, w):
        if torch.is_grad_enabled():
            seen_leaves.append((x.stride(), w.stride()))
        y = _Strides.apply(x * w.sum(), seen_grads)
        return y, y.sum(-1)

    def visible(x, w):
        return x * w.sum(), (x * w.sum()).sum(-1)

    w = torch.randn(2, 3).t().requires_grad_()
    w.data.copy_(torch.arange(6.0).reshape(2, 3).t())
    x = value.requires_grad_()
    y, s = visible_forward(visible, reference, x, w)
    torch.autograd.backward((y, s), (grad, torch.ones_like(s)))
    return seen_grads, seen_leaves, x.grad, w.grad


def test_visible_vjp_canonicalizes_gradient_and_leaf_strides():
    torch.manual_seed(5)
    value = torch.randn(4, 3)
    grad = torch.randn(4, 3)
    transposed = grad.t().contiguous().t()
    sliced = torch.cat((grad, torch.zeros(4, 1)), -1)[:, :3]
    results = [
        _run(g, v.clone())
        for g in (grad, transposed, sliced)
        for v in (value, value.t().contiguous().t())
    ]
    for seen_grads, seen_leaves, *_ in results:
        assert seen_grads == [(3, 1)]
        assert seen_leaves == [((3, 1), (2, 1))]
    _, _, x_ref, w_ref = results[0]
    for _, _, x_grad, w_grad in results[1:]:
        assert torch.equal(x_grad, x_ref) and torch.equal(w_grad, w_ref)

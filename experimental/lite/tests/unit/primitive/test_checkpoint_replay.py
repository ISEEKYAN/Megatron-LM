# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch

from megatron.lite.primitive.parallel.source_state import KVSourceState, replayed


class _Rows(torch.autograd.Function):
    """A producer whose adjoint needs only its saved input, like the MoE hooks."""

    @staticmethod
    def forward(ctx, x, calls):
        ctx.save_for_backward(x)

        def produce():
            calls.append(torch.is_grad_enabled())
            return x.square() * 3

        return replayed(produce)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * 6 * x, None


def test_replay_reuses_first_forward_value(cpu_core_checkpoint):
    torch.manual_seed(7)
    state = KVSourceState()
    x = torch.randn(5, 3, requires_grad=True)
    w = torch.randn(3, requires_grad=True)
    calls = []

    def segment(value):
        first = _Rows.apply(value * w, calls)
        return _Rows.apply(first.sin(), calls) + first

    with state.microbatch():
        actual = state.checkpoint(segment, x, checkpoint_fn=cpu_core_checkpoint)
    actual.sum().backward()
    # Both producers ran once, in the no-grad first forward; the replay reused them.
    assert calls == [False, False]

    rx, rw = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    first = (rx * rw).square() * 3
    reference = first.sin().square() * 3 + first
    reference.sum().backward()
    assert torch.equal(actual, reference)
    assert torch.allclose(x.grad, rx.grad) and torch.allclose(w.grad, rw.grad)


def test_replayed_outside_checkpoint_produces():
    calls = []
    assert replayed(lambda: calls.append(1) or 4) == 4
    assert replayed(lambda: calls.append(1) or 5) == 5
    assert calls == [1, 1]

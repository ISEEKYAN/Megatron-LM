# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-visible autograd arithmetic and owned gradients."""

import torch
from torch.autograd.function import once_differentiable


def _canonical(x):
    """Same values in contiguous strides.

    The reference VJP runs eager reductions/GEMMs whose kernel choice follows
    operand strides. Upstream gradients of equal value arrive in different
    layouts (e.g. a PP carrier slice vs. an einsum-transposed PP1 gradient),
    so the owned gradient must not see producer strides. Strides of size-1
    dimensions are not a layout (torch treats such tensors as contiguous) and
    are left alone to avoid a full activation copy per backward call.
    """
    return None if x is None else x.contiguous()


class _Visible(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, reference, *inputs):
        ctx.reference = reference
        ctx.save_for_backward(*inputs)
        if inputs[0].numel() == 0:
            # A CP rank can own no rows of a document. Native kernels launched
            # with an empty grid fail or leave a CUDA error for the next checked
            # launch; the reference gives the same empty rows and the backward
            # already runs the reference.
            outputs = reference(*inputs)
            if isinstance(outputs, torch.Tensor):
                return outputs.detach().clone()
            return tuple(y.detach().clone() for y in outputs)
        return visible(*inputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        tensors = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast(
            device_type=tensors[0].device.type, enabled=False
        ):
            leaves = [
                _canonical(x.detach()).requires_grad_(needed)
                for x, needed in zip(tensors, ctx.needs_input_grad[2:])
            ]
            grad_outputs = [_canonical(g) for g in grad_outputs]
            outputs = ctx.reference(*leaves)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            active = [
                (y, g)
                for y, g in zip(outputs, grad_outputs)
                if g is not None and y.requires_grad
            ]
            wanted = [x for x in leaves if x.requires_grad]
            grads = (
                torch.autograd.grad(
                    [y for y, _ in active],
                    wanted,
                    [g for _, g in active],
                    allow_unused=True,
                )
                if active and wanted
                else [None] * len(wanted)
            )
        cursor = iter(grads)
        return (
            None,
            None,
            *(next(cursor) if x.requires_grad else None for x in leaves),
        )


def visible_forward(visible, reference, *inputs):
    return _Visible.apply(visible, reference, *inputs)

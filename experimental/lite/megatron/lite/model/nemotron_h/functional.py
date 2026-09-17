"""Inference-visible arithmetic with native training VJPs for Nemotron-H."""

import torch


class _VisibleForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, visible, native, *inputs):
        ctx.native = native
        ctx.save_for_backward(*inputs)
        return visible(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        with torch.enable_grad():
            inputs = tuple(
                x.detach().requires_grad_(required)
                for x, required in zip(
                    ctx.saved_tensors, ctx.needs_input_grad[2:], strict=True
                )
            )
            active = tuple(x for x in inputs if x.requires_grad)
            output = ctx.native(*inputs)
            gradients = iter(torch.autograd.grad(output, active, grad_outputs))
        return (
            None,
            None,
            *(next(gradients) if x.requires_grad else None for x in inputs),
        )


def visible_forward(visible, native, *inputs):
    """Keep inference rounding in forward; recompute the native VJP only in backward."""
    if not torch.is_grad_enabled() or not any(x.requires_grad for x in inputs):
        return visible(*inputs)
    return _VisibleForward.apply(visible, native, *inputs)


def linear(x, weight, bias=None):
    from vllm.model_executor.determinism.batch_invariant import linear_batch_invariant

    inputs = (x, weight) if bias is None else (x, weight, bias)
    return visible_forward(linear_batch_invariant, torch.nn.functional.linear, *inputs)


class GatedRMSNorm(torch.nn.Module):
    def __init__(self, width, group_size, eps, *, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.group_size, self.eps = group_size, eps

    def forward(self, x, gate):
        from vllm.model_executor.models.nemotron_h_alignment import gated_forward

        def visible(x, gate, weight):
            return gated_forward(x, gate, weight, self.group_size, self.eps)

        def native(x, gate, weight):
            y = x.float() * torch.nn.functional.silu(gate.float())
            groups = y.unflatten(-1, (-1, self.group_size))
            groups = groups * torch.rsqrt(
                groups.square().mean(-1, keepdim=True) + self.eps
            )
            return weight * groups.flatten(-2).to(x.dtype)

        return visible_forward(visible, native, x, gate, self.weight)


class RMSNorm(torch.nn.Module):
    """Inference rounding, including FP32 residual addition before normalization."""

    def __init__(self, width, eps, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x, residual=None):
        from vllm.model_executor.models.nemotron_h_alignment import rms_forward

        if residual is None:

            def visible(x, weight):
                return rms_forward(x, weight, self.eps)

            def native(x, weight):
                y = x.float()
                y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
                return weight * y.to(x.dtype)

            return visible_forward(visible, native, x, self.weight)

        def visible(x, residual, weight):
            return rms_forward(x, weight, self.eps, residual)

        def native(x, residual, weight):
            y = x.float() + residual.float()
            residual_out = y.to(weight.dtype)
            y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
            return y.to(weight.dtype) * weight, residual_out

        return visible_forward(visible, native, x, residual, self.weight)

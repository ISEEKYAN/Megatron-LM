# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-visible linear arithmetic and owned gradients."""

import torch
from megatron.lite.primitive.quantization.ste import clamp_with_inclusive_vjp

from .autograd import visible_forward


class _BF16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        compute = weight.bfloat16()
        ctx.save_for_backward(x, compute)
        ctx.input_dtype = x.dtype
        flat = x.reshape(-1, x.shape[-1]).bfloat16()
        from vllm.model_executor.kernels.linear.cute_dsl.batch_invariant import (
            bf16_fixed_mm,
        )

        output = bf16_fixed_mm(flat, compute)
        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, compute = ctx.saved_tensors
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dx = (grad.float() @ compute.float()).to(ctx.input_dtype)
            dw = (
                grad.reshape(-1, compute.shape[0]).float().T
                @ x.reshape(-1, compute.shape[1]).float()
            )
        return dx, dw


def bf16_fp32_linear(x, weight):
    return _BF16Linear.apply(x, weight)


def shared_swiglu(gate_up, limit):
    """SiLU rounds to input dtype before multiplying up; rounding uses an STE.

    Backward differentiates FP32 SiLU and clamp. The up VJP uses the rounded
    activation value; the gate VJP uses the smooth FP32 derivative. This is an
    explicitly owned surrogate, not finite differences of the discrete kernel.
    """

    def visible(y):
        out = torch.empty(
            (*y.shape[:-1], y.shape[-1] // 2), device=y.device, dtype=y.dtype
        )
        if limit > 0:
            torch.ops._C.silu_and_mul_with_clamp(out, y, limit, 1.0, 0.0)
        else:
            torch.ops._C.silu_and_mul(out, y)
        return out

    def reference(y):
        gate, up = y.float().chunk(2, -1)
        if limit > 0:
            gate = clamp_with_inclusive_vjp(gate, max=limit)
            up = clamp_with_inclusive_vjp(up, min=-limit, max=limit)
        activated = torch.nn.functional.silu(gate)
        rounded = activated + (activated.to(y.dtype).float() - activated).detach()
        return (rounded * up).to(y.dtype)

    return visible_forward(visible, reference, gate_up)


def decoded_bf16_master(weight):
    """BF16 deployment value, FP32 master gradient without a BF16 gradient cast."""
    return weight + (weight.bfloat16().float() - weight).detach()

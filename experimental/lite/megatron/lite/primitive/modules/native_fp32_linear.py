# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bias-free projections with a persistent FP32 master and native FP32 wgrad.

``native_fp32`` runs the forward GEMM in the activation dtype; ``block32_fp8``
runs :func:`dynamic_fp8_linear`. In both, the weight gradient is an FP32 GEMM of
FP32 operands returned directly to the FP32 leaf, never a BF16 product cast up.
Models opt in through :func:`linear_provider`; ``"default"`` returns
``torch.nn.Linear`` itself so default construction is unchanged.
"""

from __future__ import annotations

import functools

import torch
from torch.nn import functional as F

from megatron.lite.primitive.quantization.block32_fp8 import BLOCK, dynamic_fp8_linear

LINEAR_PROVIDERS = ("default", "native_fp32", "block32_fp8")


class _NativeFP32Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        compute_weight = weight.to(x.dtype)
        ctx.save_for_backward(x, compute_weight)
        return F.linear(x, compute_weight)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx = grad @ weight
        with torch.autocast(device_type=grad.device.type, enabled=False):
            dw = (
                grad.reshape(-1, weight.shape[0]).float().T
                @ x.reshape(-1, weight.shape[1]).float()
            )
        return dx, dw


def native_fp32_linear(x, weight):
    if weight.dtype != torch.float32:
        raise TypeError(
            f"native FP32 wgrad requires an FP32 master, got {weight.dtype}"
        )
    return _NativeFP32Linear.apply(x, weight)


class Linear(torch.nn.Module):
    """``nn.Linear``-compatible (``.weight``, no bias) FP32-master projection."""

    def __init__(self, in_features, out_features, *, mode, bias=False):
        super().__init__()
        if bias or mode not in LINEAR_PROVIDERS[1:]:
            raise ValueError(
                f"unsupported FP32-master Linear: mode={mode!r}, bias={bias}"
            )
        if mode == "block32_fp8" and (in_features % BLOCK or out_features % BLOCK):
            raise ValueError(
                f"block32_fp8 needs in/out features divisible by {BLOCK}, "
                f"got {in_features}x{out_features}"
            )
        self.in_features, self.out_features, self.mode = in_features, out_features, mode
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        if self.weight.dtype != torch.float32:
            raise TypeError("FP32-master Linear weight was cast; restore_fp32_masters")
        if self.mode == "block32_fp8":
            return dynamic_fp8_linear(x, self.weight)
        return native_fp32_linear(x, self.weight)


def linear_provider(name):
    """Return a ``(in_features, out_features, bias=False) -> module`` factory."""
    if name not in LINEAR_PROVIDERS:
        raise ValueError(
            f"unknown linear provider {name!r}; expected {LINEAR_PROVIDERS}"
        )
    return (
        torch.nn.Linear if name == "default" else functools.partial(Linear, mode=name)
    )


def restore_fp32_masters(model):
    """Re-materialize FP32 masters after a module-wide low-precision cast."""
    for module in model.modules():
        if isinstance(module, Linear):
            module.weight.data = module.weight.data.float()
    return model


__all__ = [
    "LINEAR_PROVIDERS",
    "Linear",
    "linear_provider",
    "native_fp32_linear",
    "restore_fp32_masters",
]

"""Training bridges for vLLM-visible dense and router linear kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ._contract import (
    check_parameter_versions,
    fp32_linear_vjp,
    own_visible_tensor,
    parameter_versions,
)


class _VLLMBlockFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        master_weight: torch.Tensor,
        visible_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        output = own_visible_tensor(visible_op(value, master_weight))
        ctx.save_for_backward(value)
        # Keep a direct reference so the bridge can issue its own actionable
        # version error before autograd's saved-tensor check fires.
        ctx.master_weight = master_weight
        ctx.versions = parameter_versions((master_weight,))
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        master_weight = ctx.master_weight
        check_parameter_versions((master_weight,), ctx.versions)
        (value,) = ctx.saved_tensors
        grad_value, grad_weight = fp32_linear_vjp(
            grad_output, value, master_weight
        )
        return grad_value, grad_weight, None


class _VLLMGateLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        master_weight: torch.Tensor,
        visible_op: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        output = visible_op(value)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = own_visible_tensor(output)
        ctx.save_for_backward(value)
        ctx.master_weight = master_weight
        ctx.versions = parameter_versions((master_weight,))
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        master_weight = ctx.master_weight
        check_parameter_versions((master_weight,), ctx.versions)
        (value,) = ctx.saved_tensors
        grad_value, grad_weight = fp32_linear_vjp(
            grad_output, value, master_weight
        )
        return grad_value, grad_weight, None


def block_fp8_linear(
    visible_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    master_weight: torch.Tensor,
) -> torch.Tensor:
    """Run the unchanged deployment GEMM and attach BF16-master VJP."""

    return visible_linear(visible_op, value, master_weight)


def visible_linear(
    visible_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    master_weight: torch.Tensor,
) -> torch.Tensor:
    """Run an exact deployment linear forward with the BF16-master VJP."""

    return _VLLMBlockFP8LinearFunction.apply(value, master_weight, visible_op)


def gate_linear(
    visible_op: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    master_weight: torch.Tensor,
) -> torch.Tensor:
    """Run vLLM GateLinear while differentiating its bound BF16 master."""

    return _VLLMGateLinearFunction.apply(value, master_weight, visible_op)


__all__ = [
    "_VLLMBlockFP8LinearFunction",
    "_VLLMGateLinearFunction",
    "block_fp8_linear",
    "gate_linear",
    "visible_linear",
]

from __future__ import annotations

import torch


def own_visible_tensor(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("a vLLM training bridge must own a tensor output")
    return value.clone() if value.is_inference() else value


def fp32_linear_vjp(
    grad_output: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = value.shape
    output_shape = grad_output.shape
    x2d = value.reshape(-1, value.shape[-1]).float()
    dy2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
    dx = torch.mm(dy2d, weight.float()).to(value.dtype).reshape(input_shape)
    dw = torch.mm(dy2d.T, x2d).to(weight.dtype)
    if output_shape[:-1] != input_shape[:-1]:
        raise RuntimeError("linear bridge received incompatible grad_output shape")
    return dx, dw

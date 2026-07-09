# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Standalone gradient norm and clipping helpers for M-FSDP shards."""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.distributed as dist


def local_grad_sq_sum(
    params: Iterable[torch.nn.Parameter],
    *,
    dtype: str | torch.dtype,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    """Accumulate the squared L2 norm of local Tensor or DTensor-like grads."""
    resolved_dtype = resolve_torch_dtype(dtype)
    total: torch.Tensor | None = None
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        local_grad = to_local_tensor(grad)
        if total is None:
            total = torch.zeros((), device=local_grad.device, dtype=resolved_dtype)
        total.add_(local_grad.detach().to(resolved_dtype).pow(2).sum())
    if total is None:
        total = torch.zeros(
            (), device=default_device or torch.device("cpu"), dtype=resolved_dtype
        )
    return total


def to_local_tensor(tensor: Any) -> torch.Tensor:
    """Return local storage for DTensor-like values and tensors unchanged."""
    local = getattr(tensor, "_local_tensor", None)
    if isinstance(local, torch.Tensor):
        return local
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Expected a Tensor or DTensor-like value, got {type(tensor)!r}."
        )
    return tensor


def resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        resolved = getattr(torch, dtype.removeprefix("torch."), None)
    if (
        not isinstance(resolved, torch.dtype)
        or not torch.empty((), dtype=resolved).is_floating_point()
    ):
        raise ValueError(
            f"Gradient norm accumulation dtype must be floating point: {dtype!r}."
        )
    return resolved


def all_reduce_scalar_(
    value: torch.Tensor, *, op: dist.ReduceOp, group: dist.ProcessGroup
) -> None:
    """All-reduce a scalar on a device accepted by the process-group backend."""
    reduced = value
    try:
        backend = str(dist.get_backend(group)).lower()
    except (RuntimeError, TypeError, ValueError):
        backend = ""
    if "nccl" in backend and value.device.type != "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL scalar all-reduce requires a CUDA tensor.")
        reduced = value.to(torch.device("cuda", torch.cuda.current_device()))
    elif "gloo" in backend and value.device.type != "cpu":
        reduced = value.cpu()
    dist.all_reduce(reduced, op=op, group=group)
    if reduced is not value:
        value.copy_(reduced.to(device=value.device, dtype=value.dtype))

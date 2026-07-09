# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Standalone gradient norm and clipping helpers for M-FSDP shards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class GradNormBreakdown:
    dense_sq: float
    dense_tp_sharded_sq: float
    dense_tp_replicated_sq: float
    expert_sq: float
    global_dense_sq: float
    global_dense_tp_sharded_sq: float
    global_dense_tp_replicated_sq: float
    global_expert_sq: float
    total_sq: float
    grad_norm: float
    dense_params: int
    dense_tp_replicated_params: int
    expert_params: int
    dense_names: tuple[str, ...] = ()
    dense_tp_replicated_names: tuple[str, ...] = ()
    expert_names: tuple[str, ...] = ()


class CanonicalGradNormMegatronFSDPOptimizer:
    """Legacy delegating wrapper retained for callers outside the native path."""

    name = "megatron_fsdp"

    def __init__(self, optimizer: Any, ps: Any, **_kwargs: Any) -> None:
        self._inner_optimizer = optimizer
        self.ps = ps

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner_optimizer, name)


def compute_mfsdp_grad_norm(
    optimizer: Any,
    ps: Any,
    *,
    param_is_expert: dict[int, bool] | None = None,
    param_names: dict[int, str] | None = None,
) -> GradNormBreakdown:
    """Compute the global L2 norm from this rank's optimizer shards."""
    del param_names
    dense: list[torch.nn.Parameter] = []
    expert: list[torch.nn.Parameter] = []
    for param in _unique_parameters(optimizer):
        (
            expert if bool((param_is_expert or {}).get(id(param), False)) else dense
        ).append(param)

    dense_sq = local_grad_sq_sum(dense, dtype=torch.float32)
    expert_sq = local_grad_sq_sum(
        expert, dtype=torch.float32, default_device=dense_sq.device
    )
    _sum_if_distributed(
        dense_sq, getattr(ps, "dp_cp_group", None) or getattr(ps, "dp_group", None)
    )
    _sum_if_distributed(expert_sq, getattr(ps, "ep_dp_group", None))
    total_sq = dense_sq + expert_sq.to(dense_sq.device)
    _sum_if_distributed(total_sq, getattr(ps, "pp_group", None))
    dense_value = float(dense_sq.item())
    expert_value = float(expert_sq.item())
    total_value = float(total_sq.item())
    return GradNormBreakdown(
        dense_sq=dense_value,
        dense_tp_sharded_sq=0.0,
        dense_tp_replicated_sq=0.0,
        expert_sq=expert_value,
        global_dense_sq=dense_value,
        global_dense_tp_sharded_sq=0.0,
        global_dense_tp_replicated_sq=0.0,
        global_expert_sq=expert_value,
        total_sq=total_value,
        grad_norm=math.sqrt(total_value),
        dense_params=len(dense),
        dense_tp_replicated_params=0,
        expert_params=len(expert),
    )


@torch.no_grad()
def _clip_mfsdp_grads_by_total_norm(
    optimizer: Any,
    grad_norm: float,
    **_kwargs: Any,
) -> None:
    """Scale unique normal or decoupled grads using each leaf's clip limit."""
    if not math.isfinite(float(grad_norm)):
        return
    seen: set[int] = set()
    for leaf in _leaf_optimizers(optimizer):
        if bool(getattr(leaf, "is_stub_optimizer", False)):
            continue
        config = getattr(leaf, "config", None)
        max_norm = float(getattr(config, "clip_grad", 0.0))
        if max_norm <= 0:
            continue
        coefficient = min(1.0, max_norm / (float(grad_norm) + 1.0e-6))
        use_decoupled = bool(
            getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
        for group in getattr(leaf, "param_groups", ()):
            for param in group.get("params", ()):
                if id(param) in seen:
                    continue
                seen.add(id(param))
                grad = (
                    getattr(param, "decoupled_grad", None)
                    if use_decoupled
                    else param.grad
                )
                if grad is not None:
                    grad.mul_(coefficient)


def _leaf_optimizers(optimizer: Any) -> list[Any]:
    leaves = getattr(optimizer, "chained_optimizers", None)
    return list(leaves) if leaves is not None else [optimizer]


def _unique_parameters(optimizer: Any) -> Iterable[torch.nn.Parameter]:
    seen: set[int] = set()
    for leaf in _leaf_optimizers(optimizer):
        groups = getattr(leaf, "param_groups", None)
        if groups is None:
            groups = getattr(getattr(leaf, "optimizer", None), "param_groups", ())
        for group in groups:
            for param in group.get("params", ()):
                if id(param) not in seen:
                    seen.add(id(param))
                    yield param


def _sum_if_distributed(value: torch.Tensor, group: dist.ProcessGroup | None) -> None:
    if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return
    all_reduce_scalar_(value, op=dist.ReduceOp.SUM, group=group)


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


__all__ = [
    "all_reduce_scalar_",
    "CanonicalGradNormMegatronFSDPOptimizer",
    "GradNormBreakdown",
    "compute_mfsdp_grad_norm",
    "local_grad_sq_sum",
    "resolve_torch_dtype",
    "to_local_tensor",
]

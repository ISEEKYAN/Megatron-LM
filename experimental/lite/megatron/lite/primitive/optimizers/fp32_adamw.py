# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Backend-neutral FP32 AdamW update kernel for local parameter shards."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import torch
import torch.nn as nn

LocalTensorFn = Callable[[nn.Parameter], torch.Tensor]
LocalGradFn = Callable[[nn.Parameter], torch.Tensor | None]
CopyMasterSliceFn = Callable[[nn.Parameter, torch.Tensor, int, int], None]


class GradTransferPolicy(Protocol):
    """Scratch-transfer contract used by the sliced update pipeline."""

    depth: int

    def copy_grad_slice(self, grad: torch.Tensor, start: int, length: int) -> Any: ...

    def wait_grad_slice(self, ticket: Any) -> torch.Tensor: ...

    def release_grad_slice(self, ticket: Any) -> None: ...

    def drain(self) -> None: ...


class SynchronousGradTransfer:
    """Default transfer policy for gradients already on the update device."""

    depth = 1

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = None if device is None else torch.device(device)

    def copy_grad_slice(
        self, grad: torch.Tensor, start: int, length: int
    ) -> torch.Tensor:
        value = grad.detach().reshape(-1).narrow(0, start, length)
        return value.to(device=self.device or value.device, dtype=torch.float32)

    def wait_grad_slice(self, ticket: torch.Tensor) -> torch.Tensor:
        return ticket

    def release_grad_slice(self, ticket: torch.Tensor) -> None:
        del ticket

    def drain(self) -> None:
        pass


def normalize_param_groups(
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
    *,
    default_weight_decay: float,
) -> list[dict[str, Any]]:
    """Normalize torch-style parameters without introducing backend knowledge."""
    items = list(params)
    if not items:
        return []
    if all(isinstance(item, dict) for item in items):
        groups: list[dict[str, Any]] = []
        for item in items:
            group = dict(item)
            group_params = list(group.get("params", ()))
            if not group_params:
                continue
            group["params"] = group_params
            group.setdefault("weight_decay", default_weight_decay)
            groups.append(group)
        return groups
    return [{"params": items, "weight_decay": default_weight_decay}]


class FP32AdamW:
    """Scalar AdamW over ordinary local tensors with optional bounded slicing.

    Sharding, collectives, model residency, and checkpoint envelopes remain the
    responsibility of backend adapters. This class owns only FP32 master/moment
    state and the elementwise update.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        local_param: LocalTensorFn | None = None,
        local_grad: LocalGradFn | None = None,
        copy_master_slice_to_param: CopyMasterSliceFn | None = None,
        master_device: torch.device | str | None = None,
        pin_master: bool = False,
        transfer_policy: GradTransferPolicy | None = None,
        slice_capacity: int | None = None,
    ) -> None:
        if slice_capacity is not None and slice_capacity <= 0:
            raise ValueError("slice_capacity must be a positive element count or None.")
        self.param_groups = normalize_param_groups(
            params, default_weight_decay=weight_decay
        )
        self.params: list[nn.Parameter] = []
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.step_count = 0
        self.slice_capacity = slice_capacity
        self._local_param = local_param or (lambda param: param.detach())
        self._local_grad = local_grad or (lambda param: param.grad)
        self._copy_master_slice = (
            copy_master_slice_to_param or self._default_copy_master_slice
        )
        self._master_device = (
            None if master_device is None else torch.device(master_device)
        )
        if pin_master and self._master_device != torch.device("cpu"):
            raise ValueError("pin_master requires master_device='cpu'.")
        self._pin_master = bool(pin_master)
        self._transfer_policy = transfer_policy or SynchronousGradTransfer(
            self._master_device
        )
        if int(self._transfer_policy.depth) <= 0:
            raise ValueError("gradient transfer depth must be positive.")

        self.state: dict[nn.Parameter, dict[str, Any]] = {}
        for group in self.param_groups:
            group.setdefault("lr", self.lr)
            group.setdefault("wd_mult", 1.0)
            group["weight_decay"] = float(group.get("weight_decay", self.weight_decay))
            for param in group["params"]:
                self.params.append(param)
                local = self._local_param(param).detach()
                if self._pin_master:
                    master = torch.empty(
                        local.shape, dtype=torch.float32, device="cpu", pin_memory=True
                    )
                    master.copy_(local)
                elif self._master_device is None and local.dtype is torch.float32:
                    master = local
                else:
                    master = local.to(
                        device=self._master_device or local.device, dtype=torch.float32
                    ).clone()
                self.state[param] = {
                    "master_param": master,
                    "exp_avg": torch.zeros_like(master, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(master, dtype=torch.float32),
                    "step": 0,
                }

    def zero_grad(self, *args, **kwargs) -> None:
        set_to_none = bool(args[0]) if args else bool(kwargs.get("set_to_none", False))
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas
        for group in self.param_groups:
            group_lr = float(group.get("lr", self.lr))
            group_weight_decay = float(group.get("weight_decay", self.weight_decay))
            for param in group["params"]:
                grad = self._local_grad(param)
                if grad is None or grad.numel() == 0:
                    continue
                state = self.state[param]
                state["step"] = int(state["step"]) + 1
                param_step = int(state["step"])
                step_size = group_lr / (1.0 - beta1**param_step)
                bias_correction2_sqrt = (1.0 - beta2**param_step) ** 0.5
                self._step_one_param(
                    param,
                    grad,
                    state,
                    group_lr=group_lr,
                    group_weight_decay=group_weight_decay,
                    beta1=beta1,
                    beta2=beta2,
                    step_size=step_size,
                    bias_correction2_sqrt=bias_correction2_sqrt,
                )
        self._transfer_policy.drain()

    def _step_one_param(
        self,
        param: nn.Parameter,
        grad: torch.Tensor,
        state: dict[str, Any],
        *,
        group_lr: float,
        group_weight_decay: float,
        beta1: float,
        beta2: float,
        step_size: float,
        bias_correction2_sqrt: float,
    ) -> None:
        master = state["master_param"].reshape(-1)
        exp_avg = state["exp_avg"].reshape(-1)
        exp_avg_sq = state["exp_avg_sq"].reshape(-1)
        capacity = self.slice_capacity or master.numel()
        slices = iter(
            (start, min(capacity, master.numel() - start))
            for start in range(0, master.numel(), capacity)
        )
        pending: deque[tuple[int, int, Any]] = deque()
        for _ in range(min(int(self._transfer_policy.depth), master.numel())):
            try:
                start, length = next(slices)
            except StopIteration:
                break
            pending.append(
                (
                    start,
                    length,
                    self._transfer_policy.copy_grad_slice(grad, start, length),
                )
            )

        while pending:
            start, length, ticket = pending.popleft()
            grad_slice = self._transfer_policy.wait_grad_slice(ticket)
            master_slice = master.narrow(0, start, length)
            exp_avg_slice = exp_avg.narrow(0, start, length)
            exp_avg_sq_slice = exp_avg_sq.narrow(0, start, length)
            if group_weight_decay != 0.0:
                master_slice.mul_(1.0 - group_lr * group_weight_decay)
            exp_avg_slice.mul_(beta1).add_(grad_slice, alpha=1.0 - beta1)
            exp_avg_sq_slice.mul_(beta2).addcmul_(
                grad_slice, grad_slice, value=1.0 - beta2
            )
            denom = exp_avg_sq_slice.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            master_slice.addcdiv_(exp_avg_slice, denom, value=-step_size)
            self._copy_master_slice(param, state["master_param"], start, length)
            self._transfer_policy.release_grad_slice(ticket)
            try:
                next_start, next_length = next(slices)
            except StopIteration:
                continue
            pending.append(
                (
                    next_start,
                    next_length,
                    self._transfer_policy.copy_grad_slice(
                        grad, next_start, next_length
                    ),
                )
            )

    def _default_copy_master_slice(
        self, param: nn.Parameter, master: torch.Tensor, start: int, length: int
    ) -> None:
        target = self._local_param(param).detach().reshape(-1).narrow(0, start, length)
        source = master.reshape(-1).narrow(0, start, length)
        target.copy_(source.to(device=target.device, dtype=target.dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "fp32_adamw",
            "step_count": self.step_count,
            "master_params": [
                self.state[param]["master_param"] for param in self.params
            ],
            "exp_avgs": [self.state[param]["exp_avg"] for param in self.params],
            "exp_avg_sqs": [self.state[param]["exp_avg_sq"] for param in self.params],
            "steps": [int(self.state[param]["step"]) for param in self.params],
            "lrs": [
                float(group.get("lr", self.lr))
                for group in self.param_groups
                for _param in group["params"]
            ],
            "weight_decays": [
                float(group.get("weight_decay", self.weight_decay))
                for group in self.param_groups
                for _param in group["params"]
            ],
            "betas": self.betas,
            "eps": self.eps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("type") != "fp32_adamw":
            raise ValueError("Invalid FP32 AdamW state_dict.")
        self.step_count = int(state_dict.get("step_count", 0))
        for source_name, target_name in (
            ("master_params", "master_param"),
            ("exp_avgs", "exp_avg"),
            ("exp_avg_sqs", "exp_avg_sq"),
        ):
            loaded = state_dict.get(source_name)
            if not isinstance(loaded, list) or len(loaded) != len(self.params):
                raise ValueError(f"Invalid FP32 AdamW {source_name} state.")
            for param, source in zip(self.params, loaded, strict=True):
                target = self.state[param][target_name]
                local_source = getattr(source, "_local_tensor", source)
                if (
                    not isinstance(local_source, torch.Tensor)
                    or local_source.shape != target.shape
                ):
                    raise ValueError(f"Invalid FP32 AdamW {source_name} tensor.")
                target.copy_(local_source.to(device=target.device, dtype=target.dtype))
        steps = state_dict.get("steps")
        if steps is not None:
            if not isinstance(steps, list) or len(steps) != len(self.params):
                raise ValueError("Invalid FP32 AdamW steps state.")
            for param, step in zip(self.params, steps, strict=True):
                self.state[param]["step"] = int(step)
        else:
            for param in self.params:
                self.state[param]["step"] = self.step_count
        loaded_lrs = state_dict.get("lrs")
        if loaded_lrs is not None:
            if not isinstance(loaded_lrs, list) or len(loaded_lrs) != len(self.params):
                raise ValueError("Invalid FP32 AdamW lr state.")
            index = 0
            for group in self.param_groups:
                group["lr"] = float(loaded_lrs[index])
                index += len(group["params"])
        weight_decays = state_dict.get("weight_decays")
        if weight_decays is not None:
            if not isinstance(weight_decays, list) or len(weight_decays) != len(
                self.params
            ):
                raise ValueError("Invalid FP32 AdamW weight_decay state.")
            index = 0
            for group in self.param_groups:
                group["weight_decay"] = float(weight_decays[index])
                index += len(group["params"])
        loaded_betas = state_dict.get("betas")
        if loaded_betas is not None:
            self.betas = (float(loaded_betas[0]), float(loaded_betas[1]))
        if "eps" in state_dict:
            self.eps = float(state_dict["eps"])
        for param in self.params:
            master = self.state[param]["master_param"]
            self._copy_master_slice(param, master, 0, master.numel())


__all__ = [
    "FP32AdamW",
    "GradTransferPolicy",
    "SynchronousGradTransfer",
    "normalize_param_groups",
]

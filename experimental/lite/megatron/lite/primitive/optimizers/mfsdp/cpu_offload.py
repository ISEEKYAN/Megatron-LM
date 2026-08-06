# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU Adam momentum offload for M-FSDP shard parameters."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class _CpuOptimizerCollection:
    """Compatibility facade for one CPU optimizer per offloaded parameter."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    @property
    def state(self) -> dict[torch.Tensor, dict[str, Any]]:
        return {
            param: value
            for optimizer in self.optimizers
            for param, value in optimizer.state.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_states = state_dict.get("optimizers")
        if optimizer_states is not None:
            if len(optimizer_states) != len(self.optimizers):
                raise ValueError(
                    "M-FSDP CPU optimizer checkpoint parameter count does not "
                    f"match: expected {len(self.optimizers)}, got {len(optimizer_states)}."
                )
            for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
                optimizer.load_state_dict(optimizer_state)
            return

        # Backward compatibility with checkpoints from the former aggregate
        # AdamW. Torch state ids are ordered by the flattened param_groups.
        group_by_param_id = {
            param_id: group
            for group in state_dict["param_groups"]
            for param_id in group["params"]
        }
        for param_id, optimizer in enumerate(self.optimizers):
            group = dict(group_by_param_id[param_id])
            group["params"] = [0]
            local_state = state_dict["state"].get(param_id)
            optimizer.load_state_dict(
                {
                    "state": {} if local_state is None else {0: local_state},
                    "param_groups": [group],
                }
            )


class CpuAdamGroup:
    """Adam momentum state on CPU for a subset of M-FSDP shard parameters.

    Memory savings: the authoritative fp32 master plus exp_avg and exp_avg_sq
    live on CPU. ``shard_param.data`` is the GPU compute/all-gather mirror and
    is refreshed after each CPU update.

    Per-step flow overlaps later D2H copies and earlier H2D copies with each
    per-parameter CPU AdamW update using dedicated streams and D2H events.
    """

    def __init__(
        self,
        gpu_param_groups: list[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        use_pinned = torch.cuda.is_available()
        self._use_pinned = use_pinned
        self._gpu_params: list[nn.Parameter] = []
        self._cpu_params: list[torch.Tensor] = []
        self._cpu_grad_bufs: list[torch.Tensor | None] = []
        cpu_groups: list[dict[str, Any]] = []

        for group in gpu_param_groups:
            cpu_group_params: list[torch.Tensor] = []
            for gpu_param in group["params"]:
                flat = gpu_param.detach().cpu().float().contiguous()
                if use_pinned:
                    pinned = torch.empty_like(flat, pin_memory=True)
                    pinned.copy_(flat)
                    flat = pinned
                # Wrap as leaf tensor with grad support so AdamW can update it.
                cpu_p = flat.detach().requires_grad_(True)
                self._gpu_params.append(gpu_param)
                self._cpu_params.append(cpu_p)
                self._cpu_grad_bufs.append(None)
                cpu_group_params.append(cpu_p)

            cpu_group = {k: v for k, v in group.items() if k != "params"}
            cpu_group["params"] = cpu_group_params
            cpu_groups.append(cpu_group)

        cpu_optimizers: list[torch.optim.Optimizer] = []
        for group in cpu_groups:
            group_defaults = {k: v for k, v in group.items() if k != "params"}
            for cpu_param in group["params"]:
                cpu_optimizers.append(
                    torch.optim.AdamW(
                        [{**group_defaults, "params": [cpu_param]}],
                        lr=lr,
                        betas=betas,
                        eps=eps,
                        # A one-parameter optimizer has no foreach opportunity.
                        foreach=False,
                    )
                )
        self._cpu_optimizer = _CpuOptimizerCollection(cpu_optimizers)
        if use_pinned:
            self._d2h_stream = torch.cuda.Stream()
            self._h2d_stream = torch.cuda.Stream()

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._cpu_optimizer.param_groups

    def step(self) -> None:
        d2h_events: list[torch.cuda.Event | None] = []
        current_stream = torch.cuda.current_stream() if self._use_pinned else None
        if self._use_pinned:
            self._d2h_stream.wait_stream(current_stream)
            self._h2d_stream.wait_stream(current_stream)

        for i, (gpu_param, cpu_param) in enumerate(
            zip(self._gpu_params, self._cpu_params)
        ):
            grad = getattr(gpu_param, "main_grad", gpu_param.grad)
            if grad is None:
                cpu_param.grad = None
                d2h_events.append(None)
                continue
            flat_gpu_grad = grad.detach().view(-1)
            buf = self._cpu_grad_bufs[i]
            if buf is None or buf.shape != flat_gpu_grad.shape:
                buf = torch.zeros(
                    flat_gpu_grad.shape, dtype=torch.float32, device="cpu"
                )
                if self._use_pinned:
                    buf = buf.pin_memory()
                self._cpu_grad_bufs[i] = buf
            if self._use_pinned:
                with torch.cuda.stream(self._d2h_stream):
                    buf.copy_(flat_gpu_grad, non_blocking=True)
                    d2h_events.append(self._d2h_stream.record_event())
            else:
                buf.copy_(flat_gpu_grad, non_blocking=True)
                d2h_events.append(None)
            cpu_param.grad = buf

        for gpu_param, cpu_param, cpu_optimizer, d2h_event in zip(
            self._gpu_params,
            self._cpu_params,
            self._cpu_optimizer.optimizers,
            d2h_events,
        ):
            if d2h_event is not None:
                d2h_event.synchronize()
            cpu_optimizer.step()
            if self._use_pinned:
                with torch.cuda.stream(self._h2d_stream):
                    gpu_param.data.view(-1).copy_(cpu_param.data, non_blocking=True)
            else:
                gpu_param.data.view(-1).copy_(cpu_param.data, non_blocking=True)
            cpu_param.grad = None

        if self._use_pinned:
            self._h2d_stream.record_event().wait(current_stream)

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self._cpu_optimizer.state_dict(),
            "master_params": [param.detach().clone() for param in self._cpu_params],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_state = state_dict.get("optimizer", state_dict)
        self._cpu_optimizer.load_state_dict(optimizer_state)
        master_params = state_dict.get("master_params")
        if master_params is not None and len(master_params) != len(self._cpu_params):
            raise ValueError(
                "M-FSDP CPU optimizer checkpoint master parameter count does not "
                f"match: expected {len(self._cpu_params)}, got {len(master_params)}."
            )
        with torch.no_grad():
            for index, (gpu_param, cpu_param) in enumerate(
                zip(self._gpu_params, self._cpu_params)
            ):
                if master_params is None:
                    cpu_param.copy_(gpu_param.detach().view(-1).cpu())
                else:
                    saved = master_params[index]
                    if saved.shape != cpu_param.shape:
                        raise ValueError(
                            "M-FSDP CPU optimizer checkpoint master parameter shape "
                            f"does not match at index {index}: expected "
                            f"{tuple(cpu_param.shape)}, got {tuple(saved.shape)}."
                        )
                    cpu_param.copy_(saved)
                gpu_param.view(-1).copy_(cpu_param.to(dtype=gpu_param.dtype))

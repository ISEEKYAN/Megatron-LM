# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU Adam momentum offload for M-FSDP shard parameters."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class CpuAdamGroup:
    """Adam momentum state on CPU for a subset of M-FSDP shard parameters.

    Memory savings: the authoritative fp32 master plus exp_avg and exp_avg_sq
    live on CPU. ``shard_param.data`` is the GPU compute/all-gather mirror and
    is refreshed after each CPU update.

    Per-step flow:
      1. D2H — enqueue GPU gradient → pinned CPU grad buffer, then fence once
      2. CPU AdamW step updates cpu_param
      3. H2D — enqueue updated cpu_param → GPU shard_param.data
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

        self._cpu_optimizer = torch.optim.AdamW(
            cpu_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            # Keep scalar CPU AdamW: the foreach path regresses the isolated
            # optimizer step for the representative M-FSDP MoE workload.
            foreach=False,
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._cpu_optimizer.param_groups

    def step(self) -> None:
        for i, (gpu_param, cpu_param) in enumerate(
            zip(self._gpu_params, self._cpu_params)
        ):
            if gpu_param.grad is None:
                cpu_param.grad = None
                continue

            flat_gpu_grad = gpu_param.grad.detach().view(-1)

            buf = self._cpu_grad_bufs[i]
            if buf is None or buf.shape != flat_gpu_grad.shape:
                buf = torch.zeros(
                    flat_gpu_grad.shape, dtype=torch.float32, device="cpu"
                )
                if self._use_pinned:
                    buf = buf.pin_memory()
                self._cpu_grad_bufs[i] = buf

            buf.copy_(flat_gpu_grad, non_blocking=True)
            cpu_param.grad = buf

        if self._use_pinned:
            # CPU AdamW must not read pinned gradients until every queued D2H
            # copy has completed. One fence avoids synchronizing per tensor.
            torch.cuda.current_stream().synchronize()
        self._cpu_optimizer.step()

        for gpu_param, cpu_param in zip(self._gpu_params, self._cpu_params):
            gpu_param.data.view(-1).copy_(cpu_param.data, non_blocking=True)
            cpu_param.grad = None

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

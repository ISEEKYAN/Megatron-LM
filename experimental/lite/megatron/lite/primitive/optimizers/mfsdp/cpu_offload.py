# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU Adam momentum offload for M-FSDP shard parameters."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class CpuAdamGroup:
    """Adam momentum state on CPU for a subset of M-FSDP shard parameters.

    Memory savings: exp_avg + exp_avg_sq moved off GPU (2 × 4B × numel).
    Hard constraint: shard_param.data (fp32 master) stays on GPU for all-gather.

    Per-step flow (synchronous):
      1. D2H — copy GPU gradient → pinned CPU grad buffer  (blocking)
      2. CPU AdamW step updates cpu_param
      3. H2D — copy updated cpu_param → GPU shard_param.data  (blocking)
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
                    flat = flat.pin_memory()
                # Wrap as leaf tensor with grad support so AdamW can update it.
                cpu_p = flat.clone().detach().requires_grad_(True)
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
            foreach=False,
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._cpu_optimizer.param_groups

    @property
    def gpu_param_id_set(self) -> set[int]:
        return {id(p) for p in self._gpu_params}

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

            buf.copy_(flat_gpu_grad)
            cpu_param.grad = buf

        self._cpu_optimizer.step()

        for gpu_param, cpu_param in zip(self._gpu_params, self._cpu_params):
            gpu_param.data.view(-1).copy_(cpu_param.data)
            cpu_param.grad = None

    def state_dict(self) -> dict[str, Any]:
        return self._cpu_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cpu_optimizer.load_state_dict(state_dict)
        for gpu_param, cpu_param in zip(self._gpu_params, self._cpu_params):
            cpu_param.data.copy_(gpu_param.data.view(-1).cpu())

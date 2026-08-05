# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU Adam momentum offload for M-FSDP shard parameters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TransferRun:
    """Adjacent shard views transferred as one storage span."""

    indices: tuple[int, ...]
    cpu_master: torch.Tensor | None = None
    cpu_grad: torch.Tensor | None = None


def _are_adjacent_views(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.numel() == 0 or right.numel() == 0:
        return False
    return (
        left.device == right.device
        and left.dtype == right.dtype
        and left.layout == torch.strided
        and right.layout == torch.strided
        and left.is_contiguous()
        and right.is_contiguous()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and right.storage_offset() == left.storage_offset() + left.numel()
    )


def _contiguous_runs(tensors: list[torch.Tensor]) -> list[_TransferRun]:
    runs: list[_TransferRun] = []
    for index, tensor in enumerate(tensors):
        if runs and _are_adjacent_views(tensors[runs[-1].indices[-1]], tensor):
            runs[-1].indices += (index,)
        else:
            runs.append(_TransferRun(indices=(index,)))
    return runs


def _storage_span(
    tensors: list[torch.Tensor], indices: tuple[int, ...]
) -> torch.Tensor:
    first = tensors[indices[0]].detach()
    total_numel = sum(tensors[index].numel() for index in indices)
    return torch.as_strided(
        first, (total_numel,), (1,), storage_offset=first.storage_offset()
    )


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

    Adjacent shard views sharing flat M-FSDP storage transfer as one run. The
    run-level D2H and H2D copies overlap with CPU AdamW updates on other runs;
    discontiguous or partially missing gradients retain the scalar fallback.
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
        group_indices: list[list[int]] = []
        cpu_groups: list[dict[str, Any]] = []

        for group in gpu_param_groups:
            indices: list[int] = []
            for gpu_param in group["params"]:
                indices.append(len(self._gpu_params))
                self._gpu_params.append(gpu_param)
            group_indices.append(indices)

        self._transfer_runs = _contiguous_runs(self._gpu_params)
        cpu_params: list[torch.Tensor | None] = [None] * len(self._gpu_params)
        for run in self._transfer_runs:
            total_numel = sum(self._gpu_params[i].numel() for i in run.indices)
            run.cpu_master = torch.empty(
                total_numel, dtype=torch.float32, device="cpu", pin_memory=use_pinned
            )
            run.cpu_master.copy_(_storage_span(self._gpu_params, run.indices))
            offset = 0
            for index in run.indices:
                numel = self._gpu_params[index].numel()
                # AdamW sees leaf views while transfers use the owning flat buffer.
                cpu_params[index] = (
                    run.cpu_master.narrow(0, offset, numel)
                    .detach()
                    .requires_grad_(True)
                )
                offset += numel

        self._cpu_params = [param for param in cpu_params if param is not None]
        if len(self._cpu_params) != len(self._gpu_params):
            raise RuntimeError("M-FSDP CPU master construction lost a parameter view.")
        self._cpu_grad_bufs: list[torch.Tensor | None] = [None] * len(self._gpu_params)

        for group, indices in zip(gpu_param_groups, group_indices):
            cpu_group = {k: v for k, v in group.items() if k != "params"}
            cpu_group["params"] = [self._cpu_params[index] for index in indices]
            cpu_groups.append(cpu_group)

        logger.info(
            "M-FSDP CPU offload packs %d parameter shards into %d contiguous "
            "transfer runs.",
            len(self._gpu_params),
            len(self._transfer_runs),
        )

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

        for run in self._transfer_runs:
            if run.cpu_grad is None:
                assert run.cpu_master is not None
                run.cpu_grad = torch.zeros_like(
                    run.cpu_master, pin_memory=self._use_pinned
                )
                offset = 0
                for index in run.indices:
                    numel = self._gpu_params[index].numel()
                    self._cpu_grad_bufs[index] = run.cpu_grad.narrow(0, offset, numel)
                    offset += numel

            active_indices = [
                index
                for index in run.indices
                if self._gpu_params[index].grad is not None
            ]
            active_index_set = set(active_indices)
            for index in run.indices:
                self._cpu_params[index].grad = (
                    self._cpu_grad_bufs[index] if index in active_index_set else None
                )

            def copy_grads() -> None:
                assert run.cpu_grad is not None
                grads = [
                    self._gpu_params[index].grad.detach().view(-1)
                    for index in active_indices
                ]
                if len(active_indices) == len(run.indices) and all(
                    _are_adjacent_views(left, right)
                    for left, right in zip(grads, grads[1:])
                ):
                    run.cpu_grad.copy_(
                        _storage_span(grads, tuple(range(len(grads)))),
                        non_blocking=True,
                    )
                    return
                for index, grad in zip(active_indices, grads):
                    buf = self._cpu_grad_bufs[index]
                    assert buf is not None
                    buf.copy_(grad, non_blocking=True)

            if self._use_pinned:
                with torch.cuda.stream(self._d2h_stream):
                    copy_grads()
                    d2h_events.append(self._d2h_stream.record_event())
            else:
                copy_grads()
                d2h_events.append(None)

        for run, d2h_event in zip(self._transfer_runs, d2h_events):
            if d2h_event is not None:
                d2h_event.synchronize()
            for index in run.indices:
                self._cpu_optimizer.optimizers[index].step()
            assert run.cpu_master is not None
            gpu_span = _storage_span(self._gpu_params, run.indices)
            if self._use_pinned:
                with torch.cuda.stream(self._h2d_stream):
                    gpu_span.copy_(run.cpu_master, non_blocking=True)
            else:
                gpu_span.copy_(run.cpu_master, non_blocking=True)
            for index in run.indices:
                self._cpu_params[index].grad = None

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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded CPU AdamW adapter for M-FSDP shard parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from megatron.lite.primitive.optimizers.fp32_adamw import FP32AdamW, STATE_DICT_TYPE


@dataclass(slots=True)
class _RingTicket:
    slot: int
    length: int
    event: torch.cuda.Event | None


class _PinnedGradRing:
    """Two reusable gradient slots plus the M-FSDP transfer streams."""

    depth = 2

    def __init__(self, *, capacity: int, total_numel: int, use_cuda: bool) -> None:
        if capacity <= 0:
            raise ValueError("M-FSDP CPU AdamW bucket_size must be positive.")
        self.capacity = int(capacity)
        self.total_numel = int(total_numel)
        self.use_cuda = bool(use_cuda)
        self._slots: list[torch.Tensor | None] = [None, None]
        self._in_use = [False, False]
        self._next_slot = 0
        self._d2h_stream: torch.cuda.Stream | None = None
        self._h2d_stream: torch.cuda.Stream | None = None
        self._last_h2d_event: torch.cuda.Event | None = None
        self.high_water_elements = 0
        self.d2h_bytes = 0
        self.h2d_bytes = 0

    @property
    def allocated_elements(self) -> int:
        return sum(slot.numel() for slot in self._slots if slot is not None)

    @property
    def live_leases(self) -> int:
        return sum(self._in_use)

    def _ensure_runtime(self) -> None:
        if not self.use_cuda or self._d2h_stream is not None:
            return
        self._d2h_stream = torch.cuda.Stream()
        self._h2d_stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        self._d2h_stream.wait_stream(current)
        self._h2d_stream.wait_stream(current)

    def _ensure_slot(self, index: int, length: int) -> torch.Tensor:
        if length > self.capacity:
            raise RuntimeError(
                f"gradient slice length {length} exceeds ring capacity {self.capacity}."
            )
        slot = self._slots[index]
        if slot is None or slot.numel() < length:
            old_numel = 0 if slot is None else slot.numel()
            proposed = self.allocated_elements - old_numel + length
            hard_bound = min(self.total_numel, self.depth * self.capacity)
            if proposed > hard_bound:
                raise RuntimeError(
                    "M-FSDP pinned gradient ring exceeded its hard bound: "
                    f"requested {proposed} elements, bound {hard_bound}."
                )
            slot = torch.empty(
                length, dtype=torch.float32, device="cpu", pin_memory=self.use_cuda
            )
            self._slots[index] = slot
            self.high_water_elements = max(self.high_water_elements, proposed)
        return slot

    def copy_grad_slice(
        self, grad: torch.Tensor, start: int, length: int
    ) -> _RingTicket:
        self._ensure_runtime()
        slot_index = self._next_slot
        self._next_slot = (self._next_slot + 1) % self.depth
        if self._in_use[slot_index]:
            raise RuntimeError("M-FSDP gradient ring slot was reused before release.")
        slot = self._ensure_slot(slot_index, length)
        source = grad.detach().reshape(-1).narrow(0, start, length)
        target = slot.narrow(0, 0, length)
        event = None
        if self.use_cuda:
            assert self._d2h_stream is not None
            # The persistent transfer stream must depend on this step's
            # reduce-scatter result, not only on the stream state that existed
            # when the ring was first constructed.
            self._d2h_stream.wait_stream(torch.cuda.current_stream(source.device))
            with torch.cuda.stream(self._d2h_stream):
                target.copy_(source, non_blocking=True)
                event = self._d2h_stream.record_event()
        else:
            target.copy_(source)
        self._in_use[slot_index] = True
        self.d2h_bytes += length * torch.float32.itemsize
        return _RingTicket(slot=slot_index, length=length, event=event)

    def wait_grad_slice(self, ticket: _RingTicket) -> torch.Tensor:
        if ticket.event is not None:
            ticket.event.synchronize()
        slot = self._slots[ticket.slot]
        if slot is None:
            raise RuntimeError("M-FSDP gradient ring ticket references no slot.")
        return slot.narrow(0, 0, ticket.length)

    def release_grad_slice(self, ticket: _RingTicket) -> None:
        if not self._in_use[ticket.slot]:
            raise RuntimeError("M-FSDP gradient ring ticket was released twice.")
        self._in_use[ticket.slot] = False

    def copy_master_slice_to_param(
        self, param: nn.Parameter, master: torch.Tensor, start: int, length: int
    ) -> None:
        target = param.detach().reshape(-1).narrow(0, start, length)
        source = master.reshape(-1).narrow(0, start, length)
        if self.use_cuda:
            self._ensure_runtime()
            assert self._h2d_stream is not None
            with torch.cuda.stream(self._h2d_stream):
                target.copy_(source, non_blocking=True)
                self._last_h2d_event = self._h2d_stream.record_event()
        else:
            target.copy_(source.to(dtype=target.dtype))
        self.h2d_bytes += length * target.element_size()

    def drain(self) -> None:
        if self._last_h2d_event is not None:
            self._last_h2d_event.synchronize()
            self._last_h2d_event = None
        if self.live_leases:
            raise RuntimeError(
                f"M-FSDP optimizer returned with {self.live_leases} live transfer leases."
            )

    def release_runtime(self) -> None:
        self.drain()
        self._slots = [None, None]
        self._d2h_stream = None
        self._h2d_stream = None


class CpuAdamGroup:
    """M-FSDP adapter around the shared FP32 AdamW local-shard kernel."""

    _FORMAT_VERSION = 2

    def __init__(
        self,
        gpu_param_groups: list[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        bucket_size: int,
    ) -> None:
        self._gpu_params = [
            param for group in gpu_param_groups for param in group["params"]
        ]
        total_numel = sum(param.numel() for param in self._gpu_params)
        use_cuda = bool(
            torch.cuda.is_available()
            and any(param.device.type == "cuda" for param in self._gpu_params)
        )
        self._ring = _PinnedGradRing(
            capacity=int(bucket_size), total_numel=total_numel, use_cuda=use_cuda
        )
        default_weight_decay = float(
            gpu_param_groups[0].get("weight_decay", 0.0) if gpu_param_groups else 0.0
        )
        self._optimizer = FP32AdamW(
            gpu_param_groups,
            lr=lr,
            weight_decay=default_weight_decay,
            betas=betas,
            eps=eps,
            local_param=lambda param: param.detach(),
            local_grad=_mfsdp_optimizer_grad,
            copy_master_slice_to_param=self._ring.copy_master_slice_to_param,
            master_device="cpu",
            pin_master=use_cuda,
            transfer_policy=self._ring,
            slice_capacity=int(bucket_size),
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._optimizer.param_groups

    @property
    def gpu_params(self) -> tuple[nn.Parameter, ...]:
        """Stable original shard parameters used for membership checks."""
        return tuple(self._gpu_params)

    @property
    def _cpu_params(self) -> list[torch.Tensor]:
        return [
            self._optimizer.state[param]["master_param"]
            for param in self._optimizer.params
        ]

    @property
    def ring_depth(self) -> int:
        return self._ring.depth

    @property
    def ring_allocated_elements(self) -> int:
        return self._ring.allocated_elements

    @property
    def ring_high_water_elements(self) -> int:
        return self._ring.high_water_elements

    @property
    def live_transfer_leases(self) -> int:
        return self._ring.live_leases

    @property
    def d2h_bytes(self) -> int:
        return self._ring.d2h_bytes

    @property
    def h2d_bytes(self) -> int:
        return self._ring.h2d_bytes

    def step(self) -> None:
        self._optimizer.step()

    def release_transfer_state(self) -> None:
        self._ring.release_runtime()

    def state_dict(self) -> dict[str, Any]:
        self._ring.drain()
        return {
            "format_version": self._FORMAT_VERSION,
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._ring.drain()
        optimizer_state = state_dict.get("optimizer", state_dict)
        if optimizer_state.get("type") == STATE_DICT_TYPE:
            self._optimizer.load_state_dict(optimizer_state)
        elif optimizer_state.get("type") is not None:
            raise ValueError(
                "Invalid M-FSDP CPU optimizer state_dict type: "
                f"{optimizer_state.get('type')!r}."
            )
        else:
            self._optimizer.load_state_dict(self._convert_legacy_state(state_dict))
        self._ring.drain()

    def _convert_legacy_state(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        optimizer_state = state_dict.get("optimizer", state_dict)
        master_params = state_dict.get("master_params")
        if master_params is not None and len(master_params) != len(self._gpu_params):
            raise ValueError(
                "M-FSDP CPU optimizer checkpoint master parameter count does not "
                f"match: expected {len(self._gpu_params)}, got {len(master_params)}."
            )
        current = self._optimizer.state_dict()
        exp_avgs = [torch.zeros_like(value) for value in current["exp_avgs"]]
        exp_avg_sqs = [torch.zeros_like(value) for value in current["exp_avg_sqs"]]
        steps = [0 for _ in self._gpu_params]
        lrs = list(current["lrs"])
        weight_decays = list(current["weight_decays"])
        loaded_betas = self._optimizer.betas
        loaded_eps = self._optimizer.eps

        per_param = optimizer_state.get("optimizers")
        if per_param is not None:
            if len(per_param) != len(self._gpu_params):
                raise ValueError(
                    "M-FSDP CPU optimizer checkpoint parameter count does not "
                    f"match: expected {len(self._gpu_params)}, got {len(per_param)}."
                )
            entries = []
            for local in per_param:
                groups = local.get("param_groups", [])
                local_state = next(iter(local.get("state", {}).values()), None)
                entries.append((groups[0] if groups else {}, local_state))
        else:
            group_by_param_id = {
                param_id: group
                for group in optimizer_state.get("param_groups", [])
                for param_id in group.get("params", [])
            }
            raw_state = optimizer_state.get("state", {})
            entries = [
                (group_by_param_id.get(index, {}), raw_state.get(index))
                for index in range(len(self._gpu_params))
            ]

        for index, (group, local_state) in enumerate(entries):
            if group:
                lrs[index] = float(group.get("lr", lrs[index]))
                weight_decays[index] = float(
                    group.get("weight_decay", weight_decays[index])
                )
                group_betas = tuple(group.get("betas", loaded_betas))
                group_eps = float(group.get("eps", loaded_eps))
                if tuple(float(value) for value in group_betas) != tuple(loaded_betas):
                    if index != 0:
                        raise ValueError(
                            "M-FSDP legacy CPU AdamW checkpoint has per-parameter betas."
                        )
                    loaded_betas = tuple(float(value) for value in group_betas)
                if group_eps != loaded_eps:
                    if index != 0:
                        raise ValueError(
                            "M-FSDP legacy CPU AdamW checkpoint has per-parameter eps."
                        )
                    loaded_eps = group_eps
            if not local_state:
                continue
            exp_avgs[index].copy_(local_state["exp_avg"])
            exp_avg_sqs[index].copy_(local_state["exp_avg_sq"])
            raw_step = local_state.get("step", 0)
            steps[index] = int(
                raw_step.item() if isinstance(raw_step, torch.Tensor) else raw_step
            )

        masters = (
            [
                param.detach().to(device="cpu", dtype=torch.float32).clone()
                for param in self._gpu_params
            ]
            if master_params is None
            else list(master_params)
        )
        return {
            "type": STATE_DICT_TYPE,
            "step_count": max(steps, default=0),
            "master_params": masters,
            "exp_avgs": exp_avgs,
            "exp_avg_sqs": exp_avg_sqs,
            "steps": steps,
            "lrs": lrs,
            "weight_decays": weight_decays,
            "betas": loaded_betas,
            "eps": loaded_eps,
        }


def _mfsdp_optimizer_grad(param: nn.Parameter) -> torch.Tensor | None:
    decoupled = getattr(param, "decoupled_grad", None)
    if decoupled is not None:
        return decoupled
    main_grad = getattr(param, "main_grad", None)
    return main_grad if main_grad is not None else param.grad


__all__ = ["CpuAdamGroup"]

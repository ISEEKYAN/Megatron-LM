# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Communication-buffer allocators with optional NCCL user-buffer registration.

The hot path always has a Torch allocator.  Apex's NCCL allocator is discovered
lazily and only decorates allocations; an unavailable or incompatible optional
backend emits one warning and falls back without changing optimizer semantics.
"""

from __future__ import annotations

import importlib
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.primitive.optimizers.mfsdp.config import MFSDPConfig

logger = logging.getLogger(__name__)


class NCCLUserBuffer:
    """Best-effort Apex NCCL memory-pool adapter."""

    def __init__(
        self,
        *,
        enabled: bool,
        groups: tuple[dist.ProcessGroup, ...],
        symmetric: bool,
    ) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.groups = groups
        self.symmetric = symmetric
        self.module: Any | None = None
        self.pool: Any | None = None
        self._warned = False
        if self.enabled:
            self._initialize()

    @property
    def active(self) -> bool:
        return self.module is not None and self.pool is not None

    def _initialize(self) -> None:
        try:
            module = importlib.import_module("apex.contrib.nccl_allocator")
            module.init()
            try:
                pool = module.create_nccl_mem_pool(symmetric=self.symmetric)
            except TypeError:
                pool = module.create_nccl_mem_pool()
            self.module = module
            self.pool = pool
        except (ImportError, AttributeError, RuntimeError, TypeError) as error:
            self._disable(f"NCCL user-buffer allocation unavailable: {error}")

    def allocation_context(self, group: dist.ProcessGroup | None):
        if not self.active or group is None:
            return nullcontext()
        try:
            return self.module.nccl_mem(self.pool, group=group)
        except (AttributeError, RuntimeError, TypeError) as error:
            self._disable(f"NCCL user-buffer context failed: {error}")
            return nullcontext()

    def register_additional_groups(self) -> None:
        if not self.active or len(self.groups) < 2:
            return
        for group in self.groups[1:]:
            try:
                backend = group._get_backend(
                    torch.device("cuda", torch.cuda.current_device())
                )
                backend.register_mem_pool(self.pool)
            except (AttributeError, RuntimeError, TypeError) as error:
                self._disable(f"NCCL user-buffer group registration failed: {error}")
                return

    def _disable(self, reason: str) -> None:
        self.module = None
        self.pool = None
        if not self._warned:
            logger.warning("%s; using Torch communication buffers.", reason)
            self._warned = True


@dataclass(slots=True)
class BufferLease:
    tensor: torch.Tensor
    owner: "TemporaryBufferAllocator"
    key: tuple[Any, ...]
    slot: int | None = None
    registered: bool = False
    _released: bool = False

    def release(self) -> None:
        if not self._released:
            self.owner.release(self)
            self._released = True


class TemporaryBufferAllocator:
    """Allocate communication storage and release it at the pipeline boundary."""

    def __init__(self, user_buffer: NCCLUserBuffer | None = None) -> None:
        self.user_buffer = user_buffer

    def allocate(
        self,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup | None,
        key: tuple[Any, ...],
    ) -> BufferLease:
        context = (
            self.user_buffer.allocation_context(group)
            if self.user_buffer is not None
            else nullcontext()
        )
        registered = bool(self.user_buffer is not None and self.user_buffer.active)
        try:
            with context:
                tensor = torch.empty(numel, dtype=dtype, device=device)
            if registered:
                self.user_buffer.register_additional_groups()
        except (RuntimeError, TypeError) as error:
            if self.user_buffer is not None:
                self.user_buffer._disable(f"Registered allocation failed: {error}")
            tensor = torch.empty(numel, dtype=dtype, device=device)
            registered = False
        return BufferLease(tensor=tensor, owner=self, key=key, registered=registered)

    def release(self, lease: BufferLease) -> None:
        # Dynamic storage is reclaimed when the lease drops its final reference.
        return None


class DoubleBufferAllocator(TemporaryBufferAllocator):
    """Two persistent communication slots per dtype/device/group key.

    This mirrors the bounded double-buffer property of MCore without bringing
    in its HSDP, FP8, and CUDA-graph allocation branches.
    """

    def __init__(self, user_buffer: NCCLUserBuffer | None = None) -> None:
        super().__init__(user_buffer)
        self._slots: dict[tuple[Any, ...], list[torch.Tensor | None]] = {}
        self._busy: dict[tuple[Any, ...], set[int]] = {}

    def allocate(
        self,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup | None,
        key: tuple[Any, ...],
    ) -> BufferLease:
        pool_key = (*key, dtype, device)
        slots = self._slots.setdefault(pool_key, [None, None])
        busy = self._busy.setdefault(pool_key, set())
        for slot in range(2):
            if slot in busy:
                continue
            tensor = slots[slot]
            if tensor is None or tensor.numel() < numel:
                lease = super().allocate(
                    numel,
                    dtype=dtype,
                    device=device,
                    group=group,
                    key=pool_key,
                )
                tensor = lease.tensor
                slots[slot] = tensor
                registered = lease.registered
            else:
                registered = bool(
                    self.user_buffer is not None and self.user_buffer.active
                )
            busy.add(slot)
            return BufferLease(
                tensor=tensor.narrow(0, 0, numel),
                owner=self,
                key=pool_key,
                slot=slot,
                registered=registered,
            )
        # More than two simultaneous buckets is legal; it simply loses the
        # persistent/registered allocation optimization for this one request.
        return super().allocate(
            numel,
            dtype=dtype,
            device=device,
            group=group,
            key=pool_key,
        )

    def release(self, lease: BufferLease) -> None:
        if lease.slot is not None:
            self._busy.get(lease.key, set()).discard(lease.slot)


def build_temporary_allocator(
    config: MFSDPConfig,
    groups: tuple[dist.ProcessGroup, ...],
) -> TemporaryBufferAllocator:
    user_buffer = NCCLUserBuffer(
        enabled=config.nccl_ub,
        groups=groups,
        symmetric=not config.disable_symmetric_registration,
    )
    allocator_type = (
        DoubleBufferAllocator if config.fsdp_double_buffer else TemporaryBufferAllocator
    )
    return allocator_type(user_buffer)


__all__ = [
    "BufferLease",
    "DoubleBufferAllocator",
    "NCCLUserBuffer",
    "TemporaryBufferAllocator",
    "build_temporary_allocator",
]

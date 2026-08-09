# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bucketed parameter and gradient storage for standalone M-FSDP.

The layout follows the useful core of MCore's ParamAndGradBuffer: parameters
are concatenated once, the global flat buffer is padded once per bucket, and
each rank owns one contiguous shard.  An all-gather therefore reconstructs the
global flat layout directly; compute parameters are zero-copy views into the
collective output and optimizer parameters/gradients are zero-copy views into
persistent local main buffers.
"""

from __future__ import annotations

import importlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.lite.primitive.optimizers.mfsdp.config import (
    MFSDPConfig, MFSDPProcessGroups, MixedPrecisionPolicy, group_rank,
    group_size)


class NCCLUserBuffer:
    """Apex NCCL memory-pool adapter required when ``nccl_ub`` is enabled."""

    def __init__(
        self, *, enabled: bool, groups: tuple[dist.ProcessGroup, ...], symmetric: bool
    ) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.groups = groups
        self.symmetric = symmetric
        self.module: Any | None = None
        self.pool: Any | None = None
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
            self._fail(f"NCCL user-buffer allocation unavailable: {error}")

    def allocation_context(self, group: dist.ProcessGroup | None):
        if not self.active or group is None:
            return nullcontext()
        try:
            return self.module.nccl_mem(self.pool, group=group)
        except (AttributeError, RuntimeError, TypeError) as error:
            self._fail(f"NCCL user-buffer context failed: {error}")

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
                self._fail(f"NCCL user-buffer group registration failed: {error}")

    def _fail(self, reason: str) -> None:
        self.module = None
        self.pool = None
        raise RuntimeError(reason)


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
                self.user_buffer._fail(f"Registered allocation failed: {error}")
            tensor = torch.empty(numel, dtype=dtype, device=device)
            registered = False
        return BufferLease(tensor=tensor, owner=self, key=key, registered=registered)

    def release(self, lease: BufferLease) -> None:
        _free_storage(lease.tensor)

    def release_cached(self, *, force: bool = False) -> None:
        """Drop allocator-owned communication storage before a device move."""


class DoubleBufferAllocator(TemporaryBufferAllocator):
    """Two persistent communication slots per dtype/device/group key."""

    def __init__(self, user_buffer: NCCLUserBuffer | None = None) -> None:
        super().__init__(user_buffer)
        self._slots: dict[tuple[Any, ...], list[torch.Tensor | None]] = {}
        self._busy: dict[tuple[Any, ...], set[int]] = {}
        self._reuse_events: dict[tuple[Any, ...], list[Any | None]] = {}

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
        reuse_events = self._reuse_events.setdefault(pool_key, [None, None])
        for slot in range(2):
            if slot in busy:
                continue
            tensor = slots[slot]
            if tensor is None or tensor.numel() < numel:
                lease = super().allocate(
                    numel, dtype=dtype, device=device, group=group, key=pool_key
                )
                tensor = lease.tensor
                slots[slot] = tensor
                registered = lease.registered
            else:
                registered = bool(
                    self.user_buffer is not None and self.user_buffer.active
                )
            _wait_for_reuse_event(reuse_events[slot], tensor)
            reuse_events[slot] = None
            busy.add(slot)
            return BufferLease(
                tensor=tensor.narrow(0, 0, numel),
                owner=self,
                key=pool_key,
                slot=slot,
                registered=registered,
            )
        raise RuntimeError(
            "M-FSDP double-buffer capacity exhausted; drain a completed "
            "gradient reduce before acquiring another slot."
        )

    def release(self, lease: BufferLease) -> None:
        if lease.slot is not None:
            events = self._reuse_events.setdefault(lease.key, [None, None])
            events[lease.slot] = _record_reuse_event(lease.tensor)
            self._busy.get(lease.key, set()).discard(lease.slot)

    def release_cached(self, *, force: bool = False) -> None:
        # The busy check catches genuine misuse — releasing the cache while a
        # collective is legitimately in flight — on the normal path. On an
        # exception-driven teardown (``force=True``) a slot may be left busy by
        # an aborted collective; raising here would replace the primary error
        # (e.g. the OOM that triggered the teardown) with a misleading
        # "active buffers" RuntimeError, so force-drop the cache instead.
        if not force and any(self._busy.values()):
            raise RuntimeError("Cannot release active M-FSDP communication buffers.")
        self._slots.clear()
        self._busy.clear()
        self._reuse_events.clear()


def _record_reuse_event(tensor: torch.Tensor) -> Any | None:
    if tensor.device.type != "cuda":
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(tensor.device))
    return event


def _wait_for_reuse_event(event: Any | None, tensor: torch.Tensor) -> None:
    if event is not None:
        torch.cuda.current_stream(tensor.device).wait_event(event)


def _free_storage(tensor: torch.Tensor) -> None:
    """Physically release completed temporary communication storage.

    Replacing a tensor view does not return its allocation while the lease still
    owns that view.  MCore releases temporary bucket storage after completion;
    mirror that lifecycle for non-registered, non-double-buffer allocations.
    """
    if tensor.numel() == 0 or tensor.storage_offset() != 0:
        return
    storage = tensor.untyped_storage()
    if storage.nbytes():
        storage.resize_(0)


def build_temporary_allocator(
    config: MFSDPConfig, groups: tuple[dist.ProcessGroup, ...]
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


@dataclass(frozen=True, slots=True)
class ParamBinding:
    module: nn.Module
    attribute: str


@dataclass(slots=True)
class ParamSpec:
    name: str
    full_param: nn.Parameter
    bindings: list[ParamBinding]
    shape: torch.Size
    numel: int
    full_offset: int = 0
    shard_numel: int = 0
    local_offset: int = 0
    param_offset: int = 0
    shard_param: nn.Parameter | None = None
    retain_full_storage_through_backward: bool = False


@dataclass(frozen=True, slots=True)
class SavedParamView:
    bucket: "ParamBucket"
    size: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


class ParamBucket:
    """One communication bucket with persistent optimizer shard storage."""

    def __init__(
        self,
        bucket_id: int,
        specs: list[ParamSpec],
        *,
        process_group: dist.ProcessGroup | None,
        gather_group: dist.ProcessGroup | None,
        config: MFSDPConfig,
        allocator: TemporaryBufferAllocator,
        allocator_layout_key: tuple[Any, ...],
    ) -> None:
        if not specs:
            raise ValueError("An M-FSDP parameter bucket cannot be empty.")
        self.bucket_id = int(bucket_id)
        self.specs = specs
        self.process_group = process_group
        self.gather_group = gather_group or process_group
        self.world_size = group_size(process_group)
        self.rank = group_rank(process_group)
        self.config = config
        self.allocator = allocator
        self.allocator_layout_key = allocator_layout_key
        self.retain_full_storage_through_backward = all(
            spec.retain_full_storage_through_backward for spec in specs
        )
        self.device = specs[0].full_param.device
        compute_dtype = specs[0].full_param.dtype
        main_params_dtype = (
            compute_dtype if config.full_optimizer_offload else config.main_params_dtype
        )
        self.policy = MixedPrecisionPolicy(
            compute_dtype=compute_dtype,
            main_params_dtype=main_params_dtype,
            main_grads_dtype=config.main_grads_dtype,
            grad_comm_dtype=config.grad_comm_dtype,
        )

        unpadded_numel = sum(spec.numel for spec in specs)
        self.local_numel = math.ceil(unpadded_numel / self.world_size)
        self.full_numel = self.local_numel * self.world_size
        offset = 0
        local_begin = self.rank * self.local_numel
        local_end = local_begin + self.local_numel
        for spec in specs:
            spec.full_offset = offset
            spec_begin = offset
            spec_end = offset + spec.numel
            intersection_begin = max(spec_begin, local_begin)
            intersection_end = min(spec_end, local_end)
            spec.shard_numel = max(0, intersection_end - intersection_begin)
            spec.local_offset = min(
                self.local_numel, max(0, intersection_begin - local_begin)
            )
            spec.param_offset = min(spec.numel, max(0, intersection_begin - spec_begin))
            offset = spec_end

        self.main_param_buffer = torch.zeros(
            self.local_numel, dtype=self.policy.main_params_dtype, device=self.device
        )
        self.main_grad_buffer = torch.zeros(
            self.local_numel, dtype=self.policy.main_grads_dtype, device=self.device
        )
        self.grad_shard_buffer = self.main_grad_buffer
        self.local_compute_buffer = (
            self.main_param_buffer
            if self.policy.main_params_dtype == self.policy.compute_dtype
            else torch.empty(0, dtype=self.policy.compute_dtype, device=self.device)
        )
        self.local_grad_comm_buffer = (
            self.main_grad_buffer
            if self.policy.grad_comm_dtype == self.policy.main_grads_dtype
            else torch.empty(0, dtype=self.policy.grad_comm_dtype, device=self.device)
        )
        self.full_buffer = torch.empty(0, dtype=compute_dtype, device=self.device)
        self.full_main_grad_buffer = torch.empty(
            0, dtype=self.policy.main_grads_dtype, device=self.device
        )
        self._full_lease: BufferLease | None = None
        self._full_main_grad_lease: BufferLease | None = None
        self._grad_lease: BufferLease | None = None
        self._local_compute_lease: BufferLease | None = None
        self._local_grad_comm_lease: BufferLease | None = None
        self._param_gather_work: Any | None = None
        self._param_gather_event: Any | None = None
        self._grad_reduce_work: Any | None = None
        self._grad_reduce_event: Any | None = None
        self._grad_reduce_launched = False
        self._grad_reduce_finished = False
        self._microbatch_reduced = False
        self._has_accumulated_grad = False
        self._full_ready = True
        self.grad_sync_enabled = False
        self.grad_ready_callback: Callable[["ParamBucket"], None] | None = None
        self.before_main_grad_allocate: Callable[["ParamBucket"], None] | None = None
        self._grad_ready_ids: set[int] = set()
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        with torch.no_grad():
            for spec in self.specs:
                if spec.shard_numel:
                    source = (
                        spec.full_param.detach()
                        .reshape(-1)
                        .narrow(0, spec.param_offset, spec.shard_numel)
                    )
                    self.main_param_buffer.narrow(
                        0, spec.local_offset, spec.shard_numel
                    ).copy_(source)
                shard_view = self.main_param_buffer.narrow(
                    0, spec.local_offset, spec.shard_numel
                )
                shard_param = nn.Parameter(
                    shard_view, requires_grad=spec.full_param.requires_grad
                )
                _copy_parameter_metadata(spec.full_param, shard_param)
                shard_param._mfsdp_original_ndim = spec.full_param.ndim
                spec.shard_param = shard_param
                # TE returns a dummy ``.grad`` when its wgrad GEMM writes the
                # real gradient directly into ``main_grad``.
                spec.full_param.grad_added_to_main_grad = False
                spec.full_param.__fsdp_param__ = True
                spec.full_param.overwrite_main_grad = True
                spec.full_param.get_main_grad = self._make_main_grad_getter(spec)
                spec.full_param.register_post_accumulate_grad_hook(
                    self._make_grad_ready_hook(spec)
                )

    def install_sharded_parameters(self) -> None:
        for spec in self.specs:
            assert spec.shard_param is not None
            for binding in spec.bindings:
                setattr(binding.module, binding.attribute, spec.shard_param)

    def install_full_parameters(self) -> None:
        self.wait_param_gather()
        for spec in self.specs:
            view = self.full_buffer.narrow(0, spec.full_offset, spec.numel).view(
                spec.shape
            )
            spec.full_param.data = view
            # TE checks these MCore FSDP markers before using the lazy FP32
            # wgrad destination, so restore them on every materialization.
            spec.full_param.grad_added_to_main_grad = False
            spec.full_param.__fsdp_param__ = True
            spec.full_param.overwrite_main_grad = True
            for binding in spec.bindings:
                setattr(binding.module, binding.attribute, spec.full_param)

    def prepare_main_grads(self) -> None:
        """Attach bounded FP32 views for fused wgrad accumulation."""
        if self._full_main_grad_lease is None:
            self._enforce_main_grad_slot_limit()
            # Unlike full parameters, gradient staging has no autograd-saved
            # views after its reduce-scatter completes. Pool it across bucket
            # layouts so released per-bucket slots do not become resident.
            self._full_main_grad_lease = self.allocator.allocate(
                self.full_numel,
                dtype=self.policy.main_grads_dtype,
                device=self.device,
                group=self.process_group,
                key=("main_grad", id(self.process_group)),
            )
            self.full_main_grad_buffer = self._full_main_grad_lease.tensor
            self.full_main_grad_buffer.zero_()
        for spec in self.specs:
            self.get_main_grad(spec)

    def get_main_grad(self, spec: ParamSpec) -> torch.Tensor:
        """Lazily allocate this bucket's full-precision gradient staging view."""
        # A prior microbatch may still be reducing this bucket's staging
        # buffer.  MCore carries that work across the next forward and only
        # resolves storage pressure when main_grad is requested again.
        self._enforce_main_grad_slot_limit()
        if self._full_main_grad_lease is None:
            self._full_main_grad_lease = self.allocator.allocate(
                self.full_numel,
                dtype=self.policy.main_grads_dtype,
                device=self.device,
                group=self.process_group,
                key=("main_grad", id(self.process_group)),
            )
            self.full_main_grad_buffer = self._full_main_grad_lease.tensor
            self.full_main_grad_buffer.zero_()
        main_grad = self.full_main_grad_buffer.narrow(
            0, spec.full_offset, spec.numel
        ).view(spec.shape)
        spec.full_param.main_grad = main_grad
        return main_grad

    def _enforce_main_grad_slot_limit(self) -> None:
        """Retire an in-flight reduction before consuming a third grad slot."""
        if self.before_main_grad_allocate is not None:
            self.before_main_grad_allocate(self)

    def _make_main_grad_getter(self, spec: ParamSpec) -> Callable[[], torch.Tensor]:
        return lambda: self.get_main_grad(spec)

    def _release_full_main_grads(self) -> None:
        if self._full_main_grad_lease is not None:
            self._full_main_grad_lease.release()
            self._full_main_grad_lease = None
        self.full_main_grad_buffer = torch.empty(
            0, dtype=self.policy.main_grads_dtype, device=self.device
        )
        for spec in self.specs:
            spec.full_param.main_grad = self.full_main_grad_buffer

    def prepare_param_gather(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._full_lease is None:
            self._full_lease = self.allocator.allocate(
                self.full_numel,
                dtype=self.policy.compute_dtype,
                device=self.device,
                group=self.gather_group,
                key=("param", id(self.gather_group), self.allocator_layout_key),
            )
            self.full_buffer = self._full_lease.tensor
        if self.policy.main_params_dtype != self.policy.compute_dtype:
            if self._local_compute_lease is None:
                self._local_compute_lease = self.allocator.allocate(
                    self.local_numel,
                    dtype=self.policy.compute_dtype,
                    device=self.device,
                    group=self.gather_group,
                    key=(
                        "param-local",
                        id(self.gather_group),
                        self.allocator_layout_key,
                    ),
                )
                self.local_compute_buffer = self._local_compute_lease.tensor
            self.local_compute_buffer.copy_(self.main_param_buffer)
        return self.full_buffer, self.local_compute_buffer

    def mark_param_gather_launched(
        self, work: Any | None, completion_event: Any | None = None
    ) -> None:
        self._param_gather_work = work
        self._param_gather_event = completion_event

    def wait_param_gather(self) -> None:
        if self._full_ready:
            return
        if self._param_gather_work is None and self._param_gather_event is None:
            output, local = self.prepare_param_gather()
            if self.world_size == 1:
                output.copy_(local)
            else:
                self._param_gather_work = dist.all_gather_into_tensor(
                    output, local, group=self.gather_group, async_op=True
                )
        if self._param_gather_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._param_gather_event)
            self._param_gather_event = None
            self._param_gather_work = None
        elif self._param_gather_work is not None:
            self._param_gather_work.wait()
            self._param_gather_work = None
        self._release_local_compute_buffer()
        self._full_ready = True

    def release_full_parameters(self) -> None:
        self.install_sharded_parameters()
        if self._param_gather_event is not None:
            self._param_gather_event.synchronize()
            self._param_gather_event = None
        if self._param_gather_work is not None:
            self._param_gather_work.wait()
            self._param_gather_work = None
        if self._full_lease is not None:
            self._full_lease.release()
            self._full_lease = None
        self._release_local_compute_buffer()
        self.full_buffer = torch.empty(
            0, dtype=self.policy.compute_dtype, device=self.device
        )

        self._full_ready = False

    def discard_full_parameter_views(self) -> None:
        """Release full-parameter references after autograd no longer needs them."""
        empty = torch.empty(0, dtype=self.policy.compute_dtype, device=self.device)
        with torch.no_grad():
            for spec in self.specs:
                spec.full_param.data = empty

    def _release_local_compute_buffer(self) -> None:
        if self._local_compute_lease is not None:
            self._local_compute_lease.release()
            self._local_compute_lease = None
        if self.policy.main_params_dtype != self.policy.compute_dtype:
            self.local_compute_buffer = torch.empty(
                0, dtype=self.policy.compute_dtype, device=self.device
            )

    def _release_local_grad_comm_buffer(self) -> None:
        if self._local_grad_comm_lease is not None:
            self._local_grad_comm_lease.release()
            self._local_grad_comm_lease = None
        if self.policy.grad_comm_dtype != self.policy.main_grads_dtype:
            self.local_grad_comm_buffer = torch.empty(
                0, dtype=self.policy.grad_comm_dtype, device=self.device
            )
        else:
            self.local_grad_comm_buffer = self.main_grad_buffer

    def move_model_state(self, device: torch.device, *, load_grad: bool) -> None:
        """Move persistent sharded storage without breaking optimizer aliases."""
        device = torch.device(device)
        self.release_full_parameters()
        self.discard_full_parameter_views()
        if self._grad_reduce_launched:
            self.wait_grad_reduce()
        else:
            self._release_full_main_grads()
        self.allocator.release_cached()

        grad_present = {
            id(spec): spec.shard_param is not None and spec.shard_param.grad is not None
            for spec in self.specs
        }
        compute_shares_main = self.local_compute_buffer is self.main_param_buffer
        self.main_param_buffer = self.main_param_buffer.to(device)
        grad_comm_shares_main = self.local_grad_comm_buffer is self.main_grad_buffer
        self.main_grad_buffer = self.main_grad_buffer.to(device)
        self.grad_shard_buffer = self.main_grad_buffer
        self.local_compute_buffer = (
            self.main_param_buffer
            if compute_shares_main
            else self.local_compute_buffer.to(device)
        )
        self.local_grad_comm_buffer = (
            self.main_grad_buffer
            if grad_comm_shares_main
            else self.local_grad_comm_buffer.to(device)
        )
        self.device = device
        self.full_buffer = torch.empty(
            0, dtype=self.policy.compute_dtype, device=device
        )
        self.full_main_grad_buffer = torch.empty(
            0, dtype=self.policy.main_grads_dtype, device=device
        )

        with torch.no_grad():
            for spec in self.specs:
                assert spec.shard_param is not None
                spec.shard_param.data = self.main_param_buffer.narrow(
                    0, spec.local_offset, spec.shard_numel
                )
                if load_grad and grad_present[id(spec)]:
                    spec.shard_param.grad = self.main_grad_buffer.narrow(
                        0, spec.local_offset, spec.shard_numel
                    ).view_as(spec.shard_param)
                else:
                    spec.shard_param.grad = None
                spec.full_param.data = self.full_buffer

    def release_scratch_keep_weights(self) -> None:
        """Drop the all-gather scratch while leaving the sharded weights resident.

        This is ``move_model_state``'s scratch-release prologue without the
        device move: release the full-parameter all-gather buffer, discard the
        full-parameter views, drain any in-flight grad reduce, and hand the
        allocator's cached double-buffer slots back to the driver -- but keep
        ``main_param_buffer`` (and the optimizer-aliased shard views installed by
        ``release_full_parameters``) on their current device. A colocated vLLM
        weight-pool wake needs the transient full-model scratch back before it
        can ``create_and_map``; the export that follows the wake gathers from the
        resident shards, so the persistent weights must not move.
        """
        self.release_full_parameters()
        self.discard_full_parameter_views()
        if self._grad_reduce_launched:
            self.wait_grad_reduce()
        else:
            self._release_full_main_grads()
        self.allocator.release_cached()

    def prepare_grad_reduce(
        self, *, force: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._grad_reduce_launched or self._microbatch_reduced:
            return None
        if not force and len(self._grad_ready_ids) != len(self.specs):
            return None
        self._grad_reduce_launched = True
        self.prepare_main_grads()
        if (
            self.policy.grad_comm_dtype != self.policy.main_grads_dtype
            or self._has_accumulated_grad
        ):
            self._local_grad_comm_lease = self.allocator.allocate(
                self.local_numel,
                dtype=self.policy.grad_comm_dtype,
                device=self.device,
                group=self.process_group,
                key=("grad-local", id(self.process_group)),
            )
            self.local_grad_comm_buffer = self._local_grad_comm_lease.tensor
        with torch.no_grad():
            for spec in self.specs:
                grad = spec.full_param.grad
                if grad is not None and not spec.full_param.grad_added_to_main_grad:
                    spec.full_param.main_grad.add_(grad)
                spec.full_param.grad = None
        if self.policy.grad_comm_dtype == self.policy.main_grads_dtype:
            grad_input = self.full_main_grad_buffer
        else:
            self._grad_lease = self.allocator.allocate(
                self.full_numel,
                dtype=self.policy.grad_comm_dtype,
                device=self.device,
                group=self.process_group,
                key=("grad", id(self.process_group)),
            )
            grad_input = self._grad_lease.tensor
            grad_input.copy_(self.full_main_grad_buffer)
        if self.local_grad_comm_buffer.numel() != self.local_numel:
            raise RuntimeError(
                "M-FSDP reduce-scatter output staging has the wrong size: "
                f"bucket={self.bucket_id} output={self.local_grad_comm_buffer.numel()} "
                f"expected={self.local_numel} accumulated={self._has_accumulated_grad} "
                f"lease={self._local_grad_comm_lease is not None}."
            )
        return self.local_grad_comm_buffer, grad_input

    def mark_grad_reduce_launched(
        self, work: Any | None, completion_event: Any | None = None
    ) -> None:
        self._grad_reduce_work = work
        self._grad_reduce_event = completion_event

    def wait_grad_reduce(self) -> None:
        if self._grad_reduce_finished:
            return
        if not self._grad_reduce_launched:
            tensors = self.prepare_grad_reduce(force=True)
            assert tensors is not None
            output, grad_input = tensors
            if self.world_size == 1:
                output.copy_(grad_input)
            else:
                self._grad_reduce_work = dist.reduce_scatter_tensor(
                    output,
                    grad_input,
                    op=dist.ReduceOp.SUM,
                    group=self.process_group,
                    async_op=True,
                )
        if self._grad_reduce_event is not None:
            self._grad_reduce_event.synchronize()
            self._grad_reduce_event = None
        if self._grad_reduce_work is not None:
            self._grad_reduce_work.wait()
            self._grad_reduce_work = None
        reduced_grad = self.local_grad_comm_buffer
        if self.config.average_gradients and self.world_size > 1:
            reduced_grad.div_(self.world_size)
        if self._has_accumulated_grad:
            self.main_grad_buffer.add_(reduced_grad)
        elif reduced_grad is not self.main_grad_buffer:
            self.main_grad_buffer.copy_(reduced_grad)
        self._has_accumulated_grad = True
        for spec in self.specs:
            assert spec.shard_param is not None
            main_grad = self.main_grad_buffer.narrow(
                0, spec.local_offset, spec.shard_numel
            ).view_as(spec.shard_param)
            if spec.shard_param.dtype == main_grad.dtype:
                spec.shard_param.grad = main_grad
            else:
                # PyTorch requires ``Parameter.grad`` to have the parameter's
                # dtype. The CPU optimizer consumes this explicit FP32 view;
                # leave ``.grad`` empty instead of allocating a lossy BF16
                # mirror that no optimizer should read.
                spec.shard_param.main_grad = main_grad
                spec.shard_param.grad = None
        if self._grad_lease is not None:
            self._grad_lease.release()
            self._grad_lease = None
        self._release_local_grad_comm_buffer()
        self._release_full_main_grads()
        self._grad_reduce_launched = False
        self._grad_reduce_finished = True
        self._grad_ready_ids.clear()
        self._microbatch_reduced = True

    def wait_for_inflight_grad_reduce(self) -> None:
        """Drain already-launched reduce work without starting new work.

        Exception teardown must not call :meth:`wait_grad_reduce`: that method
        intentionally launches a missing reduction for the normal completion
        path.  Here we only wait on work that exists, before its staging leases
        and their shared allocator can be reclaimed.
        """
        error: BaseException | None = None
        if self._grad_reduce_event is not None:
            try:
                self._grad_reduce_event.synchronize()
            except BaseException as caught:
                error = caught
            finally:
                self._grad_reduce_event = None
        if self._grad_reduce_work is not None:
            try:
                self._grad_reduce_work.wait()
            except BaseException as caught:
                if error is None:
                    error = caught
            finally:
                self._grad_reduce_work = None
        if error is not None:
            raise error

    def copy_full_parameters_to_shards(self) -> None:
        self.wait_param_gather()
        local_begin = self.rank * self.local_numel
        self.main_param_buffer.copy_(
            self.full_buffer.narrow(0, local_begin, self.local_numel)
        )

    def reset_grad_state(self) -> None:
        if self._grad_reduce_work is not None:
            self._grad_reduce_work.wait()
        self._grad_reduce_work = None
        self._grad_reduce_event = None
        self._grad_reduce_launched = False
        self._grad_reduce_finished = False
        self._microbatch_reduced = False
        self._has_accumulated_grad = False
        self.grad_sync_enabled = False
        self._grad_ready_ids.clear()
        self.main_grad_buffer.zero_()
        self._release_local_grad_comm_buffer()
        for spec in self.specs:
            spec.full_param.grad = None
            spec.full_param.grad_added_to_main_grad = False
            if spec.shard_param is not None:
                spec.shard_param.grad = None
        self._release_full_main_grads()

    def start_microbatch(self) -> None:
        self._microbatch_reduced = False
        self._grad_reduce_finished = False

    def set_grad_sync_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self.grad_sync_enabled:
            self._grad_ready_ids.clear()
        self.grad_sync_enabled = enabled

    def saved_view(self, tensor: torch.Tensor) -> SavedParamView | None:
        if not self._full_ready or tensor.numel() == 0:
            return None
        if _storage_pointer(tensor) != _storage_pointer(self.full_buffer):
            return None
        return SavedParamView(
            bucket=self,
            size=tuple(tensor.size()),
            stride=tuple(tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
        )

    def restore_saved_view(self, view: SavedParamView) -> torch.Tensor:
        self.wait_param_gather()
        return torch.as_strided(
            self.full_buffer,
            size=view.size,
            stride=view.stride,
            storage_offset=view.storage_offset,
        )

    def _make_grad_ready_hook(self, spec: ParamSpec) -> Callable[[nn.Parameter], None]:
        def grad_ready(param: nn.Parameter) -> None:
            main_grad = param.get_main_grad()
            if not param.grad_added_to_main_grad:
                with torch.no_grad():
                    if param.grad is None:
                        main_grad.zero_()
                    else:
                        # This is the data-distributed MCore path: each
                        # microbatch owns a fresh unsharded communication
                        # bucket, so copy rather than accumulate into it.
                        main_grad.copy_(param.grad)
            if param.grad is not None:
                param.grad = None
            # MCore treats this as a one-backward notification from TE.  It
            # must not leak into the next microbatch's accumulation decision.
            param.grad_added_to_main_grad = False
            # get_main_grad() may retire this same bucket's previous
            # microbatch, whose completion clears the old ready set. Mark the
            # current parameter only after that retirement.
            self._grad_ready_ids.add(id(spec))
            if len(self._grad_ready_ids) != len(self.specs):
                return
            if self.grad_ready_callback is not None:
                self.grad_ready_callback(self)
            self.release_full_parameters()
            self.discard_full_parameter_views()
            if not self.grad_sync_enabled:
                self._grad_ready_ids.clear()

        return grad_ready


class ParamAndGradBuffer:
    """Build dense/expert buckets and expose their owning module boundaries."""

    def __init__(
        self,
        module: nn.Module,
        *,
        groups: MFSDPProcessGroups,
        config: MFSDPConfig,
        is_expert: Callable[[str], bool],
        unit_modules: Iterable[type[nn.Module] | str] | None,
    ) -> None:
        self.module = module
        self.groups = groups
        self.config = config
        self.allocator = build_temporary_allocator(config, groups.registration_groups())
        self.buckets, self.owners = self._build(
            is_expert=is_expert, unit_modules=unit_modules or ()
        )

    def _build(
        self,
        *,
        is_expert: Callable[[str], bool],
        unit_modules: Iterable[type[nn.Module] | str],
    ) -> tuple[list[ParamBucket], dict[int, list[int]]]:
        module_by_name = dict(self.module.named_modules())
        module_order = {
            id(value): index for index, value in enumerate(module_by_name.values())
        }
        unit_types = _resolve_module_types(unit_modules)
        specs_by_id: dict[int, ParamSpec] = {}
        owner_by_param_id: dict[int, nn.Module] = {}
        expert_by_param_id: dict[int, bool] = {}

        for name, param in self.module.named_parameters(remove_duplicate=False):
            if not param.requires_grad:
                continue
            parent_name, _, attribute = name.rpartition(".")
            parent = module_by_name[parent_name]
            spec = specs_by_id.get(id(param))
            if spec is None:
                spec = ParamSpec(
                    name=name,
                    full_param=param,
                    bindings=[],
                    shape=param.shape,
                    numel=param.numel(),
                )
                specs_by_id[id(param)] = spec
                owner_by_param_id[id(param)] = _parameter_owner(
                    parent_name, parent, module_by_name, unit_types
                )
                expert_by_param_id[id(param)] = bool(is_expert(name))
            spec.bindings.append(ParamBinding(parent, attribute))

        grouped: dict[
            tuple[int, bool, torch.dtype, torch.device], list[ParamSpec]
        ] = defaultdict(list)
        owner_for_key: dict[
            tuple[int, bool, torch.dtype, torch.device], nn.Module
        ] = {}
        for param_id, spec in specs_by_id.items():
            owner = owner_by_param_id[param_id]
            expert = expert_by_param_id[param_id]
            key = (
                module_order[id(owner)],
                expert,
                spec.full_param.dtype,
                spec.full_param.device,
            )
            grouped[key].append(spec)
            owner_for_key[key] = owner

        buckets: list[ParamBucket] = []
        owners: dict[int, list[int]] = defaultdict(list)
        for key in sorted(grouped, key=lambda item: item[0]):
            _owner_order, expert, _dtype, _device = key
            owner = owner_for_key[key]
            partitions = _split_specs(grouped[key], self.config.bucket_size)
            owner_type = type(owner)
            owner_layout = f"{owner_type.__module__}.{owner_type.__qualname__}"
            for partition_index, partition in enumerate(partitions):
                bucket_id = len(buckets)
                buckets.append(
                    ParamBucket(
                        bucket_id,
                        partition,
                        process_group=self.groups.data_group(expert=expert),
                        gather_group=self.groups.gather_group(expert=expert),
                        config=self.config,
                        allocator=self.allocator,
                        allocator_layout_key=(
                            owner_layout,
                            expert,
                            partition_index,
                            len(partitions),
                            sum(spec.numel for spec in partition),
                        ),
                    )
                )
                owners[id(owner)].append(bucket_id)
        return buckets, dict(owners)


def _split_specs(
    specs: list[ParamSpec], bucket_size: int | None
) -> list[list[ParamSpec]]:
    if bucket_size is None:
        return [specs]
    result: list[list[ParamSpec]] = []
    current: list[ParamSpec] = []
    current_numel = 0
    for spec in specs:
        if current and current_numel + spec.numel > bucket_size:
            result.append(current)
            current = []
            current_numel = 0
        current.append(spec)
        current_numel += spec.numel
    if current:
        result.append(current)
    return result


def _parameter_owner(
    parent_name: str,
    parent: nn.Module,
    module_by_name: dict[str, nn.Module],
    unit_types: tuple[type[nn.Module], ...],
) -> nn.Module:
    if not unit_types:
        return parent
    parts = parent_name.split(".") if parent_name else []
    for end in range(len(parts), -1, -1):
        candidate = module_by_name[".".join(parts[:end])]
        if isinstance(candidate, unit_types):
            return candidate
    return module_by_name[""]


def _resolve_module_types(
    module_types: Iterable[type[nn.Module] | str],
) -> tuple[type[nn.Module], ...]:
    resolved: list[type[nn.Module]] = []
    for value in module_types:
        if isinstance(value, type) and issubclass(value, nn.Module):
            resolved.append(value)
            continue
        if not isinstance(value, str) or "." not in value:
            raise TypeError(f"Invalid M-FSDP unit module: {value!r}")
        module_name, _, attribute = value.rpartition(".")
        module_type = getattr(importlib.import_module(module_name), attribute)
        if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
            raise TypeError(f"M-FSDP unit is not an nn.Module type: {value!r}")
        resolved.append(module_type)
    return tuple(resolved)


def _copy_parameter_metadata(source: nn.Parameter, target: nn.Parameter) -> None:
    for name, value in source.__dict__.items():
        if name not in {"grad", "_grad"}:
            setattr(target, name, value)


def _storage_pointer(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return 0


class CommunicationStream:
    """Launch collectives on a dedicated CUDA stream when CUDA is available."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.stream: torch.cuda.Stream | None = None
        if device.type == "cuda":
            with torch.cuda.device(device):
                self.stream = torch.cuda.Stream(device=device, priority=-1)

    def launch(
        self, callback: Callable[[], Any | None], tensors: Iterable[torch.Tensor]
    ) -> Any | None:
        if self.stream is None:
            return callback()
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            work = callback()
            for tensor in tensors:
                tensor.record_stream(self.stream)
        return work

    def wait_for_current(self) -> None:
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def record_event(self) -> torch.cuda.Event | None:
        if self.stream is None:
            return None
        event = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            event.record(self.stream)
        return event


def _resolve_suggested_communication_unit_size(
    buckets: list["ParamBucket"],
    owner_bucket_ids: Iterable[Iterable[int]] | None,
    *,
    explicit: int | None,
) -> int:
    """Resolve MCore's shared RS queue / AG prefetch element budget."""
    configured = explicit
    if configured is None and buckets:
        configured = getattr(
            buckets[0].config, "suggested_communication_unit_size", None
        )
    if configured is not None:
        return int(configured)

    groups = tuple(tuple(group) for group in (owner_bucket_ids or ()))
    if groups:
        total_elements = sum(
            sum(buckets[bucket_id].full_numel for bucket_id in group)
            for group in groups
        )
        average_owner_elements = total_elements // len(groups)
        suggested = average_owner_elements * 2
    else:
        suggested = 1_000_000_000
    # This follows MCore exactly: its default is never smaller than 1B
    # elements, while an explicitly configured value is accepted as-is.
    return max(1_000_000_000, suggested)


class AllGatherPipeline:
    """Prefetch parameter buckets in forward and reverse-backward order."""

    def __init__(
        self,
        buckets: list[ParamBucket],
        owner_bucket_ids: Iterable[Iterable[int]] | None = None,
        *,
        suggested_communication_unit_size: int | None = None,
    ) -> None:
        self.buckets = buckets
        self.overlap = bool(buckets and buckets[0].config.overlap_param_gather)
        owner_bucket_ids = tuple(tuple(group) for group in (owner_bucket_ids or ()))
        self.suggested_prefetch_elements = (
            _resolve_suggested_communication_unit_size(
                buckets,
                owner_bucket_ids,
                explicit=suggested_communication_unit_size,
            )
            // 2
        )
        groups = owner_bucket_ids or ((bucket.bucket_id,) for bucket in buckets)
        self._bucket_groups = sorted(
            (tuple(sorted(set(group))) for group in groups),
            key=lambda group: group[0],
        )
        self._bucket_to_group = {
            bucket_id: index
            for index, group in enumerate(self._bucket_groups)
            for bucket_id in group
        }
        device = buckets[0].device if buckets else torch.device("cpu")
        self.comm_stream = CommunicationStream(device)
        self._forward_cursor = 0
        self._backward_cursor = len(buckets) - 1

    def async_bucket_gather(self, bucket_id: int, bwd: bool = False) -> None:
        bucket = self.buckets[bucket_id]
        if (
            bucket._full_ready
            or bucket._param_gather_work is not None
            or bucket._param_gather_event is not None
        ):
            return
        output, local = bucket.prepare_param_gather()

        def collective():
            if bucket.world_size == 1:
                output.copy_(local)
                return None
            return dist.all_gather_into_tensor(
                output, local, group=bucket.gather_group, async_op=True
            )

        work = self.comm_stream.launch(collective, (output, local))
        bucket.mark_param_gather_launched(work, self.comm_stream.record_event())

    def wait_bucket_ready(self, bucket_id: int, bwd: bool = False) -> None:
        bucket = self.buckets[bucket_id]
        if (
            self.overlap
            and not bucket._full_ready
            and bucket._param_gather_work is None
            and bucket._param_gather_event is None
        ):
            # Prefetch/overlap path: launch the dense all-gather ahead of use on the
            # dedicated priority communication stream.
            self.async_bucket_gather(bucket_id, bwd=bwd)
        # When overlap is disabled (the guard forces this for any CP>=2 + DP-sharded
        # config, see optimizer._order_param_gathers_for_parallel_collectives), do NOT
        # launch on the side comm stream. Let wait_param_gather issue the all-gather on
        # the current compute stream instead, so dense_ag (gather_group == dp_cp_ag_group)
        # is enqueued in program order alongside the model's CP collectives (cp_group).
        # Those two process groups intersect; launching them from different streams lets
        # their NCCL enqueue order diverge across ranks on multi-node runs -> collective
        # deadlock (invisible on single-node NVLink proxies). Program-order serialization
        # on one stream mirrors how FSDP2 avoids the same hazard.
        bucket.wait_param_gather()

    def begin_forward(self) -> None:
        self.release_all()
        self._forward_cursor = 0
        if self.buckets and self.overlap:
            self.async_bucket_gather(0)

    def acquire_forward(self, bucket_ids: Iterable[int]) -> None:
        requested = tuple(sorted(set(bucket_ids)))
        self._launch_with_prefetch(requested, bwd=False)
        for bucket_id in requested:
            self.wait_bucket_ready(bucket_id)
            self.buckets[bucket_id].install_full_parameters()
        if requested:
            self._forward_cursor = max(self._forward_cursor, requested[-1] + 1)

    def _launch_with_prefetch(self, bucket_ids: tuple[int, ...], *, bwd: bool) -> None:
        if not bucket_ids:
            return
        # The non-overlap path deliberately gathers on the current stream in
        # wait_bucket_ready() so intersecting CP/DP process groups keep one
        # program order across ranks.
        if not self.overlap:
            return
        for bucket_id in bucket_ids:
            self.async_bucket_gather(bucket_id, bwd=bwd)

        direction = -1 if bwd else 1
        edge_bucket = bucket_ids[0] if bwd else bucket_ids[-1]
        group_index = self._bucket_to_group[edge_bucket] + direction
        resident_groups = {self._bucket_to_group[bucket_id] for bucket_id in bucket_ids}
        double_buffer = bool(
            getattr(self.buckets[0].config, "fsdp_double_buffer", False)
        )
        if double_buffer and len(resident_groups) > 2:
            raise ValueError(
                "M-FSDP double buffers cannot materialize more than two owners at once."
            )
        prefetched = 0
        while 0 <= group_index < len(self._bucket_groups):
            if prefetched >= self.suggested_prefetch_elements:
                break
            if double_buffer and group_index not in resident_groups:
                if len(resident_groups) >= 2:
                    break
                resident_groups.add(group_index)
            group = self._bucket_groups[group_index]
            for bucket_id in group:
                self.async_bucket_gather(bucket_id, bwd=bwd)
                prefetched += self.buckets[bucket_id].full_numel
            group_index += direction

    def begin_backward(self) -> None:
        self._backward_cursor = len(self.buckets) - 1
        if self.buckets and self.overlap:
            self.async_bucket_gather(self._backward_cursor, bwd=True)

    def acquire_backward(self, bucket: ParamBucket) -> None:
        bucket_id = bucket.bucket_id
        self.wait_bucket_ready(bucket_id, bwd=True)
        bucket.install_full_parameters()
        self._backward_cursor = min(self._backward_cursor, bucket_id - 1)
        if self.overlap and self._backward_cursor >= 0:
            self.async_bucket_gather(self._backward_cursor, bwd=True)

    def acquire_backward_ids(self, bucket_ids: Iterable[int]) -> None:
        requested = tuple(sorted(set(bucket_ids)))
        self._launch_with_prefetch(requested, bwd=True)
        for bucket_id in requested:
            self.wait_bucket_ready(bucket_id, bwd=True)
            self.buckets[bucket_id].install_full_parameters()
        if requested:
            self._backward_cursor = min(self._backward_cursor, requested[0] - 1)

    def materialize_all(self) -> None:
        for bucket in self.buckets:
            self.async_bucket_gather(bucket.bucket_id)
        for bucket in self.buckets:
            self.wait_bucket_ready(bucket.bucket_id)
            bucket.install_full_parameters()

    def release_all(self) -> None:
        for bucket in self.buckets:
            bucket.release_full_parameters()

    def reset_device(self, device: torch.device) -> None:
        self.comm_stream = CommunicationStream(device)


class GradReducePipeline:
    """Launch ready reduce-scatter buckets on a communication stream."""

    def __init__(
        self,
        buckets: list[ParamBucket],
        owner_bucket_ids: Iterable[Iterable[int]] | None = None,
        *,
        suggested_communication_unit_size: int | None = None,
    ) -> None:
        self.buckets = buckets
        self._bucket_ids = {
            id(bucket): getattr(bucket, "bucket_id", index)
            for index, bucket in enumerate(buckets)
        }
        groups = tuple(tuple(sorted(set(group))) for group in (owner_bucket_ids or ()))
        self._bucket_groups = tuple(sorted(groups, key=lambda group: group[0])) or tuple(
            (self._bucket_ids[id(bucket)],) for bucket in buckets
        )
        self._bucket_to_group = {
            bucket_id: group
            for group in self._bucket_groups
            for bucket_id in group
        }
        device = buckets[0].device if buckets else torch.device("cpu")
        self.comm_stream = CommunicationStream(device)
        self._pending: list[tuple[ParamBucket, int]] = []
        self._pending_elements = 0
        self._microbatch_id = 0
        self._bucket_ready_microbatch: dict[int, int] = {}
        self._group_launched_microbatch: dict[int, int] = {}
        self._pending_capacity_elements = _resolve_suggested_communication_unit_size(
            buckets,
            owner_bucket_ids,
            explicit=suggested_communication_unit_size,
        )
        for bucket in buckets:
            bucket.grad_ready_callback = self.reduce_gradients
            bucket.before_main_grad_allocate = self._prepare_main_grad_allocate

    def _retire_oldest(self) -> None:
        completed, completed_elements = self._pending.pop(0)
        completed.wait_grad_reduce()
        self._pending_elements -= completed_elements

    def _enforce_double_buffer_limit(self, incoming: ParamBucket) -> None:
        """Keep at most two live full-gradient bucket slots.

        This is the MCore ``_enforce_double_buffer_limit`` lifecycle at the
        lazy main-gradient allocation boundary.  Slot pressure is a count of
        in-flight bucket allocations, not a byte watermark: differently sized
        buckets still each consume one of the allocator's two reusable slots.
        """
        if not incoming.config.fsdp_double_buffer:
            return
        while len(self._pending) >= 2:
            self._retire_oldest()

    def _prepare_main_grad_allocate(self, incoming: ParamBucket) -> None:
        # This standalone bucket object owns one staging lease.  Preserve
        # MCore's cross-microbatch overlap by carrying reductions through the
        # next forward, then retire in FIFO order immediately before the same
        # bucket's staging is reused by backward.
        retired_incoming = False
        while any(bucket is incoming for bucket, _elements in self._pending):
            retired_incoming = retired_incoming or self._pending[0][0] is incoming
            self._retire_oldest()
        if retired_incoming:
            # wait_grad_reduce() finalized the previous microbatch after this
            # microbatch's begin_backward() reset. Re-open the bucket for the
            # reduction that will be produced by the current backward.
            incoming.start_microbatch()
        self._enforce_double_buffer_limit(incoming)

    def _ready_bucket_group(self, bucket: ParamBucket) -> tuple[int, ...] | None:
        """Return the deterministic owner group once every bucket is ready.

        MCore does not issue reduce-scatter directly from a rank-local bucket
        completion hook.  It waits for the whole FSDP-unit bucket group and
        launches that group in bucket-id order, otherwise conditional/unused
        gradients can give different ranks different collective sequences.
        """
        bucket_id = self._bucket_ids[id(bucket)]
        group = self._bucket_to_group.get(bucket_id, (bucket_id,))
        self._bucket_ready_microbatch[bucket_id] = self._microbatch_id
        group_key = group[0]
        if self._group_launched_microbatch.get(group_key) == self._microbatch_id:
            return None
        if any(
            self._bucket_ready_microbatch.get(bucket_id) != self._microbatch_id
            for bucket_id in group
        ):
            return None
        self._group_launched_microbatch[group_key] = self._microbatch_id
        return group

    def _reduce_bucket(self, bucket: ParamBucket, *, force: bool = False) -> None:
        self._wait_for_previous_grad_reduce()
        tensors = bucket.prepare_grad_reduce(force=force)
        if tensors is None:
            return
        output, grad_input = tensors

        def collective():
            if bucket.world_size == 1:
                output.copy_(grad_input)
                return None
            return dist.reduce_scatter_tensor(
                output,
                grad_input,
                op=dist.ReduceOp.SUM,
                group=bucket.process_group,
                async_op=True,
            )

        work = self.comm_stream.launch(collective, (output, grad_input))
        completion_event = None
        if self.comm_stream.stream is not None:
            completion_event = torch.cuda.Event()
            completion_event.record(self.comm_stream.stream)
        bucket.mark_grad_reduce_launched(work, completion_event)
        element_count = grad_input.numel()
        self._pending.append((bucket, element_count))
        self._pending_elements += element_count

    def reduce_gradients(self, bucket: ParamBucket, *, force: bool = False) -> None:
        if force:
            bucket_id = self._bucket_ids[id(bucket)]
            group = self._bucket_to_group.get(bucket_id, (bucket_id,))
        else:
            group = self._ready_bucket_group(bucket)
            if group is None:
                return
        for bucket_id in group:
            self._reduce_bucket(self.buckets[bucket_id], force=force)

    def _wait_for_previous_grad_reduce(self) -> None:
        while (
            self._pending
            and self._pending_elements > self._pending_capacity_elements
        ):
            self._retire_oldest()

    def has_microbatch_work(self) -> bool:
        return bool(self._pending) or any(
            bucket._grad_ready_ids or bucket._full_main_grad_lease is not None
            for bucket in self.buckets
        )

    def finish(self) -> None:
        # Finish whole owner groups in a globally deterministic order.  Calling
        # reduce_gradients(force=True) once per bucket would relaunch the same
        # group, so issue each unfinished member directly here.
        for group in self._bucket_groups:
            for bucket_id in group:
                bucket = self.buckets[bucket_id]
                if not bucket._microbatch_reduced:
                    self._reduce_bucket(bucket, force=True)
        self.comm_stream.wait_for_current()
        while self._pending:
            self._retire_oldest()

    def abort(self) -> None:
        """Forget queued work after a forward exception has become primary."""
        self._pending.clear()
        self._pending_elements = 0
        self._bucket_ready_microbatch.clear()
        self._group_launched_microbatch.clear()

    def start_microbatch(self) -> None:
        self._microbatch_id += 1
        for bucket in self.buckets:
            bucket.start_microbatch()

    def reset(self) -> None:
        self._pending.clear()
        self._pending_elements = 0
        self._microbatch_id = 0
        self._bucket_ready_microbatch.clear()
        self._group_launched_microbatch.clear()
        for bucket in self.buckets:
            bucket.reset_grad_state()

    def set_enabled(self, enabled: bool) -> None:
        for bucket in self.buckets:
            bucket.set_grad_sync_enabled(enabled)

    def reset_device(self, device: torch.device) -> None:
        self.comm_stream = CommunicationStream(device)


class CommunicationPipelines:
    """Join parameter all-gather and gradient reduce-scatter pipelines."""

    def __init__(
        self,
        buckets: list[ParamBucket],
        owner_bucket_ids: Iterable[Iterable[int]] | None = None,
    ) -> None:
        self.buckets = buckets
        owner_bucket_ids = tuple(tuple(group) for group in (owner_bucket_ids or ()))
        explicit = (
            buckets[0].config.suggested_communication_unit_size if buckets else None
        )
        self.all_gather = AllGatherPipeline(
            buckets,
            owner_bucket_ids,
            suggested_communication_unit_size=explicit,
        )
        self.grad_reduce = GradReducePipeline(
            buckets,
            owner_bucket_ids,
            suggested_communication_unit_size=explicit,
        )
        self._backward_started = False

    def begin_forward(self) -> None:
        self.all_gather.begin_forward()

    def acquire_forward(self, bucket_ids: Iterable[int]) -> None:
        self.all_gather.acquire_forward(bucket_ids)

    def begin_backward(self) -> bool:
        if self._backward_started:
            return False
        self._backward_started = True
        self.grad_reduce.start_microbatch()
        self.all_gather.begin_backward()
        return True

    def end_backward(self) -> None:
        self._backward_started = False

    def acquire_backward(self, bucket: ParamBucket) -> None:
        self.all_gather.acquire_backward(bucket)

    def acquire_backward_ids(self, bucket_ids: Iterable[int]) -> None:
        self.all_gather.acquire_backward_ids(bucket_ids)

    def release_backward_ids(self, bucket_ids: Iterable[int]) -> None:
        for bucket_id in bucket_ids:
            bucket = self.buckets[bucket_id]
            bucket.release_full_parameters()
            bucket.discard_full_parameter_views()

    def release_forward_ids(self, bucket_ids: Iterable[int]) -> None:
        for bucket_id in bucket_ids:
            bucket = self.buckets[bucket_id]
            bucket.release_full_parameters()
            bucket.discard_full_parameter_views()

    def end_forward(self) -> None:
        for bucket in self.buckets:
            bucket.release_full_parameters()
            bucket.discard_full_parameter_views()

    def materialize_all(self) -> None:
        self.all_gather.materialize_all()

    def stream_materialize_buckets(self) -> Iterator["ParamBucket"]:
        """Materialize one bucket at a time, releasing each before the next.

        ``materialize_all`` all-gathers *every* bucket into the persistent
        double-buffer simultaneously, so the whole unsharded model is resident
        at once -- a transient full-model peak (tens of GiB/rank) that lands on
        top of steady-state and drives the colocated resync OOM. A
        full-parameter *export* only ever reads one
        parameter at a time, so it does not need the whole model resident; this
        generator bounds the transient footprint to a single bucket by
        gather → install → yield → release per bucket, mirroring FSDP2's
        per-parameter ``full_tensor`` path (which retains no whole-model buffer
        and shows no export peak). Consumers MUST finish reading a bucket's
        parameters before requesting the next (plain iteration guarantees this).
        """
        for bucket in self.buckets:
            self.all_gather.async_bucket_gather(bucket.bucket_id)
            self.all_gather.wait_bucket_ready(bucket.bucket_id)
            bucket.install_full_parameters()
            try:
                yield bucket
            finally:
                bucket.release_full_parameters()
                bucket.discard_full_parameter_views()

    def release_all(self) -> None:
        self.all_gather.release_all()

    def discard_full_parameter_views(self) -> None:
        for bucket in self.buckets:
            bucket.discard_full_parameter_views()

    def move_model_state(self, device: torch.device, *, load_grad: bool) -> None:
        device = torch.device(device)
        for bucket in self.buckets:
            bucket.move_model_state(device, load_grad=load_grad)
        self.all_gather.reset_device(device)
        self.grad_reduce.reset_device(device)

    def release_scratch_keep_weights(self) -> None:
        """Reclaim every bucket's all-gather scratch, keeping weights resident.

        Per-bucket counterpart to ``move_model_state`` that stops short of the
        device move: it hands the retained double-buffer slots back to the driver
        (what a colocated vLLM wake needs) but leaves the sharded weights and
        their optimizer aliases in place as the export gather source. See
        ``ParamBucket.release_scratch_keep_weights``.
        """
        for bucket in self.buckets:
            bucket.release_scratch_keep_weights()

    def finish_grad_sync(self) -> None:
        self.grad_reduce.finish()
        # MCore synchronizes and resets the parameter-gather pipeline before
        # exposing optimizer shards.  This also releases backward-prefetched
        # buckets that were never consumed (for example at graph boundaries).
        self.all_gather.release_all()
        self.discard_full_parameter_views()

    def abort(self) -> None:
        """Best-effort two-phase exception teardown for shared communication storage."""
        self.grad_reduce.abort()
        self._backward_started = False

        # Phase one drains work and drops every bucket lease.  Continue after a
        # cleanup failure: a primary forward exception takes precedence, and
        # force-clearing a shared allocator early would invalidate another
        # bucket's live lease.
        for bucket in self.buckets:
            try:
                bucket.wait_for_inflight_grad_reduce()
            except BaseException:
                pass
            bucket._grad_reduce_launched = False
            for release in (
                bucket._release_local_grad_comm_buffer,
                bucket._release_full_main_grads,
                bucket.release_full_parameters,
                bucket.discard_full_parameter_views,
            ):
                try:
                    release()
                except BaseException:
                    pass

        # Phase two force-clears each shared allocator once, after every lease
        # that can refer to it has been released.
        seen_allocators: set[int] = set()
        for bucket in self.buckets:
            allocator = bucket.allocator
            if id(allocator) in seen_allocators:
                continue
            seen_allocators.add(id(allocator))
            try:
                allocator.release_cached(force=True)
            except BaseException:
                pass

    def reset_grad_state(self) -> None:
        self.grad_reduce.reset()

    def set_grad_sync_enabled(self, enabled: bool) -> None:
        self.grad_reduce.set_enabled(enabled)

    def copy_full_parameters_to_shards(self) -> None:
        for bucket in self.buckets:
            bucket.copy_full_parameters_to_shards()
        self.release_all()

    def pack_saved_tensor(self, tensor: torch.Tensor) -> torch.Tensor | SavedParamView:
        for bucket in self.buckets:
            view = bucket.saved_view(tensor)
            if view is not None:
                return view
        return tensor

    def unpack_saved_tensor(self, value: torch.Tensor | SavedParamView) -> torch.Tensor:
        if isinstance(value, SavedParamView):
            self.acquire_backward(value.bucket)
            return value.bucket.restore_saved_view(value)
        return value


__all__ = [
    "AllGatherPipeline",
    "BufferLease",
    "CommunicationPipelines",
    "CommunicationStream",
    "DoubleBufferAllocator",
    "GradReducePipeline",
    "NCCLUserBuffer",
    "ParamAndGradBuffer",
    "ParamBucket",
    "ParamSpec",
    "SavedParamView",
    "TemporaryBufferAllocator",
]

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
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.mfsdp.config import (
    MFSDPConfig,
    MFSDPProcessGroups,
    MixedPrecisionPolicy,
    group_rank,
    group_size,
)

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

    def release_cached(self) -> None:
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
        return super().allocate(
            numel,
            dtype=dtype,
            device=device,
            group=group,
            key=pool_key,
        )

    def release(self, lease: BufferLease) -> None:
        if lease.slot is not None:
            events = self._reuse_events.setdefault(lease.key, [None, None])
            events[lease.slot] = _record_reuse_event(lease.tensor)
            self._busy.get(lease.key, set()).discard(lease.slot)

    def release_cached(self) -> None:
        if any(self._busy.values()):
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
        self.policy = MixedPrecisionPolicy(
            compute_dtype=compute_dtype,
            main_params_dtype=config.main_params_dtype,
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
                self.local_numel,
                max(0, intersection_begin - local_begin),
            )
            spec.param_offset = min(
                spec.numel,
                max(0, intersection_begin - spec_begin),
            )
            offset = spec_end

        self.main_param_buffer = torch.zeros(
            self.local_numel,
            dtype=self.policy.main_params_dtype,
            device=self.device,
        )
        self.main_grad_buffer = torch.zeros(
            self.local_numel,
            dtype=self.policy.main_grads_dtype,
            device=self.device,
        )
        self.grad_shard_buffer = self.main_grad_buffer
        self.local_compute_buffer = torch.empty(
            self.local_numel,
            dtype=self.policy.compute_dtype,
            device=self.device,
        )
        self.local_grad_comm_buffer = (
            self.main_grad_buffer
            if self.policy.grad_comm_dtype == self.policy.main_grads_dtype
            else torch.empty(
                self.local_numel,
                dtype=self.policy.grad_comm_dtype,
                device=self.device,
            )
        )
        self.full_buffer = torch.empty(0, dtype=compute_dtype, device=self.device)
        self._full_lease: BufferLease | None = None
        self._grad_lease: BufferLease | None = None
        self._param_gather_work: Any | None = None
        self._grad_reduce_work: Any | None = None
        self._grad_reduce_launched = False
        self._full_ready = True
        self.grad_sync_enabled = False
        self.grad_ready_callback: Callable[["ParamBucket"], None] | None = None
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
                    shard_view,
                    requires_grad=spec.full_param.requires_grad,
                )
                _copy_parameter_metadata(spec.full_param, shard_param)
                shard_param._mfsdp_original_ndim = spec.full_param.ndim
                spec.shard_param = shard_param
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
            for binding in spec.bindings:
                setattr(binding.module, binding.attribute, spec.full_param)

    def prepare_param_gather(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._full_lease is None:
            self._full_lease = self.allocator.allocate(
                self.full_numel,
                dtype=self.policy.compute_dtype,
                device=self.device,
                group=self.gather_group,
                key=(
                    "param",
                    id(self.gather_group),
                    self.allocator_layout_key,
                ),
            )
            self.full_buffer = self._full_lease.tensor
        self.local_compute_buffer.copy_(self.main_param_buffer)
        return self.full_buffer, self.local_compute_buffer

    def mark_param_gather_launched(self, work: Any | None) -> None:
        self._param_gather_work = work

    def wait_param_gather(self) -> None:
        if self._full_ready:
            return
        if self._param_gather_work is None:
            output, local = self.prepare_param_gather()
            if self.world_size == 1:
                output.copy_(local)
            else:
                self._param_gather_work = dist.all_gather_into_tensor(
                    output,
                    local,
                    group=self.gather_group,
                    async_op=True,
                )
        if self._param_gather_work is not None:
            self._param_gather_work.wait()
            self._param_gather_work = None
        self._full_ready = True

    def release_full_parameters(self) -> None:
        self.install_sharded_parameters()
        if self._param_gather_work is not None:
            self._param_gather_work.wait()
            self._param_gather_work = None
        if self._full_lease is not None:
            self._full_lease.release()
            self._full_lease = None
        self.full_buffer = torch.empty(
            0,
            dtype=self.policy.compute_dtype,
            device=self.device,
        )

        self._full_ready = False

    def discard_full_parameter_views(self) -> None:
        """Release full-parameter references after autograd no longer needs them."""
        empty = torch.empty(
            0,
            dtype=self.policy.compute_dtype,
            device=self.device,
        )
        with torch.no_grad():
            for spec in self.specs:
                spec.full_param.data = empty

    def move_model_state(self, device: torch.device, *, load_grad: bool) -> None:
        """Move persistent sharded storage without breaking optimizer aliases."""
        device = torch.device(device)
        self.release_full_parameters()
        self.discard_full_parameter_views()
        if self._grad_reduce_launched:
            self.wait_grad_reduce()
        self.allocator.release_cached()

        grad_present = {
            id(spec): spec.shard_param is not None and spec.shard_param.grad is not None
            for spec in self.specs
        }
        self.main_param_buffer = self.main_param_buffer.to(device)
        grad_comm_shares_main = self.local_grad_comm_buffer is self.main_grad_buffer
        self.main_grad_buffer = self.main_grad_buffer.to(device)
        self.grad_shard_buffer = self.main_grad_buffer
        self.local_compute_buffer = self.local_compute_buffer.to(device)
        self.local_grad_comm_buffer = (
            self.main_grad_buffer
            if grad_comm_shares_main
            else self.local_grad_comm_buffer.to(device)
        )
        self.device = device
        self.full_buffer = torch.empty(
            0,
            dtype=self.policy.compute_dtype,
            device=device,
        )

        with torch.no_grad():
            for spec in self.specs:
                assert spec.shard_param is not None
                spec.shard_param.data = self.main_param_buffer.narrow(
                    0,
                    spec.local_offset,
                    spec.shard_numel,
                )
                if load_grad and grad_present[id(spec)]:
                    spec.shard_param.grad = self.main_grad_buffer.narrow(
                        0,
                        spec.local_offset,
                        spec.shard_numel,
                    ).view_as(spec.shard_param)
                else:
                    spec.shard_param.grad = None
                spec.full_param.data = self.full_buffer

    def prepare_grad_reduce(
        self, *, force: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._grad_reduce_launched:
            return None
        if not force and len(self._grad_ready_ids) != len(self.specs):
            return None
        self._grad_reduce_launched = True
        self._grad_lease = self.allocator.allocate(
            self.full_numel,
            dtype=self.policy.grad_comm_dtype,
            device=self.device,
            group=self.process_group,
            key=(
                "grad",
                id(self.process_group),
                self.allocator_layout_key,
            ),
        )
        grad_input = self._grad_lease.tensor
        grad_input.zero_()
        with torch.no_grad():
            for spec in self.specs:
                grad = spec.full_param.grad
                if grad is not None:
                    grad_input.narrow(0, spec.full_offset, spec.numel).copy_(
                        grad.reshape(-1)
                    )
                spec.full_param.grad = None
        return self.local_grad_comm_buffer, grad_input

    def mark_grad_reduce_launched(self, work: Any | None) -> None:
        self._grad_reduce_work = work

    def wait_grad_reduce(self) -> None:
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
        if self._grad_reduce_work is not None:
            self._grad_reduce_work.wait()
            self._grad_reduce_work = None
        self.main_grad_buffer.copy_(self.local_grad_comm_buffer)
        if self.config.average_gradients and self.world_size > 1:
            self.main_grad_buffer.div_(self.world_size)
        for spec in self.specs:
            assert spec.shard_param is not None
            spec.shard_param.grad = self.main_grad_buffer.narrow(
                0, spec.local_offset, spec.shard_numel
            ).view_as(spec.shard_param)
        if self._grad_lease is not None:
            self._grad_lease.release()
            self._grad_lease = None

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
        self._grad_reduce_launched = False
        self.grad_sync_enabled = False
        self._grad_ready_ids.clear()
        self.main_grad_buffer.zero_()
        for spec in self.specs:
            spec.full_param.grad = None
            if spec.shard_param is not None:
                spec.shard_param.grad = None

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
            if param.grad is None:
                return
            self._grad_ready_ids.add(id(spec))
            if len(self._grad_ready_ids) != len(self.specs):
                return
            if self.grad_sync_enabled and self.grad_ready_callback is not None:
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
        self.allocator = build_temporary_allocator(
            config,
            groups.registration_groups(),
        )
        self.buckets, self.owners = self._build(
            is_expert=is_expert,
            unit_modules=unit_modules or (),
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
                    parent_name,
                    parent,
                    module_by_name,
                    unit_types,
                )
                expert_by_param_id[id(param)] = bool(is_expert(name))
            spec.bindings.append(ParamBinding(parent, attribute))

        grouped: dict[
            tuple[int, bool, bool, torch.dtype, torch.device], list[ParamSpec]
        ] = defaultdict(list)
        owner_for_key: dict[
            tuple[int, bool, bool, torch.dtype, torch.device], nn.Module
        ] = {}
        for param_id, spec in specs_by_id.items():
            owner = owner_by_param_id[param_id]
            expert = expert_by_param_id[param_id]
            retain_full_storage = len(spec.shape) == 1
            spec.retain_full_storage_through_backward = retain_full_storage
            key = (
                module_order[id(owner)],
                expert,
                retain_full_storage,
                spec.full_param.dtype,
                spec.full_param.device,
            )
            grouped[key].append(spec)
            owner_for_key[key] = owner

        buckets: list[ParamBucket] = []
        owners: dict[int, list[int]] = defaultdict(list)
        for key in sorted(grouped, key=lambda item: item[0]):
            _owner_order, expert, _retain_full_storage, _dtype, _device = key
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
                            _retain_full_storage,
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
        self,
        callback: Callable[[], Any | None],
        tensors: Iterable[torch.Tensor],
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


class AllGatherPipeline:
    """Prefetch parameter buckets in forward and reverse-backward order."""

    def __init__(self, buckets: list[ParamBucket]) -> None:
        self.buckets = buckets
        self.overlap = bool(buckets and buckets[0].config.overlap_param_gather)
        device = buckets[0].device if buckets else torch.device("cpu")
        self.comm_stream = CommunicationStream(device)
        self._forward_cursor = 0
        self._backward_cursor = len(buckets) - 1

    def async_bucket_gather(self, bucket_id: int, bwd: bool = False) -> None:
        bucket = self.buckets[bucket_id]
        if bucket._full_ready or bucket._param_gather_work is not None:
            return
        output, local = bucket.prepare_param_gather()

        def collective():
            if bucket.world_size == 1:
                output.copy_(local)
                return None
            return dist.all_gather_into_tensor(
                output,
                local,
                group=bucket.gather_group,
                async_op=True,
            )

        work = self.comm_stream.launch(collective, (output, local))
        bucket.mark_param_gather_launched(work)

    def wait_bucket_ready(self, bucket_id: int, bwd: bool = False) -> None:
        bucket = self.buckets[bucket_id]
        if self.overlap and not bucket._full_ready and bucket._param_gather_work is None:
            # Prefetch/overlap path: launch the dense all-gather ahead of use on the
            # dedicated priority communication stream.
            self.async_bucket_gather(bucket_id, bwd=bwd)
        self.comm_stream.wait_for_current()
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
        for bucket_id in bucket_ids:
            self.wait_bucket_ready(bucket_id)
            self.buckets[bucket_id].install_full_parameters()
            self._forward_cursor = max(self._forward_cursor, bucket_id + 1)
            if self.overlap and self._forward_cursor < len(self.buckets):
                self.async_bucket_gather(self._forward_cursor)

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

    def __init__(self, buckets: list[ParamBucket]) -> None:
        self.buckets = buckets
        device = buckets[0].device if buckets else torch.device("cpu")
        self.comm_stream = CommunicationStream(device)
        for bucket in buckets:
            if bucket.config.overlap_grad_reduce:
                bucket.grad_ready_callback = self.reduce_gradients

    def reduce_gradients(self, bucket: ParamBucket, *, force: bool = False) -> None:
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
        bucket.mark_grad_reduce_launched(work)

    def finish(self) -> None:
        for bucket in reversed(self.buckets):
            self.reduce_gradients(bucket, force=True)
        self.comm_stream.wait_for_current()
        for bucket in self.buckets:
            bucket.wait_grad_reduce()

    def reset(self) -> None:
        for bucket in self.buckets:
            bucket.reset_grad_state()

    def set_enabled(self, enabled: bool) -> None:
        for bucket in self.buckets:
            bucket.set_grad_sync_enabled(enabled)

    def reset_device(self, device: torch.device) -> None:
        self.comm_stream = CommunicationStream(device)


class CommunicationPipelines:
    """Join parameter all-gather and gradient reduce-scatter pipelines."""

    def __init__(self, buckets: list[ParamBucket]) -> None:
        self.buckets = buckets
        self.all_gather = AllGatherPipeline(buckets)
        self.grad_reduce = GradReducePipeline(buckets)

    def begin_forward(self) -> None:
        self.all_gather.begin_forward()

    def acquire_forward(self, bucket_ids: Iterable[int]) -> None:
        self.all_gather.acquire_forward(bucket_ids)

    def begin_backward(self) -> None:
        self.all_gather.begin_backward()

    def acquire_backward(self, bucket: ParamBucket) -> None:
        self.all_gather.acquire_backward(bucket)

    def acquire_backward_ids(self, bucket_ids: Iterable[int]) -> None:
        for bucket_id in reversed(tuple(bucket_ids)):
            self.all_gather.acquire_backward(self.buckets[bucket_id])

    def release_backward_ids(self, bucket_ids: Iterable[int]) -> None:
        for bucket_id in bucket_ids:
            bucket = self.buckets[bucket_id]
            bucket.release_full_parameters()
            bucket.discard_full_parameter_views()

    def release_forward_ids(self, bucket_ids: Iterable[int]) -> None:
        for bucket_id in bucket_ids:
            bucket = self.buckets[bucket_id]
            if not bucket.retain_full_storage_through_backward:
                bucket.release_full_parameters()
                bucket.discard_full_parameter_views()

    def end_forward(self) -> None:
        for bucket in self.buckets:
            if not bucket.retain_full_storage_through_backward:
                bucket.release_full_parameters()
                bucket.discard_full_parameter_views()

    def materialize_all(self) -> None:
        self.all_gather.materialize_all()

    def release_all(self) -> None:
        self.all_gather.release_all()

    def discard_full_parameter_views(self) -> None:
        for bucket in self.buckets:
            bucket.discard_full_parameter_views()

    def release_cached_buffers(self) -> None:
        """Return retained double-buffer all-gather storage to the driver.

        ``release_all`` only drops the pipeline's own references to the
        full-parameter buffers. Under ``fsdp_double_buffer`` (the default) the
        buffers stay pinned in ``DoubleBufferAllocator``'s persistent slots so
        they can be reused across the many acquire/release cycles of a training
        step -- torch's caching allocator (hence ``empty_cache`` and
        ``expandable_segments``) can never return that storage to the driver
        while a slot still references it. That reuse is desirable during
        training but wrong after a *full-parameter export*: a colocated consumer
        (e.g. a sleeping vLLM engine that must ``wake_up``) needs the
        full-model-sized storage handed back to the driver. Dropping the cached
        slots here makes the export release bounded -- matching FSDP2's
        ``full_tensor`` path, which retains no persistent buffer. The slots are
        transparently re-allocated on the next ``materialize``.

        Must only be called when no all-gather/reduce-scatter buffer is in
        flight (the allocator raises otherwise); the export path guarantees this
        because ``release_all`` has already drained every lease.
        """
        seen: set[int] = set()
        for bucket in self.buckets:
            allocator = bucket.allocator
            if id(allocator) in seen:
                continue
            seen.add(id(allocator))
            allocator.release_cached()

    def move_model_state(self, device: torch.device, *, load_grad: bool) -> None:
        device = torch.device(device)
        for bucket in self.buckets:
            bucket.move_model_state(device, load_grad=load_grad)
        self.all_gather.reset_device(device)
        self.grad_reduce.reset_device(device)

    def finish_grad_sync(self) -> None:
        self.grad_reduce.finish()

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

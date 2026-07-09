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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.mfsdp.allocator import (
    BufferLease,
    TemporaryBufferAllocator,
    build_temporary_allocator,
)
from megatron.lite.primitive.optimizers.mfsdp.config import MFSDPConfig
from megatron.lite.primitive.optimizers.mfsdp.mixed_precision import (
    MixedPrecisionPolicy,
)
from megatron.lite.primitive.optimizers.mfsdp.process_groups import (
    MFSDPProcessGroups,
    group_rank,
    group_size,
)


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
        self.local_grad_comm_buffer = torch.empty(
            self.local_numel,
            dtype=self.policy.grad_comm_dtype,
            device=self.device,
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
                key=("param", self.bucket_id),
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
            key=("grad", self.bucket_id),
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
            if (
                self.grad_sync_enabled
                and len(self._grad_ready_ids) == len(self.specs)
                and self.grad_ready_callback is not None
            ):
                self.grad_ready_callback(self)

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

        grouped: dict[tuple[int, bool, torch.dtype, torch.device], list[ParamSpec]] = (
            defaultdict(list)
        )
        owner_for_key: dict[tuple[int, bool, torch.dtype, torch.device], nn.Module] = {}
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
            for partition in _split_specs(grouped[key], self.config.bucket_size):
                bucket_id = len(buckets)
                buckets.append(
                    ParamBucket(
                        bucket_id,
                        partition,
                        process_group=self.groups.data_group(expert=expert),
                        gather_group=self.groups.gather_group(expert=expert),
                        config=self.config,
                        allocator=self.allocator,
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


__all__ = [
    "ParamAndGradBuffer",
    "ParamBucket",
    "ParamSpec",
    "SavedParamView",
]

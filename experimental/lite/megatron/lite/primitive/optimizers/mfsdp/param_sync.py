# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-FSDP parameter gather and gradient reduce-scatter hot path.

This module owns only the M-FSDP-specific lifecycle: model parameters are kept
as optimizer shards between compute regions, gathered just before their owning
module executes, and gradients are reduce-scattered as soon as a bucket is
complete. Optimizer math, mixed-precision shards, gradient norms, and state
movement are implemented by this standalone package.
"""

from __future__ import annotations

import importlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._pytree import tree_map_only


@dataclass(frozen=True, slots=True)
class _Binding:
    module: nn.Module
    attribute: str


@dataclass(slots=True)
class _ParamSpec:
    name: str
    full_param: nn.Parameter
    bindings: list[_Binding]
    shape: torch.Size
    numel: int
    shard_numel: int = 0
    padded_numel: int = 0
    local_offset: int = 0
    full_offset: int = 0
    shard_param: nn.Parameter | None = None


@dataclass(frozen=True, slots=True)
class _SavedParamView:
    bucket: "ParamBucket"
    size: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


class _BeginBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, pipeline: "ParamSyncPipeline"
    ) -> torch.Tensor:
        ctx.pipeline = pipeline
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        ctx.pipeline.begin_backward()
        return grad, None


class ParamBucket:
    """One flat M-FSDP communication bucket.

    A single all-gather covers every local parameter shard in the bucket. The
    rank-major collective output is unpacked into a parameter-major compute
    buffer. The same communication buffer is reused in reverse for gradient
    reduce-scatter.
    """

    def __init__(
        self,
        specs: list[_ParamSpec],
        *,
        process_group: dist.ProcessGroup | None,
        average_gradients: bool,
    ) -> None:
        if not specs:
            raise ValueError("An M-FSDP parameter bucket cannot be empty.")
        self.specs = specs
        self.process_group = process_group
        self.world_size = _group_size(process_group)
        self.rank = _group_rank(process_group)
        self.average_gradients = bool(average_gradients)
        self.dtype = specs[0].full_param.dtype
        self.shard_dtype = (
            torch.float32
            if specs[0].full_param.is_floating_point()
            and specs[0].full_param.dtype != torch.float64
            else self.dtype
        )
        self.grad_dtype = self.shard_dtype
        self.device = specs[0].full_param.device
        self.local_numel = 0
        self.full_numel = 0
        for spec in specs:
            spec.shard_numel = math.ceil(spec.numel / self.world_size)
            spec.padded_numel = spec.shard_numel * self.world_size
            spec.local_offset = self.local_numel
            spec.full_offset = self.full_numel
            self.local_numel += spec.shard_numel
            self.full_numel += spec.padded_numel

        self.full_buffer = torch.empty(
            self.full_numel, dtype=self.dtype, device=self.device
        )
        self.local_buffer = torch.empty(
            self.local_numel, dtype=self.dtype, device=self.device
        )
        self.comm_buffer = torch.empty(
            self.full_numel, dtype=self.dtype, device=self.device
        )
        self.grad_shard_buffer = torch.empty(
            self.local_numel, dtype=self.grad_dtype, device=self.device
        )
        self.grad_comm_buffer = torch.empty(
            self.full_numel, dtype=self.grad_dtype, device=self.device
        )
        self._param_gather_work: Any | None = None
        self._grad_reduce_work: Any | None = None
        self._grad_reduce_launched = False
        self.grad_sync_enabled = False
        self._full_ready = True
        self._grad_ready_ids: set[int] = set()
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        with torch.no_grad():
            self.full_buffer.zero_()
            for spec in self.specs:
                full_view = self.full_buffer.narrow(
                    0, spec.full_offset, spec.padded_numel
                )
                full_view[: spec.numel].copy_(spec.full_param.detach().reshape(-1))
                local_start = self.rank * spec.shard_numel
                local_shard = full_view.narrow(0, local_start, spec.shard_numel)
                shard_param = nn.Parameter(
                    local_shard.detach().to(self.shard_dtype).clone(),
                    requires_grad=spec.full_param.requires_grad,
                )
                _copy_parameter_metadata(spec.full_param, shard_param)
                shard_param._mfsdp_model_param_dtype = spec.full_param.dtype
                shard_param._mfsdp_original_ndim = spec.full_param.ndim
                spec.shard_param = shard_param
                spec.full_param.data = full_view[: spec.numel].view(spec.shape)
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
            for binding in spec.bindings:
                setattr(binding.module, binding.attribute, spec.full_param)

    def launch_param_gather(self) -> None:
        if self._full_ready or self._param_gather_work is not None:
            return
        _allocate_storage(self.comm_buffer, self.full_numel)
        with torch.no_grad():
            for spec in self.specs:
                assert spec.shard_param is not None
                self.local_buffer.narrow(0, spec.local_offset, spec.shard_numel).copy_(
                    spec.shard_param.detach().reshape(-1)
                )
        if self.world_size == 1:
            self.comm_buffer.copy_(self.local_buffer)
            return
        self._param_gather_work = dist.all_gather_into_tensor(
            self.comm_buffer,
            self.local_buffer,
            group=self.process_group,
            async_op=True,
        )

    def wait_param_gather(self) -> None:
        if self._full_ready:
            return
        if self._param_gather_work is None:
            self.launch_param_gather()
        if self._param_gather_work is not None:
            self._param_gather_work.wait()
            self._param_gather_work = None
        _allocate_storage(self.full_buffer, self.full_numel)
        with torch.no_grad():
            self.full_buffer.zero_()
            for spec in self.specs:
                destination = self.full_buffer.narrow(
                    0, spec.full_offset, spec.padded_numel
                )
                for rank in range(self.world_size):
                    source_offset = rank * self.local_numel + spec.local_offset
                    destination_offset = rank * spec.shard_numel
                    destination.narrow(0, destination_offset, spec.shard_numel).copy_(
                        self.comm_buffer.narrow(0, source_offset, spec.shard_numel)
                    )
                spec.full_param.data = destination[: spec.numel].view(spec.shape)
        _free_storage(self.comm_buffer)
        self._full_ready = True

    def release_full_parameters(self) -> None:
        self.install_sharded_parameters()
        if self._param_gather_work is not None:
            self._param_gather_work.wait()
            self._param_gather_work = None
        _free_storage(self.full_buffer)
        _free_storage(self.comm_buffer)
        self._full_ready = False

    def copy_full_parameters_to_shards(self) -> None:
        """Update this rank's optimizer shards after loading full model weights."""
        self.wait_param_gather()
        with torch.no_grad():
            for spec in self.specs:
                assert spec.shard_param is not None
                local_start = spec.full_offset + self.rank * spec.shard_numel
                spec.shard_param.copy_(
                    self.full_buffer.narrow(0, local_start, spec.shard_numel)
                )

    def saved_view(self, tensor: torch.Tensor) -> _SavedParamView | None:
        if (
            not self._full_ready
            or tensor.device != self.device
            or tensor.dtype != self.dtype
        ):
            return None
        if _storage_pointer(tensor) != _storage_pointer(self.full_buffer):
            return None
        return _SavedParamView(
            bucket=self,
            size=tuple(tensor.size()),
            stride=tuple(tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
        )

    def restore_saved_view(self, view: _SavedParamView) -> torch.Tensor:
        self.wait_param_gather()
        return torch.as_strided(
            self.full_buffer,
            size=view.size,
            stride=view.stride,
            storage_offset=view.storage_offset,
        )

    def _make_grad_ready_hook(self, spec: _ParamSpec) -> Callable[[nn.Parameter], None]:
        def grad_ready(param: nn.Parameter) -> None:
            if param.grad is None:
                return
            self._grad_ready_ids.add(id(spec))
            if self.grad_sync_enabled and len(self._grad_ready_ids) == len(self.specs):
                self.launch_grad_reduce()

        return grad_ready

    def set_grad_sync_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self.grad_sync_enabled:
            # Earlier microbatches may have populated the ready set while their
            # gradients accumulated locally. Only hooks from the final
            # microbatch may trigger the overlapping reduce-scatter.
            self._grad_ready_ids.clear()
        self.grad_sync_enabled = enabled

    def launch_grad_reduce(self, *, force: bool = False) -> None:
        if self._grad_reduce_launched:
            return
        if not force and len(self._grad_ready_ids) != len(self.specs):
            return
        _allocate_storage(self.grad_comm_buffer, self.full_numel)
        self._grad_reduce_launched = True
        self.grad_comm_buffer.zero_()
        with torch.no_grad():
            for spec in self.specs:
                grad = spec.full_param.grad
                if grad is not None:
                    flat_grad = grad.detach().reshape(-1)
                    for rank in range(self.world_size):
                        source_offset = rank * spec.shard_numel
                        count = min(
                            spec.shard_numel, max(0, spec.numel - source_offset)
                        )
                        if count <= 0:
                            continue
                        destination_offset = rank * self.local_numel + spec.local_offset
                        self.grad_comm_buffer.narrow(
                            0, destination_offset, count
                        ).copy_(flat_grad.narrow(0, source_offset, count))
                spec.full_param.grad = None
        if self.world_size == 1:
            self.grad_shard_buffer.copy_(self.grad_comm_buffer)
            return
        self._grad_reduce_work = dist.reduce_scatter_tensor(
            self.grad_shard_buffer,
            self.grad_comm_buffer,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=True,
        )

    def wait_grad_reduce(self) -> None:
        if not self._grad_reduce_launched:
            self.launch_grad_reduce(force=True)
        if self._grad_reduce_work is not None:
            self._grad_reduce_work.wait()
            self._grad_reduce_work = None
        if self.average_gradients and self.world_size > 1:
            self.grad_shard_buffer.div_(self.world_size)
        for spec in self.specs:
            assert spec.shard_param is not None
            grad = self.grad_shard_buffer.narrow(
                0, spec.local_offset, spec.shard_numel
            ).view_as(spec.shard_param)
            spec.shard_param.grad = grad
        _free_storage(self.grad_comm_buffer)

    def reset_grad_state(self) -> None:
        if self._grad_reduce_work is not None:
            self._grad_reduce_work.wait()
            self._grad_reduce_work = None
        self._grad_reduce_launched = False
        self.grad_sync_enabled = False
        self._grad_ready_ids.clear()
        for spec in self.specs:
            spec.full_param.grad = None


class ParamSyncPipeline:
    """Schedule bucket all-gathers around module compute in both directions."""

    def __init__(
        self, buckets: list[ParamBucket], owners: dict[int, list[int]]
    ) -> None:
        self.buckets = buckets
        self.owners = owners
        self._forward_cursor = 0
        self._backward_cursor = len(buckets) - 1

    def begin_forward(self) -> None:
        self.release_all()
        self._forward_cursor = 0
        if self.buckets:
            self.buckets[0].launch_param_gather()

    def acquire_forward(self, bucket_ids: Iterable[int]) -> None:
        for bucket_id in bucket_ids:
            bucket = self.buckets[bucket_id]
            bucket.install_full_parameters()
            self._forward_cursor = max(self._forward_cursor, bucket_id + 1)
            if self._forward_cursor < len(self.buckets):
                self.buckets[self._forward_cursor].launch_param_gather()

    def begin_backward(self) -> None:
        self._backward_cursor = len(self.buckets) - 1
        if self.buckets:
            self.buckets[self._backward_cursor].launch_param_gather()

    def acquire_backward(self, bucket: ParamBucket) -> None:
        bucket_id = self.buckets.index(bucket)
        bucket.wait_param_gather()
        self._backward_cursor = min(self._backward_cursor, bucket_id - 1)
        if self._backward_cursor >= 0:
            self.buckets[self._backward_cursor].launch_param_gather()

    def end_forward(self) -> None:
        self.release_all()

    def materialize_all(self) -> None:
        for bucket in self.buckets:
            bucket.launch_param_gather()
            bucket.install_full_parameters()

    def release_all(self) -> None:
        for bucket in self.buckets:
            bucket.release_full_parameters()

    def finish_grad_sync(self) -> None:
        for bucket in reversed(self.buckets):
            bucket.launch_grad_reduce(force=True)
        for bucket in self.buckets:
            bucket.wait_grad_reduce()

    def reset_grad_state(self) -> None:
        for bucket in self.buckets:
            bucket.reset_grad_state()

    def set_grad_sync_enabled(self, enabled: bool) -> None:
        for bucket in self.buckets:
            bucket.set_grad_sync_enabled(enabled)

    def copy_full_parameters_to_shards(self) -> None:
        for bucket in self.buckets:
            bucket.copy_full_parameters_to_shards()
        self.release_all()

    def pack_saved_tensor(self, tensor: torch.Tensor) -> torch.Tensor | _SavedParamView:
        for bucket in self.buckets:
            view = bucket.saved_view(tensor)
            if view is not None:
                return view
        return tensor

    def unpack_saved_tensor(
        self, value: torch.Tensor | _SavedParamView
    ) -> torch.Tensor:
        if isinstance(value, _SavedParamView):
            self.acquire_backward(value.bucket)
            return value.bucket.restore_saved_view(value)
        return value


class MFSdpModule(nn.Module):
    """Thin model wrapper implementing the M-FSDP shard/gather lifecycle."""

    def __init__(
        self,
        module: nn.Module,
        *,
        dense_process_group: dist.ProcessGroup | None,
        expert_process_group: dist.ProcessGroup | None,
        is_expert: Callable[[str], bool],
        unit_modules: Iterable[type[nn.Module] | str] | None,
        average_gradients: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self._expose_sharded_parameters = True
        buckets, owners = _build_buckets(
            module,
            dense_process_group=dense_process_group,
            expert_process_group=expert_process_group,
            is_expert=is_expert,
            unit_modules=unit_modules,
            average_gradients=average_gradients,
        )
        self.param_sync = ParamSyncPipeline(buckets, owners)
        for owner_id, bucket_ids in owners.items():
            owner = _module_by_id(module, owner_id)
            owner.register_forward_pre_hook(
                lambda _module, _args, ids=tuple(bucket_ids): (
                    self.param_sync.acquire_forward(ids)
                )
            )
        self.param_sync.release_all()

    def forward(self, *args, **kwargs):
        self.param_sync.begin_forward()
        try:
            with saved_tensors_hooks(
                self.param_sync.pack_saved_tensor,
                self.param_sync.unpack_saved_tensor,
            ):
                output = self.module(*args, **kwargs)
                output = tree_map_only(
                    torch.Tensor,
                    lambda tensor: (
                        _BeginBackward.apply(tensor, self.param_sync)
                        if tensor.requires_grad
                        else tensor
                    ),
                    output,
                )
        finally:
            self.param_sync.end_forward()
        return output

    def finish_grad_sync(self) -> None:
        self.param_sync.finish_grad_sync()

    def zero_grad_buffer(self) -> None:
        self.param_sync.reset_grad_state()

    def start_param_sync(self, *_args, force_sync: bool = False, **_kwargs) -> None:
        if force_sync:
            self.param_sync.materialize_all()
        elif self.param_sync.buckets:
            self.param_sync.buckets[0].launch_param_gather()

    def install_optimized_model_weights(self) -> None:
        self.param_sync.materialize_all()

    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, nn.Parameter]]:
        if not self._expose_sharded_parameters:
            self.param_sync.materialize_all()
        return super().named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        self.param_sync.materialize_all()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.param_sync.materialize_all()
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.param_sync.copy_full_parameters_to_shards()
        return result


def mark_optimizer_built(module: MFSdpModule) -> None:
    module._expose_sharded_parameters = False


def _build_buckets(
    module: nn.Module,
    *,
    dense_process_group: dist.ProcessGroup | None,
    expert_process_group: dist.ProcessGroup | None,
    is_expert: Callable[[str], bool],
    unit_modules: Iterable[type[nn.Module] | str] | None,
    average_gradients: bool,
) -> tuple[list[ParamBucket], dict[int, list[int]]]:
    module_by_name = dict(module.named_modules())
    module_order = {
        id(value): index for index, value in enumerate(module_by_name.values())
    }
    unit_types = _resolve_module_types(unit_modules or ())
    specs_by_id: dict[int, _ParamSpec] = {}
    owner_by_param_id: dict[int, nn.Module] = {}
    expert_by_param_id: dict[int, bool] = {}

    for name, param in module.named_parameters(remove_duplicate=False):
        if not param.requires_grad:
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = module_by_name[parent_name]
        spec = specs_by_id.get(id(param))
        if spec is None:
            spec = _ParamSpec(
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
        spec.bindings.append(_Binding(parent, attribute))

    grouped: dict[tuple[int, bool, torch.dtype, torch.device], list[_ParamSpec]] = (
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
        group = expert_process_group if expert else dense_process_group
        bucket_id = len(buckets)
        buckets.append(
            ParamBucket(
                grouped[key], process_group=group, average_gradients=average_gradients
            )
        )
        owners[id(owner)].append(bucket_id)
    return buckets, dict(owners)


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
        candidate_name = ".".join(parts[:end])
        candidate = module_by_name[candidate_name]
        if isinstance(candidate, unit_types):
            return candidate
    # Parameters outside an explicit FSDP unit belong to the root unit. Some
    # model paths (for example fused linear cross entropy) read a leaf module's
    # weight directly without invoking that leaf, so its pre-hook cannot be the
    # materialization boundary.
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


def _module_by_id(root: nn.Module, module_id: int) -> nn.Module:
    for module in root.modules():
        if id(module) == module_id:
            return module
    raise RuntimeError("M-FSDP bucket owner is no longer part of the wrapped module.")


def _copy_parameter_metadata(source: nn.Parameter, target: nn.Parameter) -> None:
    for name, value in source.__dict__.items():
        if name not in {"grad", "_grad"}:
            setattr(target, name, value)


def _group_size(group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def _group_rank(group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def _storage_pointer(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except RuntimeError:
        return 0


def _allocate_storage(tensor: torch.Tensor, numel: int) -> None:
    storage = tensor.untyped_storage()
    expected_bytes = numel * tensor.element_size()
    if storage.nbytes() != expected_bytes:
        storage.resize_(expected_bytes)


def _free_storage(tensor: torch.Tensor) -> None:
    storage = tensor.untyped_storage()
    if storage.nbytes() != 0:
        storage.resize_(0)


__all__ = ["MFSdpModule", "ParamBucket", "ParamSyncPipeline", "mark_optimizer_built"]

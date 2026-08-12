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
from torch.distributed import _coalescing_manager


@dataclass(slots=True)
class BufferLease:
    tensor: torch.Tensor
    owner: "TemporaryBufferAllocator"
    key: tuple[Any, ...]
    _released: bool = False

    def release(self) -> None:
        if not self._released:
            self.owner.release(self)
            self._released = True


class TemporaryBufferAllocator:
    """Allocate communication storage and release it at the pipeline boundary."""

    def allocate(
        self,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup | None,
        key: tuple[Any, ...],
        pool_key: tuple[Any, ...] | None = None,
    ) -> BufferLease:
        tensor = torch.empty(numel, dtype=dtype, device=device)
        return BufferLease(tensor=tensor, owner=self, key=key)

    def release(self, lease: BufferLease) -> None:
        _free_storage(lease.tensor)

    def release_cached(self, *, force: bool = False) -> None:
        """Drop allocator-owned communication storage before a device move."""


class StorageResizeBufferAllocator(TemporaryBufferAllocator):
    """Retain one tensor object per bucket and resize only its storage.

    This is MCore's non-double-buffer ``StorageResizeBasedBucketAllocator``
    contract.  Bucket identity, rather than layout, is the cache key: equal
    TransformerLayer layouts may overlap in the AG/RS pipelines and therefore
    must not alias the same tensor object.
    """

    def __init__(self) -> None:
        super().__init__()
        self._buckets: dict[tuple[Any, ...], torch.Tensor] = {}
        self._busy: set[tuple[Any, ...]] = set()

    def allocate(
        self,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup | None,
        key: tuple[Any, ...],
        pool_key: tuple[Any, ...] | None = None,
    ) -> BufferLease:
        del group, pool_key
        cache_key = (*key, dtype, device)
        if cache_key in self._busy:
            raise RuntimeError(
                f"M-FSDP storage-resize bucket {key!r} is already in use."
            )
        tensor = self._buckets.get(cache_key)
        if tensor is None:
            tensor = torch.empty(numel, dtype=dtype, device=device)
            self._buckets[cache_key] = tensor
        else:
            if tensor.numel() != numel:
                raise RuntimeError(
                    "M-FSDP storage-resize bucket changed size: "
                    f"key={key!r} cached={tensor.numel()} requested={numel}."
                )
            _allocate_storage(tensor)
        self._busy.add(cache_key)
        return BufferLease(tensor=tensor, owner=self, key=cache_key)

    def release(self, lease: BufferLease) -> None:
        if lease.key not in self._busy:
            raise RuntimeError(
                f"M-FSDP storage-resize bucket {lease.key!r} is not in use."
            )
        _free_storage(lease.tensor)
        self._busy.remove(lease.key)

    def release_cached(self, *, force: bool = False) -> None:
        if self._busy and not force:
            raise RuntimeError("Cannot release active M-FSDP communication buffers.")
        self._buckets.clear()
        self._busy.clear()


class MCoreBufferAllocators:
    """Route weights and gradients through MCore's distinct allocators."""

    def __init__(
        self,
        weight_allocator: TemporaryBufferAllocator,
        grad_allocator: TemporaryBufferAllocator,
    ) -> None:
        self.weight_allocator = weight_allocator
        self.grad_allocator = grad_allocator

    def allocate(
        self,
        numel: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup | None,
        key: tuple[Any, ...],
        pool_key: tuple[Any, ...] | None = None,
    ) -> BufferLease:
        allocator = (
            self.weight_allocator
            if key and key[0] in {"param", "param_local"}
            else self.grad_allocator
        )
        return allocator.allocate(
            numel,
            dtype=dtype,
            device=device,
            group=group,
            key=key,
            pool_key=pool_key,
        )

    def release_cached(self, *, force: bool = False) -> None:
        self.weight_allocator.release_cached(force=force)
        self.grad_allocator.release_cached(force=force)


def _free_storage(tensor: torch.Tensor) -> None:
    """Physically release completed temporary communication storage.

    Replacing a tensor view does not return its allocation while the lease still
    owns that view.  MCore releases temporary bucket storage after completion;
    mirror that lifecycle for temporary communication allocations.
    """
    if tensor.numel() == 0 or tensor.storage_offset() != 0:
        return
    storage = tensor.untyped_storage()
    if storage.nbytes():
        storage.resize_(0)


def _allocate_storage(tensor: torch.Tensor) -> None:
    """Restore storage for a retained, zero-storage bucket tensor."""
    if tensor.storage_offset() != 0:
        raise RuntimeError("M-FSDP can only resize sole-owner bucket storage.")
    storage = tensor.untyped_storage()
    expected_nbytes = tensor.numel() * tensor.element_size()
    if storage.nbytes() == expected_nbytes:
        return
    if storage.nbytes() != 0:
        raise RuntimeError(
            "M-FSDP bucket storage must be empty before resize: "
            f"actual={storage.nbytes()} expected={expected_nbytes}."
        )
    storage.resize_(expected_nbytes)


def build_temporary_allocator() -> MCoreBufferAllocators:
    return MCoreBufferAllocators(
        StorageResizeBufferAllocator(), TemporaryBufferAllocator()
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
        allocator: MCoreBufferAllocators,
        allocator_layout_key: tuple[Any, ...],
        is_fsdp_unit: bool = True,
        chunk_size_factor: int = 1,
        is_expert: bool = False,
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
        self.is_fsdp_unit = bool(is_fsdp_unit)
        self.is_expert = bool(is_expert)
        self.chunk_size_factor = max(int(chunk_size_factor), 1)
        self.requires_grad = all(spec.full_param.requires_grad for spec in specs)
        if any(spec.full_param.requires_grad != self.requires_grad for spec in specs):
            raise ValueError("M-FSDP bucket mixes trainable and frozen parameters.")
        self.device = specs[0].full_param.device
        self._training_compute_device = self.device
        self._training_param_offload = bool(
            config.full_optimizer_offload and self.device.type == "cuda"
        )
        compute_dtype = specs[0].full_param.dtype
        main_params_dtype = (
            compute_dtype
            if not self.requires_grad or config.full_optimizer_offload
            else config.main_params_dtype
        )
        self.policy = MixedPrecisionPolicy(
            compute_dtype=compute_dtype,
            main_params_dtype=main_params_dtype,
            main_grads_dtype=config.main_grads_dtype,
            grad_comm_dtype=config.grad_comm_dtype,
        )

        self.unpadded_numel = sum(spec.numel for spec in specs)
        layout_numel = _assign_mcore_bucket_offsets(specs, self.chunk_size_factor)
        self.logical_numel = layout_numel
        shard_alignment = self.world_size * self.chunk_size_factor
        self.full_numel = math.ceil(layout_numel / shard_alignment) * shard_alignment
        self.local_numel = self.full_numel // self.world_size
        local_begin = self.rank * self.local_numel
        local_end = local_begin + self.local_numel
        for spec in specs:
            spec_begin = spec.full_offset
            spec_end = spec.full_offset + spec.numel
            intersection_begin = max(spec_begin, local_begin)
            intersection_end = min(spec_end, local_end)
            spec.shard_numel = max(0, intersection_end - intersection_begin)
            spec.local_offset = min(
                self.local_numel, max(0, intersection_begin - local_begin)
            )
            spec.param_offset = min(spec.numel, max(0, intersection_begin - spec_begin))

        param_storage_device = (
            torch.device("cpu") if self._training_param_offload else self.device
        )
        self.main_param_buffer = torch.zeros(
            self.local_numel,
            dtype=self.policy.main_params_dtype,
            device=param_storage_device,
            pin_memory=self._training_param_offload,
        )
        # Match MCore's two persistent sharded weight buffers: optimizer state
        # updates the high-precision main shard, while parameter all-gathers
        # always consume the compute-precision model shard. Keeping this shard
        # resident avoids an FP32 -> BF16 allocation/copy on every forward and
        # backward gather; it is refreshed once after a successful optimizer
        # step instead.
        self.model_param_buffer = (
            self.main_param_buffer
            if self.policy.main_params_dtype == self.policy.compute_dtype
            else torch.zeros(
                self.local_numel,
                dtype=self.policy.compute_dtype,
                device=self.device,
            )
        )
        self.main_grad_buffer = torch.zeros(
            self.local_numel if self.requires_grad else 0,
            dtype=self.policy.main_grads_dtype,
            device=self.device,
        )
        self.grad_shard_buffer = self.main_grad_buffer
        # Compatibility alias for the all-gather pipeline and checkpoint tests.
        # Unlike the previous implementation this is persistent, just like
        # MCore's model_weight_buffer shard.
        self.local_compute_buffer = self.model_param_buffer
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
        self._local_compute_lease: BufferLease | None = None
        self._full_main_grad_lease: BufferLease | None = None
        self._grad_lease: BufferLease | None = None
        self._param_gather_work: Any | None = None
        self._param_gather_event: Any | None = None
        self._grad_reduce_work: Any | None = None
        self._grad_reduce_event: Any | None = None
        self._grad_reduce_launched = False
        self._grad_reduce_finished = False
        self._grad_accumulated_on_comm_stream = False
        self._microbatch_reduced = False
        self._has_accumulated_grad = False
        self._full_ready = True
        self.grad_sync_enabled = False
        self.grad_ready_callback: Callable[["ParamBucket"], None] | None = None
        self.before_main_grad_allocate: Callable[["ParamBucket"], None] | None = None
        self._grad_ready_ids: set[int] = set()
        self._initialize_parameters()
        if not self.is_fsdp_unit:
            self._initialize_persistent_full_parameters()

    def _initialize_persistent_full_parameters(self) -> None:
        """Keep MCore non-unit compute weights in persistent full storage."""
        persistent = torch.empty(
            self.full_numel, dtype=self.policy.compute_dtype, device=self.device
        )
        with torch.no_grad():
            persistent.zero_()
            for spec in self.specs:
                persistent.narrow(0, spec.full_offset, spec.numel).copy_(
                    spec.full_param.detach().reshape(-1)
                )
                spec.full_param.data = persistent.narrow(
                    0, spec.full_offset, spec.numel
                ).view(spec.shape)
        self.full_buffer = persistent
        self._full_ready = True

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
                    if self.model_param_buffer is not self.main_param_buffer:
                        self.model_param_buffer.narrow(
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
                # Optimizer-offload placement must be identical on every DP
                # rank and remain stable when a checkpoint is resharded.  A
                # local shard can be smaller (or empty) on boundary ranks, so
                # retain the logical parameter size for placement accounting.
                shard_param._mfsdp_global_numel = spec.numel
                shard_param._mfsdp_compute_device = self._training_compute_device
                spec.shard_param = shard_param
                # TE returns a dummy ``.grad`` when its wgrad GEMM writes the
                # real gradient directly into ``main_grad``.
                spec.full_param.grad_added_to_main_grad = False
                spec.full_param.__fsdp_param__ = True
                spec.full_param.overwrite_main_grad = True
                spec.full_param.get_main_grad = self._make_main_grad_getter(spec)
                if spec.full_param.requires_grad:
                    spec.full_param.register_post_accumulate_grad_hook(
                        self._make_grad_ready_hook(spec)
                    )

    def install_sharded_parameters(self, *, include_non_unit: bool = False) -> None:
        if not self.is_fsdp_unit and not include_non_unit:
            return
        for spec in self.specs:
            assert spec.shard_param is not None
            for binding in spec.bindings:
                setattr(binding.module, binding.attribute, spec.shard_param)

    def install_full_parameter_bindings(self) -> None:
        """Restore MCore's stable model-Parameter identity without gathering."""
        for spec in self.specs:
            for binding in spec.bindings:
                setattr(binding.module, binding.attribute, spec.full_param)

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
        self.install_full_parameter_bindings()

    def prepare_main_grads(self) -> None:
        """Attach bounded FP32 views for fused wgrad accumulation."""
        if not self.requires_grad:
            raise RuntimeError("Frozen M-FSDP buckets do not own gradient storage.")
        if self._full_main_grad_lease is None:
            self._enforce_main_grad_slot_limit()
            # MCore keeps the default-path gradient staging dynamically
            # allocated and releases it when reduce-scatter completes.
            self._full_main_grad_lease = self.allocator.allocate(
                self.full_numel,
                dtype=self.policy.main_grads_dtype,
                device=self.device,
                group=self.process_group,
                key=("main_grad", self.bucket_id),
                pool_key=(
                    "main_grad",
                    id(self.process_group),
                    self.allocator_layout_key,
                ),
            )
            self.full_main_grad_buffer = self._full_main_grad_lease.tensor
            self.full_main_grad_buffer.zero_()
        for spec in self.specs:
            self.get_main_grad(spec)

    def get_main_grad(self, spec: ParamSpec) -> torch.Tensor:
        """Lazily allocate this bucket's full-precision gradient staging view."""
        if not self.requires_grad:
            raise RuntimeError(
                f"Frozen M-FSDP parameter {spec.name!r} has no main_grad."
            )
        # A coalesced reduction from the current GraphTask may still own this
        # bucket's staging slot. Resolve that pressure at lazy allocation.
        self._enforce_main_grad_slot_limit()
        if self._full_main_grad_lease is None:
            self._full_main_grad_lease = self.allocator.allocate(
                self.full_numel,
                dtype=self.policy.main_grads_dtype,
                device=self.device,
                group=self.process_group,
                key=("main_grad", self.bucket_id),
                pool_key=(
                    "main_grad",
                    id(self.process_group),
                    self.allocator_layout_key,
                ),
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
        if self.is_fsdp_unit and self._full_lease is None:
            self._full_lease = self.allocator.allocate(
                self.full_numel,
                dtype=self.policy.compute_dtype,
                device=self.device,
                group=self.gather_group,
                key=("param", self.bucket_id),
                pool_key=(
                    "param",
                    id(self.gather_group),
                    self.allocator_layout_key,
                ),
            )
            self.full_buffer = self._full_lease.tensor
        if self._training_param_offload and self._local_compute_lease is None:
            self._local_compute_lease = self.allocator.allocate(
                self.local_numel,
                dtype=self.policy.compute_dtype,
                device=self.device,
                group=self.gather_group,
                key=("param_local", self.bucket_id),
                pool_key=(
                    "param_local",
                    id(self.gather_group),
                    self.allocator_layout_key,
                ),
            )
            self.local_compute_buffer = self._local_compute_lease.tensor
            self.local_compute_buffer.copy_(
                self.model_param_buffer, non_blocking=self.model_param_buffer.is_pinned()
            )
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
        if not self.is_fsdp_unit:
            self._release_local_compute_buffer()
            return
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
        if not self.is_fsdp_unit:
            return
        empty = torch.empty(0, dtype=self.policy.compute_dtype, device=self.device)
        with torch.no_grad():
            for spec in self.specs:
                spec.full_param.data = empty

    def invalidate_full_parameters(self) -> None:
        """Make the next forward refresh weights from updated optimizer shards."""
        if self.is_fsdp_unit:
            self.release_full_parameters()
        else:
            self._full_ready = False

    def _release_local_compute_buffer(self) -> None:
        if self._local_compute_lease is not None:
            self._local_compute_lease.release()
            self._local_compute_lease = None
        self.local_compute_buffer = self.model_param_buffer

    def _release_local_grad_comm_buffer(self) -> None:
        if self.policy.grad_comm_dtype != self.policy.main_grads_dtype:
            self.local_grad_comm_buffer = torch.empty(
                0, dtype=self.policy.grad_comm_dtype, device=self.device
            )
        else:
            self.local_grad_comm_buffer = self.main_grad_buffer

    def move_model_state(self, device: torch.device, *, load_grad: bool) -> None:
        """Move persistent sharded storage without breaking optimizer aliases."""
        device = torch.device(device)
        persistent_full = None if self.is_fsdp_unit else self.full_buffer
        self.release_full_parameters()
        self.discard_full_parameter_views()
        if self._grad_reduce_launched:
            self.wait_grad_reduce()
        else:
            self._release_full_main_grads()
        self.allocator.release_cached()

        grad_present = {
            id(spec): spec.shard_param is not None
            and (
                spec.shard_param.grad is not None
                or getattr(spec.shard_param, "main_grad", None) is not None
            )
            for spec in self.specs
        }
        compute_shares_main = self.model_param_buffer is self.main_param_buffer
        if not self._training_param_offload:
            self.main_param_buffer = self.main_param_buffer.to(device)
            self.model_param_buffer = (
                self.main_param_buffer
                if compute_shares_main
                else self.model_param_buffer.to(device)
            )
        elif self.main_param_buffer.device.type != "cpu":
            raise RuntimeError("M-FSDP training-offload parameter shard left CPU.")
        grad_comm_shares_main = self.local_grad_comm_buffer is self.main_grad_buffer
        grad_numel = self.local_numel if self.requires_grad else 0
        if load_grad:
            if self.main_grad_buffer.numel() == grad_numel:
                self.main_grad_buffer = self.main_grad_buffer.to(device)
            else:
                # Rollout offload deliberately drops gradients.  Re-entering
                # training creates a fresh zeroed shard instead of retaining a
                # useless 4 B/parameter CPU copy.
                self.main_grad_buffer = torch.zeros(
                    grad_numel,
                    dtype=self.policy.main_grads_dtype,
                    device=device,
                )
        else:
            self.main_grad_buffer = torch.empty(
                0,
                dtype=self.policy.main_grads_dtype,
                device=device,
            )
        self.grad_shard_buffer = self.main_grad_buffer
        self.local_compute_buffer = self.model_param_buffer
        self.local_grad_comm_buffer = (
            self.main_grad_buffer
            if grad_comm_shares_main
            else self.local_grad_comm_buffer.to(device)
        )
        self.device = device
        for spec in self.specs:
            if spec.shard_param is not None:
                spec.shard_param._mfsdp_compute_device = device
        self.full_buffer = (
            torch.empty(0, dtype=self.policy.compute_dtype, device=device)
            if persistent_full is None
            else persistent_full.to(device)
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
                if load_grad and grad_present[id(spec)] and spec.shard_numel:
                    main_grad = self.main_grad_buffer.narrow(
                        0, spec.local_offset, spec.shard_numel
                    ).view_as(spec.shard_param)
                    if (
                        spec.shard_param.dtype == main_grad.dtype
                        and not self.config.use_decoupled_grad
                    ):
                        spec.shard_param.grad = main_grad
                        if hasattr(spec.shard_param, "main_grad"):
                            delattr(spec.shard_param, "main_grad")
                    else:
                        spec.shard_param.grad = None
                        spec.shard_param.main_grad = main_grad
                else:
                    spec.shard_param.grad = None
                    spec.shard_param.main_grad = None
                spec.full_param.data = (
                    self.full_buffer
                    if self.is_fsdp_unit
                    else self.full_buffer.narrow(
                        0, spec.full_offset, spec.numel
                    ).view(spec.shape)
                )

    def release_scratch_keep_weights(self) -> None:
        """Drop the all-gather scratch while leaving the sharded weights resident.

        This is ``move_model_state``'s scratch-release prologue without the
        device move: release the full-parameter all-gather buffer, discard the
        full-parameter views, drain any in-flight grad reduce, and hand the
        allocator's cached communication storage back to the driver -- but keep
        ``main_param_buffer`` (and the optimizer-aliased shard views installed by
        ``release_full_parameters``) on their current device. A later bounded
        export can gather from those resident shards without moving persistent
        optimizer state.
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
        if not self.requires_grad:
            return None
        if self._grad_reduce_launched or self._microbatch_reduced:
            return None
        if not force and len(self._grad_ready_ids) != len(self.specs):
            return None
        self._grad_reduce_launched = True
        self._grad_accumulated_on_comm_stream = False
        self.prepare_main_grads()
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
                key=("grad", self.bucket_id),
                pool_key=(
                    "grad",
                    id(self.process_group),
                    self.allocator_layout_key,
                ),
            )
            grad_input = self._grad_lease.tensor
            grad_input.copy_(self.full_main_grad_buffer)
        # Match MCore's fully-sharded reduce-scatter layout: NCCL writes the
        # reduced local shard into its rank-local view of the unsharded
        # communication bucket.  The result is copied/accumulated into the
        # persistent sharded main-grad buffer after the collective. Reusing the
        # rank-local view avoids a redundant output allocation.
        local_begin = self.rank * self.local_numel
        self.local_grad_comm_buffer = grad_input.narrow(
            0, local_begin, self.local_numel
        )
        if self.local_grad_comm_buffer.numel() != self.local_numel:
            raise RuntimeError(
                "M-FSDP reduce-scatter output staging has the wrong size: "
                f"bucket={self.bucket_id} output={self.local_grad_comm_buffer.numel()} "
                f"expected={self.local_numel} accumulated={self._has_accumulated_grad}."
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
                    op=(
                        dist.ReduceOp.AVG
                        if self.config.average_gradients
                        else dist.ReduceOp.SUM
                    ),
                    group=self.process_group,
                    async_op=True,
                )
        if self._grad_reduce_event is not None:
            self._grad_reduce_event.synchronize()
            self._grad_reduce_event = None
        if self._grad_reduce_work is not None:
            self._grad_reduce_work.wait()
            self._grad_reduce_work = None
        if not self._grad_accumulated_on_comm_stream:
            # ``wait_grad_reduce`` can be used as the synchronous fallback when
            # no pipeline launched this bucket. Normal overlapped execution has
            # already enqueued this accumulation on the RS stream.
            self.main_grad_buffer.add_(self.local_grad_comm_buffer)
            self._grad_accumulated_on_comm_stream = True
        self._has_accumulated_grad = True
        for spec in self.specs:
            assert spec.shard_param is not None
            main_grad = self.main_grad_buffer.narrow(
                0, spec.local_offset, spec.shard_numel
            ).view_as(spec.shard_param)
            if main_grad.numel() == 0:
                # MCore leaves empty local shards without an optimizer gradient.
                # FusedAdam uses ``grad is None`` to exclude them from its
                # multi-tensor kernel; passing a zero-length tensor can trigger
                # an illegal memory access.
                spec.shard_param.grad = None
                spec.shard_param.main_grad = None
                continue
            if (
                spec.shard_param.dtype == main_grad.dtype
                and not self.config.use_decoupled_grad
            ):
                spec.shard_param.grad = main_grad
                if hasattr(spec.shard_param, "main_grad"):
                    delattr(spec.shard_param, "main_grad")
            else:
                # PyTorch requires ``Parameter.grad`` to have the parameter's
                # dtype. The standalone optimizer consumes this explicit view;
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
        self.copy_main_weights_to_model_weights()

    def copy_main_weights_to_model_weights(self) -> None:
        """Refresh MCore's persistent compute shard after an optimizer update."""
        if self.model_param_buffer is not self.main_param_buffer:
            self.model_param_buffer.copy_(self.main_param_buffer)

    def reset_grad_state(self) -> None:
        if self._grad_reduce_work is not None:
            self._grad_reduce_work.wait()
        self._grad_reduce_work = None
        self._grad_reduce_event = None
        self._grad_reduce_launched = False
        self._grad_reduce_finished = False
        self._grad_accumulated_on_comm_stream = False
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
                spec.shard_param.main_grad = None
        self._release_full_main_grads()

    def start_microbatch(self) -> None:
        self._microbatch_reduced = False
        self._grad_reduce_finished = False

    def set_grad_sync_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled and not self.grad_sync_enabled:
            self._grad_ready_ids.clear()
        self.grad_sync_enabled = enabled

    def _make_grad_ready_hook(self, spec: ParamSpec) -> Callable[[nn.Parameter], None]:
        @torch.compiler.disable
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
        self.allocator = build_temporary_allocator()
        self.buckets, self.owners, self.collective_groups = self._build(
            is_expert=is_expert, unit_modules=unit_modules or ()
        )
        self._bucket_id_by_param_id: dict[int, int] = {}
        for bucket in self.buckets:
            for spec in bucket.specs:
                self._bucket_id_by_param_id[id(spec.full_param)] = bucket.bucket_id
                if spec.shard_param is not None:
                    self._bucket_id_by_param_id[id(spec.shard_param)] = bucket.bucket_id

    @torch.no_grad()
    def scale_gradients(self, scaling_factor: float | torch.Tensor) -> None:
        """Scale the persistent sharded gradient data after reduction."""
        for bucket in self.buckets:
            bucket.main_grad_buffer.mul_(scaling_factor)

    def bucket_ids_for_module(
        self, module: nn.Module, *, recurse: bool
    ) -> tuple[int, ...]:
        """Return buckets touched by a module without changing bucket ownership."""
        return tuple(
            sorted(
                {
                    self._bucket_id_by_param_id[id(param)]
                    for param in module.parameters(recurse=recurse)
                    if id(param) in self._bucket_id_by_param_id
                }
            )
        )

    def _build(
        self,
        *,
        is_expert: Callable[[str], bool],
        unit_modules: Iterable[type[nn.Module] | str],
    ) -> tuple[list[ParamBucket], dict[int, list[int]], tuple[tuple[int, ...], ...]]:
        module_by_name = dict(self.module.named_modules())
        module_order = {
            id(value): index for index, value in enumerate(module_by_name.values())
        }
        unit_types = _resolve_module_types(unit_modules)
        specs_by_id: dict[int, ParamSpec] = {}
        owner_by_param_id: dict[int, nn.Module] = {}
        expert_by_param_id: dict[int, bool] = {}

        for name, param in self.module.named_parameters(remove_duplicate=False):
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
            tuple[int | None, bool, bool, torch.dtype, torch.device], list[ParamSpec]
        ] = {}
        unit_for_key: dict[
            tuple[int | None, bool, bool, torch.dtype, torch.device], nn.Module | None
        ] = {}
        for param_id, spec in specs_by_id.items():
            owner = owner_by_param_id[param_id]
            expert = expert_by_param_id[param_id]
            unit = owner if unit_types and isinstance(owner, unit_types) else None
            key = (
                id(unit) if unit is not None else None,
                expert,
                spec.full_param.requires_grad,
                spec.full_param.dtype,
                spec.full_param.device,
            )
            grouped.setdefault(key, []).append(spec)
            unit_for_key[key] = unit

        buckets: list[ParamBucket] = []
        owners: dict[int, list[int]] = defaultdict(list)
        collective_key_by_bucket: dict[int, tuple[int, bool] | None] = {}
        for key, grouped_specs in grouped.items():
            unit_id, expert, _requires_grad, _dtype, _device = key
            unit = unit_for_key[key]
            is_fsdp_unit = unit is not None
            coarse_partitions = _split_specs_by_bucket_policy(
                grouped_specs, None if is_fsdp_unit else self.config.bucket_size
            )
            partitions = [
                segmented
                for coarse in coarse_partitions
                for segmented in _split_specs_by_chunk_factor(coarse, expert=expert)
            ]
            if unit is None:
                owner_layout = "non-fsdp-unit"
            else:
                owner_type = type(unit)
                owner_layout = f"{owner_type.__module__}.{owner_type.__qualname__}"
            for partition_index, (partition, chunk_size_factor) in enumerate(
                partitions
            ):
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
                            chunk_size_factor,
                        ),
                        is_fsdp_unit=is_fsdp_unit,
                        chunk_size_factor=chunk_size_factor,
                        is_expert=expert,
                    )
                )
                hook_owner_ids = {
                    id(owner_by_param_id[id(spec.full_param)]) for spec in partition
                }
                for owner_id in sorted(hook_owner_ids, key=module_order.__getitem__):
                    owners[owner_id].append(bucket_id)
                collective_key_by_bucket[bucket_id] = (
                    (unit_id, expert) if unit_id is not None else None
                )

        unit_collectives: dict[tuple[int, bool], list[int]] = {}
        collective_group_list: list[tuple[int, ...]] = []
        for bucket in buckets:
            collective_key = collective_key_by_bucket[bucket.bucket_id]
            if collective_key is None:
                collective_group_list.append((bucket.bucket_id,))
            else:
                unit_collectives.setdefault(collective_key, []).append(bucket.bucket_id)
        collective_group_list.extend(
            tuple(group) for group in unit_collectives.values()
        )
        collective_group_list.sort(key=lambda group: group[0])
        collective_groups = tuple(collective_group_list)
        return buckets, dict(owners), collective_groups


def _split_specs_by_bucket_policy(
    specs: list[ParamSpec], bucket_size: int | None
) -> list[list[ParamSpec]]:
    """Apply MCore's shared-embedding and non-unit bucket boundaries."""
    result: list[list[ParamSpec]] = []
    current: list[ParamSpec] = []
    current_numel = 0
    for spec in specs:
        if bool(getattr(spec.full_param, "shared_embedding", False)):
            if current:
                result.append(current)
            result.append([spec])
            current = []
            current_numel = 0
            continue
        current.append(spec)
        current_numel += spec.numel
        if bucket_size is not None and current_numel >= bucket_size:
            result.append(current)
            current = []
            current_numel = 0
    if current:
        result.append(current)
    return result


def _split_specs_by_chunk_factor(
    specs: list[ParamSpec], *, expert: bool
) -> list[tuple[list[ParamSpec], int]]:
    """Port MCore's shape-factor communication segmentation."""
    remaining = sorted(specs, key=lambda spec: math.prod(spec.shape[1:]), reverse=True)
    result: list[tuple[list[ParamSpec], int]] = []
    while remaining:
        chunk_size_factor = max(math.prod(remaining[0].shape[1:]), 1)
        same_factor: list[ParamSpec] = []
        deferred: list[ParamSpec] = []
        for spec in remaining:
            param_factor = max(math.prod(spec.shape[1:]), 1)
            heterogeneous_grouped_expert = (
                expert
                and param_factor != chunk_size_factor
                and (
                    len(spec.shape) >= 3
                    or any(len(existing.shape) >= 3 for existing in same_factor)
                )
            )
            if heterogeneous_grouped_expert:
                deferred.append(spec)
                continue
            if (
                param_factor == chunk_size_factor
                or (
                    chunk_size_factor % param_factor == 0
                    and spec.numel % chunk_size_factor == 0
                )
                or spec.numel < chunk_size_factor
            ):
                same_factor.append(spec)
            else:
                chunk_size_factor = math.lcm(chunk_size_factor, param_factor)
                same_factor.append(spec)
        result.append((same_factor, chunk_size_factor))
        remaining = deferred
    return result


def _assign_mcore_bucket_offsets(specs: list[ParamSpec], chunk_size_factor: int) -> int:
    """Assign the exact packed item layout used by MCore's DP buffer index."""
    fragments = [spec for spec in specs if spec.numel < chunk_size_factor]
    regular = [spec for spec in specs if spec.numel >= chunk_size_factor]
    data_index = 0
    while regular:
        spec = regular.pop(0)
        spec.full_offset = data_index
        if spec.numel % chunk_size_factor == 0:
            data_index += spec.numel
            continue

        gap_offset = data_index + spec.numel
        data_index += (spec.numel // chunk_size_factor + 1) * chunk_size_factor
        remainder = spec.numel % chunk_size_factor
        space = chunk_size_factor - remainder
        paired: ParamSpec | None = None
        for candidate in regular:
            candidate_remainder = candidate.numel % chunk_size_factor
            if (
                candidate_remainder
                and remainder + candidate_remainder <= chunk_size_factor
            ):
                paired = candidate
                break
        if paired is not None:
            regular.remove(paired)
            paired_remainder = paired.numel % chunk_size_factor
            paired.full_offset = data_index - paired_remainder
            space -= paired_remainder
            data_index += paired.numel // chunk_size_factor * chunk_size_factor

        for fragment in tuple(fragments):
            if fragment.numel > space:
                continue
            fragment.full_offset = gap_offset
            gap_offset += fragment.numel
            space -= fragment.numel
            fragments.remove(fragment)

    for fragment in fragments:
        fragment.full_offset = data_index
        data_index += fragment.numel
    return data_index


def _parameter_owner(
    parent_name: str,
    parent: nn.Module,
    module_by_name: dict[str, nn.Module],
    unit_types: tuple[type[nn.Module], ...],
) -> nn.Module:
    if not unit_types:
        return parent
    parts = parent_name.split(".") if parent_name else []
    # MCore records units in module traversal order and assigns every parameter
    # to the first matching unit.  Since ``Module.modules()`` is pre-order, the
    # outermost matching unit owns a nested unit's parameters.
    for end in range(0, len(parts) + 1):
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
            getattr(buckets[0], "config", None),
            "suggested_communication_unit_size",
            None,
        )
    if configured is not None:
        return int(configured)

    groups = tuple(tuple(group) for group in (owner_bucket_ids or ()))
    unit_groups = tuple(
        group
        for group in groups
        if group and all(buckets[bucket_id].is_fsdp_unit for bucket_id in group)
    )
    if unit_groups:
        total_elements = sum(
            sum(buckets[bucket_id].unpadded_numel for bucket_id in group)
            for group in unit_groups
        )
        average_owner_elements = total_elements // len(unit_groups)
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
        fsdp_unit_bucket_ids: Iterable[Iterable[int]] | None = None,
        suggested_communication_unit_size: int | None = None,
    ) -> None:
        self.buckets = buckets
        self.overlap = bool(buckets and buckets[0].config.overlap_param_gather)
        owner_bucket_ids = tuple(tuple(group) for group in (owner_bucket_ids or ()))
        fsdp_unit_bucket_ids = tuple(
            tuple(group) for group in (fsdp_unit_bucket_ids or ())
        )
        capacity_groups = fsdp_unit_bucket_ids or owner_bucket_ids
        self.suggested_prefetch_elements = (
            _resolve_suggested_communication_unit_size(
                buckets, capacity_groups, explicit=suggested_communication_unit_size
            )
            // 2
        )
        groups = owner_bucket_ids or ((bucket.bucket_id,) for bucket in buckets)
        self._bucket_groups = sorted(
            (tuple(sorted(set(group))) for group in groups), key=lambda group: group[0]
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
        group_index = self._bucket_to_group[bucket_id]
        self._async_bucket_group_gather(self._bucket_groups[group_index], bwd=bwd)

    def _async_bucket_group_gather(
        self, bucket_ids: Iterable[int], *, bwd: bool
    ) -> None:
        pending = []
        for bucket_id in bucket_ids:
            bucket = self.buckets[bucket_id]
            if (
                bucket._full_ready
                or bucket._param_gather_work is not None
                or bucket._param_gather_event is not None
            ):
                continue
            output, local = bucket.prepare_param_gather()
            pending.append((bucket, output, local))
        if not pending:
            return

        process_group = pending[0][0].gather_group
        if any(bucket.gather_group is not process_group for bucket, _, _ in pending):
            raise RuntimeError("M-FSDP all-gather bucket group spans process groups.")

        def collective():
            if pending[0][0].world_size == 1:
                for _bucket, output, local in pending:
                    output.copy_(local)
                return None
            with _coalescing_manager(process_group, async_ops=True) as group_work:
                for _bucket, output, local in pending:
                    dist.all_gather_into_tensor(
                        output, local, group=process_group, async_op=True
                    )
            return group_work

        tensors = [
            tensor for _bucket, output, local in pending for tensor in (output, local)
        ]
        work = self.comm_stream.launch(collective, tensors)
        # MCore exposes the asynchronous coalescing handle itself as the
        # all-gather completion event.  A CUDA event recorded immediately after
        # leaving an ``async_ops=True`` coalescing manager can complete before
        # the grouped NCCL work, so it cannot replace ``group_work``.
        completion_event = self.comm_stream.record_event() if work is None else None
        for bucket, _output, _local in pending:
            bucket.mark_param_gather_launched(work, completion_event)

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
        prefetched = 0
        while 0 <= group_index < len(self._bucket_groups):
            if prefetched >= self.suggested_prefetch_elements:
                break
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
        capacity_owner_bucket_ids: Iterable[Iterable[int]] | None = None,
        suggested_communication_unit_size: int | None = None,
    ) -> None:
        self.buckets = buckets
        capacity_groups = tuple(
            tuple(
                bucket_id
                for bucket_id in sorted(set(group))
                if getattr(buckets[bucket_id], "is_fsdp_unit", False)
            )
            for group in (capacity_owner_bucket_ids or owner_bucket_ids or ())
        )
        capacity_groups = tuple(group for group in capacity_groups if group)
        self._bucket_ids = {
            id(bucket): getattr(bucket, "bucket_id", index)
            for index, bucket in enumerate(buckets)
        }
        groups = tuple(
            tuple(
                bucket_id
                for bucket_id in sorted(set(group))
                if getattr(buckets[bucket_id], "requires_grad", True)
            )
            for group in (owner_bucket_ids or ())
        )
        groups = tuple(group for group in groups if group)
        self._bucket_groups = tuple(
            sorted(groups, key=lambda group: group[0])
        ) or tuple(
            (self._bucket_ids[id(bucket)],)
            for bucket in buckets
            if getattr(bucket, "requires_grad", True)
        )
        self._bucket_to_group = {
            bucket_id: group for group in self._bucket_groups for bucket_id in group
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
            capacity_groups or owner_bucket_ids,
            explicit=suggested_communication_unit_size,
        )
        for bucket in buckets:
            bucket.grad_ready_callback = self.reduce_gradients
            bucket.before_main_grad_allocate = self._prepare_main_grad_allocate

    def _retire_oldest(self) -> None:
        completed, completed_elements = self._pending.pop(0)
        completed.wait_grad_reduce()
        self._pending_elements -= completed_elements

    def _prepare_main_grad_allocate(self, incoming: ParamBucket) -> None:
        # This standalone bucket object owns one staging lease. Retire an
        # earlier use in FIFO order before the same bucket is reused. The
        # normal root-post-backward path drains the remaining queue, matching
        # MCore GradReducePipeline.reset().
        retired_incoming = False
        while any(bucket is incoming for bucket, _elements in self._pending):
            retired_incoming = retired_incoming or self._pending[0][0] is incoming
            self._retire_oldest()
        if retired_incoming:
            # wait_grad_reduce() finalized the previous microbatch after this
            # microbatch's begin_backward() reset. Re-open the bucket for the
            # reduction that will be produced by the current backward.
            incoming.start_microbatch()

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
        self._reduce_bucket_group((self._bucket_ids[id(bucket)],), force=force)

    def _reduce_bucket_group(
        self, bucket_ids: Iterable[int], *, force: bool = False
    ) -> None:
        self._wait_for_previous_grad_reduce()
        pending = []
        for bucket_id in bucket_ids:
            bucket = self.buckets[bucket_id]
            tensors = bucket.prepare_grad_reduce(force=force)
            if tensors is not None:
                output, grad_input = tensors
                pending.append((bucket, output, grad_input))
        if not pending:
            return

        process_group = pending[0][0].process_group
        if any(bucket.process_group is not process_group for bucket, _, _ in pending):
            raise RuntimeError(
                "M-FSDP reduce-scatter bucket group spans process groups."
            )

        def collective():
            if pending[0][0].world_size == 1:
                for bucket, output, grad_input in pending:
                    output.copy_(grad_input)
                    bucket.main_grad_buffer.add_(output)
                    bucket._grad_accumulated_on_comm_stream = True
                return None
            # Match MCore's gradient pipeline: coalesce synchronous collective
            # calls on the dedicated RS stream, then use a CUDA event recorded
            # after the manager exits as the completion boundary.  Using
            # ``async_ops=True`` here and discarding its Work handle allowed the
            # event to fire before NCCL had populated the output shard.
            with _coalescing_manager(process_group):
                for _bucket, output, grad_input in pending:
                    dist.reduce_scatter_tensor(
                        output,
                        grad_input,
                        op=(
                            dist.ReduceOp.AVG
                            if _bucket.config.average_gradients
                            else dist.ReduceOp.SUM
                        ),
                        group=process_group,
                    )
            # MCore accumulates every reduced shard into its persistent local
            # main-grad buffer on the RS stream before recording completion.
            # Keeping this here preserves overlap and makes the event cover both
            # NCCL and the dtype-promoting accumulation kernel.
            for bucket, output, _grad_input in pending:
                bucket.main_grad_buffer.add_(output)
                bucket._grad_accumulated_on_comm_stream = True
            return None

        tensors = [
            tensor
            for _bucket, output, grad_input in pending
            for tensor in (output, grad_input)
        ]
        work = self.comm_stream.launch(collective, tensors)
        completion_event = self.comm_stream.record_event()
        for bucket, _output, grad_input in pending:
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
        self._reduce_bucket_group(group, force=force)

    def _wait_for_previous_grad_reduce(self) -> None:
        while (
            self._pending and self._pending_elements > self._pending_capacity_elements
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
            unfinished = tuple(
                bucket_id
                for bucket_id in group
                if not self.buckets[bucket_id]._microbatch_reduced
            )
            if unfinished:
                self._reduce_bucket_group(unfinished, force=True)
        self.comm_stream.wait_for_current()
        while self._pending:
            self._retire_oldest()

    def finish_microbatch(self) -> None:
        """Apply MCore's root-post-backward drain/reset contract.

        The persistent sharded ``main_grad_buffer`` accumulates across
        microbatches.  Only unsharded staging, readiness, and queued collective
        state belong to one GraphTask and are retired here.
        """
        self.finish()
        self._bucket_ready_microbatch.clear()
        self._group_launched_microbatch.clear()
        for bucket in self.buckets:
            if bucket._full_main_grad_lease is not None:
                raise RuntimeError(
                    "M-FSDP root callback left an unsharded gradient bucket live."
                )

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
        *,
        fsdp_unit_bucket_ids: Iterable[Iterable[int]] | None = None,
    ) -> None:
        self.buckets = buckets
        owner_bucket_ids = tuple(tuple(group) for group in (owner_bucket_ids or ()))
        explicit = (
            buckets[0].config.suggested_communication_unit_size if buckets else None
        )
        self.all_gather = AllGatherPipeline(
            buckets,
            owner_bucket_ids,
            fsdp_unit_bucket_ids=fsdp_unit_bucket_ids,
            suggested_communication_unit_size=explicit,
        )
        self.grad_reduce = GradReducePipeline(
            buckets,
            owner_bucket_ids,
            capacity_owner_bucket_ids=fsdp_unit_bucket_ids,
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
        try:
            self.grad_reduce.finish_microbatch()
        finally:
            self._backward_started = False

    def acquire_backward(self, bucket: ParamBucket) -> None:
        self.all_gather.acquire_backward(bucket)

    def acquire_backward_ids(self, bucket_ids: Iterable[int]) -> None:
        self.all_gather.acquire_backward_ids(bucket_ids)

    def release_backward_ids(self, bucket_ids: Iterable[int]) -> None:
        for bucket_id in bucket_ids:
            self.buckets[bucket_id].release_full_parameters()

    def release_forward_ids(self, bucket_ids: Iterable[int]) -> None:
        for bucket_id in bucket_ids:
            self.buckets[bucket_id].release_full_parameters()

    def end_forward(self) -> None:
        for bucket in self.buckets:
            bucket.release_full_parameters()

    def materialize_all(self) -> None:
        self.all_gather.materialize_all()

    def stream_materialize_buckets(self) -> Iterator["ParamBucket"]:
        """Materialize one bucket at a time, releasing each before the next.

        A full-parameter exporter only needs the current bucket. This generator
        therefore bounds transient materialization to gather → install → yield
        → release for one bucket at a time. Consumers MUST finish reading a
        bucket's parameters before requesting the next; plain iteration
        guarantees that ordering.
        """
        for bucket in self.buckets:
            self.all_gather.async_bucket_gather(bucket.bucket_id)
            self.all_gather.wait_bucket_ready(bucket.bucket_id)
            bucket.install_full_parameters()
            try:
                yield bucket
            finally:
                bucket.release_full_parameters()

    def release_all(self) -> None:
        self.all_gather.release_all()

    def invalidate_parameters(self) -> None:
        for bucket in self.buckets:
            bucket.invalidate_full_parameters()

    def copy_main_weights_to_model_weights(self) -> None:
        """Refresh persistent compute shards from optimizer-owned main shards."""
        for bucket in self.buckets:
            bucket.copy_main_weights_to_model_weights()

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
        device move: it hands retained communication storage back to the driver
        while leaving sharded weights and optimizer aliases in place as the
        source for later bounded materialization. See
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

__all__ = [
    "AllGatherPipeline",
    "BufferLease",
    "CommunicationPipelines",
    "CommunicationStream",
    "GradReducePipeline",
    "MCoreBufferAllocators",
    "ParamAndGradBuffer",
    "ParamBucket",
    "ParamSpec",
    "StorageResizeBufferAllocator",
    "TemporaryBufferAllocator",
]

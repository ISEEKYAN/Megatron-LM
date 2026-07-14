# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLite-owned Megatron-FSDP module wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._pytree import tree_map_only

from megatron.lite.primitive.optimizers.mfsdp.buffer import (
    AllGatherPipeline,
    CommunicationPipelines,
    GradReducePipeline,
    ParamAndGradBuffer,
)
from megatron.lite.primitive.optimizers.mfsdp.config import (
    MFSDPConfig,
    MFSDPProcessGroups,
)


class _BeginBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        pipeline: CommunicationPipelines,
    ) -> torch.Tensor:
        ctx.pipeline = pipeline
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        ctx.pipeline.begin_backward()
        return grad, None


class _AcquireBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        pipeline: CommunicationPipelines,
        bucket_ids: tuple[int, ...],
    ) -> torch.Tensor:
        ctx.pipeline = pipeline
        ctx.bucket_ids = bucket_ids
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        ctx.pipeline.acquire_backward_ids(ctx.bucket_ids)
        return grad, None, None


class _ReleaseBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        pipeline: CommunicationPipelines,
        bucket_ids: tuple[int, ...],
    ) -> torch.Tensor:
        ctx.pipeline = pipeline
        ctx.bucket_ids = bucket_ids
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        ctx.pipeline.release_backward_ids(ctx.bucket_ids)
        return grad, None, None


class MegatronFSDP(nn.Module):
    """Wrap a module with bucketed sharding and communication overlap."""

    def __init__(
        self,
        module: nn.Module,
        *,
        groups: MFSDPProcessGroups,
        config: MFSDPConfig,
        is_expert: Callable[[str], bool],
        unit_modules: Iterable[type[nn.Module] | str] | None,
    ) -> None:
        super().__init__()
        self.module = module
        self.mfsdp_config = config
        self._expose_sharded_parameters = True
        self.param_and_grad_buffer = ParamAndGradBuffer(
            module,
            groups=groups,
            config=config,
            is_expert=is_expert,
            unit_modules=unit_modules,
        )
        self.param_sync = CommunicationPipelines(self.param_and_grad_buffer.buckets)
        self.all_gather_pipeline: AllGatherPipeline = self.param_sync.all_gather
        self.grad_reduce_pipeline: GradReducePipeline = self.param_sync.grad_reduce
        for owner_id, bucket_ids in self.param_and_grad_buffer.owners.items():
            owner = _module_by_id(module, owner_id)
            ids = tuple(bucket_ids)

            def prepare_forward(_module, args, ids=ids):
                self.param_sync.acquire_forward(ids)
                return tree_map_only(
                    torch.Tensor,
                    lambda tensor: (
                        _ReleaseBackward.apply(tensor, self.param_sync, ids)
                        if tensor.requires_grad
                        else tensor
                    ),
                    args,
                )

            def finish_forward(_module, _args, output, ids=ids):
                output = tree_map_only(
                    torch.Tensor,
                    lambda tensor: (
                        _AcquireBackward.apply(tensor, self.param_sync, ids)
                        if tensor.requires_grad
                        else tensor
                    ),
                    output,
                )
                self.param_sync.release_forward_ids(ids)
                return output

            owner.register_forward_pre_hook(prepare_forward)
            owner.register_forward_hook(finish_forward)
        self.param_sync.release_all()
        self.param_sync.discard_full_parameter_views()

    def forward(self, *args, **kwargs):
        self.param_sync.begin_forward()

        def attach_backward(tensor: torch.Tensor) -> torch.Tensor:
            if not tensor.requires_grad:
                return tensor
            return _BeginBackward.apply(tensor, self.param_sync)

        try:
            with saved_tensors_hooks(
                self.param_sync.pack_saved_tensor,
                self.param_sync.unpack_saved_tensor,
            ):
                output = self.module(*args, **kwargs)
                output = tree_map_only(
                    torch.Tensor,
                    attach_backward,
                    output,
                )
        finally:
            self.param_sync.end_forward()
        return output

    def start_param_sync(self, *_args, force_sync: bool = False, **_kwargs) -> None:
        if force_sync:
            self.param_sync.materialize_all()
        elif (
            self.mfsdp_config.all_gather_in_start_param_sync and self.param_sync.buckets
        ):
            self.all_gather_pipeline.async_bucket_gather(0)

    def start_grad_sync(self, *_args) -> None:
        for bucket in reversed(self.param_sync.buckets):
            self.grad_reduce_pipeline.reduce_gradients(bucket, force=True)

    def finish_grad_sync(self, *_args) -> None:
        self.param_sync.finish_grad_sync()

    def zero_grad_buffer(self) -> None:
        self.param_sync.reset_grad_state()

    def move_model_state(
        self,
        device: torch.device | str,
        *,
        load_grad: bool = True,
    ) -> None:
        """Move M-FSDP-owned storage while preserving optimizer parameter aliases."""
        self.param_sync.move_model_state(
            torch.device(device),
            load_grad=load_grad,
        )

    def stream_full_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield ``(name, full_param)`` one bucket at a time for export.

        The bounded-bucket counterpart to ``materialize_all``: an exporter that
        walks parameters one at a time (see the shared HF exporter's
        ``export_hf_weights``) can consume this stream instead of pre-gathering
        the whole model, capping the transient full-parameter footprint at a
        single bucket rather than the whole unsharded model. This is the
        materialization side of the DS4 resync bounded-export protocol and
        mirrors FSDP2's per-parameter ``full_tensor`` gather, which shows no
        export memory peak.

        Each yielded parameter's ``.data`` is cloned off the bucket's gather
        buffer before the bucket is released, so a consumer that retains the
        tensor past the bucket's lifetime (e.g. the exporter buffers expert
        shards for a per-layer EP collective) keeps a private allocation and
        never aliases storage that the release has handed back to the allocator.
        """
        # Names are resolved against the currently-installed (sharded) params,
        # which is the module state on entry and after each bucket release.
        name_by_shard_id = {
            id(param): name for name, param in self.module.named_parameters()
        }
        covered: set[str] = set()
        for bucket in self.param_sync.stream_materialize_buckets():
            for spec in bucket.specs:
                name = name_by_shard_id.get(id(spec.shard_param))
                if name is None:
                    continue
                covered.add(name)
                full_param = spec.full_param
                full_param.data = full_param.data.clone()
                yield name, full_param
        # Parameters (and any params not owned by an M-FSDP bucket) that the
        # bucket walk did not cover -- emit them from the restored sharded view.
        for name, param in self.module.named_parameters():
            if name not in covered:
                yield name, param

    @contextmanager
    def full_parameter_context(self):
        """Scope a full-parameter export; parameters materialize on demand.

        Historically this all-gathered the whole model up front
        (``materialize_all``), producing a transient full-model peak. Export
        consumers now stream via ``stream_full_parameters`` (bounded to a single
        bucket), so this context no longer pre-materializes; it only guarantees
        the export all-gather storage is scoped to the export and handed back to
        the driver on exit. A consumer that still reads ``named_parameters``
        directly (rather than streaming) transparently falls back to the
        wrapper's own ``materialize_all`` there, so correctness is unchanged.
        """
        # The try/finally still wraps the whole body so any partial
        # materialization performed by the consumer routes through
        # release_cached_buffers below (which reclaims the persistent
        # double-buffer slots a colocated vLLM wake_up needs handed back).
        try:
            yield
        finally:
            self.param_sync.release_all()
            self.param_sync.discard_full_parameter_views()
            # Scope the export all-gather storage to this context. A
            # full-parameter export materializes the whole (unsharded) model;
            # under fsdp_double_buffer those all-gather buffers otherwise stay
            # pinned in the allocator's persistent slots after release_all, so
            # torch's caching allocator (and empty_cache / expandable_segments)
            # can never hand that storage back to the driver -- it lives across
            # the next colocated consumer's turn (e.g. a sleeping vLLM engine
            # waking for RL weight resync) and starves it. That persistent
            # cross-context retention is the resync-OOM bug; the export buffer's
            # lifetime must end with the export. Returning the cached slots to
            # the driver here makes the export release bounded like FSDP2's
            # full_tensor path (which retains no persistent buffer); the slots
            # are transparently re-allocated on the next export.
            self.param_sync.release_cached_buffers()

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


MFSdpModule = MegatronFSDP


def mark_optimizer_built(module: MegatronFSDP) -> None:
    module._expose_sharded_parameters = False


def _module_by_id(root: nn.Module, module_id: int) -> nn.Module:
    for module in root.modules():
        if id(module) == module_id:
            return module
    raise RuntimeError("M-FSDP bucket owner is no longer part of the wrapped module.")


__all__ = ["MFSdpModule", "MegatronFSDP", "mark_optimizer_built"]

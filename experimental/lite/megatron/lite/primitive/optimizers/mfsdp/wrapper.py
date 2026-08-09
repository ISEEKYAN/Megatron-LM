# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLite-owned Megatron-FSDP module wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._pytree import tree_map_only

from megatron.lite.primitive.optimizers.mfsdp.buffer import (  # isort: skip
    AllGatherPipeline,
    CommunicationPipelines,
    GradReducePipeline,
    ParamAndGradBuffer,
)
from megatron.lite.primitive.optimizers.mfsdp.config import (  # isort: skip
    MFSDPConfig,
    MFSDPProcessGroups,
)


class _BeginBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, tensor: torch.Tensor, pipeline: CommunicationPipelines
    ) -> torch.Tensor:
        ctx.pipeline = pipeline
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        if ctx.pipeline.begin_backward():
            torch.autograd.Variable._execution_engine.queue_callback(
                ctx.pipeline.end_backward
            )
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
        ctx, tensor: torch.Tensor, pipeline: CommunicationPipelines, owner_id: int
    ) -> torch.Tensor:
        ctx.pipeline = pipeline
        ctx.owner_id = owner_id
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        ctx.pipeline.release_post_backward_owner(ctx.owner_id)
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
        self.param_and_grad_buffer = ParamAndGradBuffer(
            module,
            groups=groups,
            config=config,
            is_expert=is_expert,
            unit_modules=unit_modules,
        )
        _enable_fused_wgrad_accumulation(module)
        self.param_sync = CommunicationPipelines(self.param_and_grad_buffer)
        self.all_gather_pipeline: AllGatherPipeline = self.param_sync.all_gather
        self.grad_reduce_pipeline: GradReducePipeline = self.param_sync.grad_reduce
        for owner_id, bucket_ids in self.param_and_grad_buffer.owners.items():
            owner = _module_by_id(module, owner_id)
            ids = tuple(bucket_ids)

            def prepare_forward(_module, args, ids=ids, owner_id=owner_id):
                self.param_sync.acquire_forward_owner(owner_id, ids)
                return tree_map_only(
                    torch.Tensor,
                    lambda tensor: (
                        _ReleaseBackward.apply(tensor, self.param_sync, owner_id)
                        if tensor.requires_grad
                        else tensor
                    ),
                    args,
                )

            def finish_forward(_module, _args, output, ids=ids, owner_id=owner_id):
                output = tree_map_only(
                    torch.Tensor,
                    lambda tensor: (
                        _AcquireBackward.apply(tensor, self.param_sync, ids)
                        if tensor.requires_grad
                        else tensor
                    ),
                    output,
                )
                self.param_sync.release_forward_owner(owner_id, ids)
                return output

            def finish_backward(_module, _grad_input, _grad_output, owner_id=owner_id):
                # Full backward hooks run after AccumulateGrad when module
                # inputs require grad.  For a graph root with grad-disabled
                # inputs they may run earlier; leave that owner for the queued
                # root callback unless at least one parameter hook has staged.
                if self.param_sync.owner_has_ready_gradients(owner_id):
                    self.param_sync.process_post_backward(owner_id)

            owner.register_forward_pre_hook(prepare_forward)
            owner.register_forward_hook(finish_forward)
            owner.register_full_backward_hook(finish_backward)
        self.param_sync.release_all()
        self.param_sync.discard_full_parameter_views()

    def forward(self, *args, **kwargs):
        self.param_sync.begin_forward()
        primary_failure = False

        def attach_backward(tensor: torch.Tensor) -> torch.Tensor:
            if not tensor.requires_grad:
                return tensor
            return _BeginBackward.apply(tensor, self.param_sync)

        try:
            with saved_tensors_hooks(
                self.param_sync.pack_saved_tensor, self.param_sync.unpack_saved_tensor
            ):
                output = self.module(*args, **kwargs)
                output = tree_map_only(torch.Tensor, attach_backward, output)
        except BaseException:
            primary_failure = True
            # A failed module forward can leave a registered double-buffer slot
            # busy.  Both cleanup phases are best-effort: neither may replace
            # the original module failure.
            try:
                self.param_sync.abort()
            except BaseException:
                pass
            raise
        finally:
            try:
                self.param_sync.end_forward()
            except BaseException:
                if not primary_failure:
                    raise
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
        self, device: torch.device | str, *, load_grad: bool = True
    ) -> None:
        """Move M-FSDP-owned storage while preserving optimizer parameter aliases."""
        self.param_sync.move_model_state(torch.device(device), load_grad=load_grad)

    def release_export_scratch(self) -> None:
        """Reclaim the training step's all-gather scratch before a colocated wake.

        verl's ``update_weights`` wakes the sleeping vLLM weight pool
        (``resume(['weights'])``) *before* exporting M-FSDP weights. At that
        point the trainer still pins the training step's transient all-gather
        scratch -- the persistent double-buffer slots plus any full-parameter
        views -- on top of the sharded weights and optimizer state, and vLLM's
        cumem ``create_and_map`` OOMs. Hand that scratch back to the driver
        (keeping the sharded weights resident as the export gather source) and
        drain the caching allocator so the wake has room.
        """
        self.param_sync.release_scratch_keep_weights()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        covered: set[str] = set()
        for bucket in self.param_sync.stream_materialize_buckets():
            for spec in bucket.specs:
                # ``ParamSpec.name`` is captured before the sharded view is
                # installed and remains the canonical module path.  Do not
                # recover a name through a shard object's identity: tied
                # bindings and wrapper view changes can make that lookup miss,
                # which would silently export a DP shard as though it were a
                # full parameter.
                covered.add(spec.name)
                full_param = spec.full_param
                # The bucket cleanup below restores the module's sharded view
                # and clears ``full_param.data``.  Return a separate Parameter
                # so an exporter that keeps a yielded tensor while advancing
                # to another bucket retains the full snapshot it was promised.
                yield spec.name, nn.Parameter(
                    full_param.detach().clone(), requires_grad=full_param.requires_grad
                )
        # Parameters (and any params not owned by an M-FSDP bucket) that the
        # bucket walk did not cover -- emit them from the restored sharded view.
        for name, param in self.module.named_parameters():
            if name not in covered:
                yield name, param

    def iter_persistent_shards(self):
        """Yield canonical names and persistent local shards for checkpointing."""
        for bucket in self.param_sync.buckets:
            for spec in bucket.specs:
                assert spec.shard_param is not None
                yield spec.name, spec.shard_param, bucket

    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, nn.Parameter]]:
        """Expose persistent optimizer shards without transient all-gathers."""
        return super().named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Return persistent shard state; full weights require explicit export."""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load persistent shard state without allocating full model weights."""
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


MFSdpModule = MegatronFSDP


def _module_by_id(root: nn.Module, module_id: int) -> nn.Module:
    for module in root.modules():
        if id(module) == module_id:
            return module
    raise RuntimeError("M-FSDP bucket owner is no longer part of the wrapped module.")


def _enable_fused_wgrad_accumulation(module: nn.Module) -> None:
    """Enable TE-style fused wgrad only inside the M-FSDP ownership boundary."""
    for child in module.modules():
        if hasattr(child, "fuse_wgrad_accumulation"):
            child.fuse_wgrad_accumulation = True


__all__ = ["MFSdpModule", "MegatronFSDP"]

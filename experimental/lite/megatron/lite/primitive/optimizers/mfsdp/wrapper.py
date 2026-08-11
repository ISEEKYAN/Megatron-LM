# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLite-owned Megatron-FSDP module wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn
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
from torch.utils._pytree import tree_map_only


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
        enable_fine_grained_param_gather_hook: bool = False,
        enable_fine_grained_param_gather_backward_hook: bool = False,
        fine_grained_recurse_module_types: Iterable[type[nn.Module]] | None = None,
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
        _configure_fused_wgrad_accumulation(
            module, enabled=config.gradient_accumulation_fusion
        )
        self.param_sync = CommunicationPipelines(
            self.param_and_grad_buffer.buckets,
            self.param_and_grad_buffer.collective_groups,
            fsdp_unit_bucket_ids=self.param_and_grad_buffer.owners.values(),
        )
        self.all_gather_pipeline: AllGatherPipeline = self.param_sync.all_gather
        self.grad_reduce_pipeline: GradReducePipeline = self.param_sync.grad_reduce
        fine_forward = bool(enable_fine_grained_param_gather_hook)
        fine_backward = bool(enable_fine_grained_param_gather_backward_hook)
        for owner_id, bucket_ids in self.param_and_grad_buffer.owners.items():
            owner = _module_by_id(module, owner_id)
            ids = tuple(bucket_ids)

            @torch.compiler.disable
            def prepare_forward(_module, args, kwargs, ids=ids):
                if not fine_forward:
                    self.param_sync.acquire_forward(ids)

                def attach_release(tensor: torch.Tensor) -> torch.Tensor:
                    if not tensor.requires_grad:
                        return tensor
                    return _ReleaseBackward.apply(tensor, self.param_sync, ids)

                return (
                    tree_map_only(torch.Tensor, attach_release, args),
                    tree_map_only(torch.Tensor, attach_release, kwargs),
                )

            @torch.compiler.disable
            def finish_forward(_module, _args, output, ids=ids):
                # MCore always installs the FSDP-unit pre-backward unshard hook,
                # including when fine-grained backward gather is enabled.  The
                # fine-grained hooks below are additional just-in-time markers;
                # they do not replace the enclosing unit boundary.
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

            owner.register_forward_pre_hook(prepare_forward, with_kwargs=True)
            owner.register_forward_hook(finish_forward)

        # Match MCore's fine-grained mode: every submodule gathers shallow
        # parameters at its actual compute entry point. Selected containers
        # recurse because their weights live on children while the container
        # itself is the side-stream entry point.
        # Bucket ownership and release remain at the outer FSDP unit.
        recurse_types = tuple(fine_grained_recurse_module_types or ())
        for fine_module in module.modules():
            if not fine_forward and not fine_backward:
                continue
            ids = self.param_and_grad_buffer.bucket_ids_for_module(
                fine_module,
                recurse=bool(recurse_types and isinstance(fine_module, recurse_types)),
            )
            if not ids:
                continue

            @torch.compiler.disable
            def prepare_fine_grained(_module, _args, _kwargs, ids=ids):
                self.param_sync.acquire_forward(ids)

            @torch.compiler.disable
            def prepare_fine_grained_backward(_module, _args, output, ids=ids):
                return tree_map_only(
                    torch.Tensor,
                    lambda tensor: (
                        _AcquireBackward.apply(tensor, self.param_sync, ids)
                        if tensor.requires_grad
                        else tensor
                    ),
                    output,
                )

            if fine_forward:
                fine_module.register_forward_pre_hook(
                    prepare_fine_grained, prepend=True, with_kwargs=True
                )
            if fine_backward:
                fine_module.register_forward_hook(prepare_fine_grained_backward)
        self.param_sync.release_all()

    def forward(self, *args, **kwargs):
        self.param_sync.begin_forward()
        primary_failure = False

        def attach_backward(tensor: torch.Tensor) -> torch.Tensor:
            if not tensor.requires_grad:
                return tensor
            return _BeginBackward.apply(tensor, self.param_sync)

        try:
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
        if force_sync or not self.mfsdp_config.overlap_param_gather:
            self.param_sync.materialize_all()
        elif (
            self.mfsdp_config.all_gather_in_start_param_sync
            and self.param_sync.buckets
        ):
            self.all_gather_pipeline.async_bucket_gather(0)

    def start_grad_sync(self, *_args) -> None:
        for bucket in reversed(self.param_sync.buckets):
            self.grad_reduce_pipeline.reduce_gradients(bucket, force=True)

    def finish_grad_sync(self, *_args) -> None:
        self.param_sync.finish_grad_sync()

    def scale_gradients(self, scaling_factor: float | torch.Tensor) -> None:
        """Scale all persistent sharded gradients, matching MCore M-FSDP."""
        self.param_and_grad_buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self) -> None:
        self.param_sync.reset_grad_state()

    def manual_buffer_registration(self) -> None:
        self.param_and_grad_buffer.manual_buffer_registration()

    def move_model_state(
        self, device: torch.device | str, *, load_grad: bool = True
    ) -> None:
        """Move M-FSDP-owned storage while preserving optimizer parameter aliases."""
        self.param_sync.move_model_state(torch.device(device), load_grad=load_grad)

    def release_export_scratch(self) -> None:
        """Reclaim all-gather scratch while retaining sharded export sources.

        This releases persistent double-buffer slots and full-parameter views,
        but keeps optimizer-owned parameter shards resident so a subsequent
        bounded export can gather from them. Draining the caching allocator
        makes the reclaimed capacity available to the next GPU consumer.
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
        single bucket rather than the whole unsharded model.

        Each yielded parameter's ``.data`` is cloned off the bucket's gather
        buffer before the bucket is released, so a consumer that retains the
        tensor past the bucket's lifetime (e.g. the exporter buffers expert
        shards for a per-layer EP collective) keeps a private allocation and
        never aliases storage that the release has handed back to the allocator.
        """
        covered: set[str] = set()
        bucket_stream = self.param_sync.stream_materialize_buckets()
        try:
            for bucket in bucket_stream:
                for spec in bucket.specs:
                    # ``ParamSpec.name`` is captured before the sharded view is
                    # installed and remains the canonical module path.  Do not
                    # recover a name through a shard object's identity: tied
                    # bindings and wrapper view changes can make that lookup
                    # miss, which would silently export a DP shard as though it
                    # were a full parameter.
                    covered.add(spec.name)
                    full_param = spec.full_param
                    # Return a separate Parameter so an exporter that keeps a
                    # yielded tensor while advancing to another bucket retains
                    # the full snapshot it was promised.
                    yield spec.name, nn.Parameter(
                        full_param.detach().clone(),
                        requires_grad=full_param.requires_grad,
                    )
        finally:
            # Closing an outer export generator must release the currently
            # borrowed MCore bucket even when the consumer stops early.
            bucket_stream.close()
        # Parameters (and any params not owned by an M-FSDP bucket) that the
        # bucket walk did not cover -- emit them from the restored sharded view.
        for name, param in self.module.named_parameters():
            if name not in covered:
                yield name, param

    def stream_borrowed_full_parameters(
        self,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield bucket-backed full parameters without making a device copy.

        Each yielded parameter is valid only until the iterator advances. This
        mirrors MCore's gather/use/release ownership rule and is intended for a
        consumer that transfers the value to owned storage before requesting
        the next item. Callers that may retain device tensors must use
        :meth:`stream_full_parameters` instead.
        """
        covered: set[str] = set()
        bucket_stream = self.param_sync.stream_materialize_buckets()
        try:
            for bucket in bucket_stream:
                for spec in bucket.specs:
                    covered.add(spec.name)
                    yield spec.name, spec.full_param
        finally:
            bucket_stream.close()
        for name, param in self.module.named_parameters():
            if name not in covered:
                yield name, param

    def named_optimizer_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield the persistent FP32 parameter shards owned by M-FSDP.

        MCore keeps non-FSDP-unit compute parameters materialized because they
        can be read across module boundaries.  Consequently the module's
        ``named_parameters()`` view is not an optimizer-state view: for those
        parameters it exposes the full compute tensor, not the persistent main
        parameter shard.  The standalone optimizer must consume the shard
        objects directly from the parameter buffer, just as MCore's optimizer
        consumes its main-weight-buffer views.
        """
        for bucket in self.param_and_grad_buffer.buckets:
            for spec in bucket.specs:
                if spec.shard_param is None:
                    raise RuntimeError(
                        f"M-FSDP parameter {spec.name!r} has no optimizer shard."
                    )
                yield spec.name, spec.shard_param

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Expose MCore's distributed optimizer parameters at idle boundaries."""
        del remove_duplicate
        if not recurse:
            return iter(())
        base = f"{prefix}.module" if prefix else "module"
        return (
            (f"{base}.{name}", param)
            for name, param in self.named_optimizer_parameters()
        )

    def _install_checkpoint_shards(self) -> None:
        self.param_sync.release_all()
        self.param_sync.discard_full_parameter_views()
        for bucket in self.param_and_grad_buffer.buckets:
            bucket.install_sharded_parameters(include_non_unit=True)

    def _restore_compute_parameter_bindings(self) -> None:
        for bucket in self.param_and_grad_buffer.buckets:
            if bucket.is_fsdp_unit:
                bucket.install_full_parameter_bindings()
            else:
                bucket.install_full_parameters()

    def state_dict(self, *args, **kwargs):
        self._install_checkpoint_shards()
        try:
            return super().state_dict(*args, **kwargs)
        finally:
            self._restore_compute_parameter_bindings()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._install_checkpoint_shards()
        try:
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
            self.param_sync.copy_main_weights_to_model_weights()
            self.param_sync.invalidate_parameters()
            return result
        finally:
            self._restore_compute_parameter_bindings()

    def sync_full_parameters_to_shards(self) -> None:
        """Commit an in-place model-state load into persistent M-FSDP shards."""
        self.param_sync.copy_full_parameters_to_shards()


MFSdpModule = MegatronFSDP


def _module_by_id(root: nn.Module, module_id: int) -> nn.Module:
    for module in root.modules():
        if id(module) == module_id:
            return module
    raise RuntimeError("M-FSDP bucket owner is no longer part of the wrapped module.")


def _configure_fused_wgrad_accumulation(
    module: nn.Module, *, enabled: bool
) -> None:
    """Apply the explicit MCore wgrad-fusion policy inside this FSDP boundary."""
    for child in module.modules():
        if hasattr(child, "fuse_wgrad_accumulation"):
            child.fuse_wgrad_accumulation = enabled


__all__ = ["MFSdpModule", "MegatronFSDP"]

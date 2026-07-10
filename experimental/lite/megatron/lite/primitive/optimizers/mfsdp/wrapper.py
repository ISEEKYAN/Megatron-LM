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
            owner.register_forward_pre_hook(
                lambda _module, _args, ids=tuple(bucket_ids): (
                    self.param_sync.acquire_forward(ids)
                )
            )
        self.param_sync.release_all()
        self.param_sync.discard_full_parameter_views()

    def forward(self, *args, **kwargs):
        self.param_sync.begin_forward()
        keep_full_parameters = False

        def attach_backward(tensor: torch.Tensor) -> torch.Tensor:
            nonlocal keep_full_parameters
            if not tensor.requires_grad:
                return tensor
            keep_full_parameters = True
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
            if not keep_full_parameters:
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

    @contextmanager
    def full_parameter_context(self):
        """Materialize full parameters for model-level export consumers."""
        self.param_sync.materialize_all()
        try:
            yield
        finally:
            self.param_sync.release_all()
            self.param_sync.discard_full_parameter_views()

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

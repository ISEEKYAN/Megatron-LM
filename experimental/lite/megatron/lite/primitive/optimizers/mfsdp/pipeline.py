# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bucket prefetch and gradient-reduction pipelines for M-FSDP."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.primitive.optimizers.mfsdp.buffer import (
    ParamBucket,
    SavedParamView,
)


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
        if not bucket._full_ready and bucket._param_gather_work is None:
            self.async_bucket_gather(bucket_id, bwd=bwd)
        self.comm_stream.wait_for_current()
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


class GradReducePipeline:
    """Launch ready reduce-scatter buckets on a second communication stream."""

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


class CommunicationPipelines:
    """Compatibility facade joining the independent AG and RS pipelines."""

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

    def end_forward(self) -> None:
        self.all_gather.release_all()

    def materialize_all(self) -> None:
        self.all_gather.materialize_all()

    def release_all(self) -> None:
        self.all_gather.release_all()

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
    "CommunicationPipelines",
    "CommunicationStream",
    "GradReducePipeline",
]

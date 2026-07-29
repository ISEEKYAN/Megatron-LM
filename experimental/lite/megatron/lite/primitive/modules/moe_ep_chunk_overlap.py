# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepEP MoE EP chunk-overlap primitive."""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    ep_chunk_ranges,
)
from megatron.lite.primitive.utils.cuda_allocator import record_workspace_shape


def _make_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    if not torch.cuda.is_available():
        raise RuntimeError("EP chunk overlap requires CUDA streams.")
    try:
        return torch.cuda.Stream(device=device)
    except TypeError:
        with torch.cuda.device(device):
            return torch.cuda.Stream()


_EP_CHUNK_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}
_EP_CHUNK_WGRAD_STREAMS: dict[int, torch.cuda.Stream] = {}


@contextmanager
def _ep_chunk_nvtx(phase: str, chunk_idx: int | None = None):
    if (
        os.environ.get("MEGATRON_LITE_EP_CHUNK_NVTX") != "1"
        or not torch.cuda.is_available()
    ):
        yield
        return
    suffix = "" if chunk_idx is None else f".chunk{chunk_idx}"
    torch.cuda.nvtx.range_push(f"chunked_ep.{phase}{suffix}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _cuda_device_index(device: torch.device | int | str) -> int:
    if isinstance(device, int):
        return device
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise RuntimeError("EP chunk overlap requires CUDA tensors.")
    return (
        torch.cuda.current_device() if cuda_device.index is None else cuda_device.index
    )


def _shared_comm_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    device_index = _cuda_device_index(device)
    stream = _EP_CHUNK_COMM_STREAMS.get(device_index)
    if stream is None:
        stream = _make_stream(device_index)
        _EP_CHUNK_COMM_STREAMS[device_index] = stream
    return stream


def _shared_wgrad_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    device_index = _cuda_device_index(device)
    stream = _EP_CHUNK_WGRAD_STREAMS.get(device_index)
    if stream is None:
        stream = _make_stream(device_index)
        _EP_CHUNK_WGRAD_STREAMS[device_index] = stream
    return stream


def _queue_backward_stream_wait(event: torch.cuda.Event, device: torch.device) -> None:
    """Make optimizer work queued after backward wait for deferred expert wgrad."""

    def wait_for_wgrad() -> None:
        with torch.cuda.device(device):
            torch.cuda.current_stream(device).wait_event(event)

    torch.autograd.Variable._execution_engine.queue_callback(wait_for_wgrad)


def _record_ep_chunk_shape(
    *,
    chunk_idx: int,
    phase: str,
    recv_rows: int,
    expert_rows: int,
    device: torch.device,
) -> None:
    record_workspace_shape(
        device_index=(
            _cuda_device_index(device) if torch.device(device).type == "cuda" else -1
        ),
        scope=f"ep_chunk_{phase}",
        slot=chunk_idx,
        dimensions={"recv_rows": recv_rows, "expert_rows": expert_rows},
    )


def _event_current_stream_wait(event: Any) -> None:
    if event is None:
        return
    if hasattr(event, "current_stream_wait"):
        event.current_stream_wait()
    else:
        torch.cuda.current_stream().wait_event(event)


@contextmanager
def _expert_act_recompute_disabled(experts: Experts):
    previous = experts.moe_act_recompute
    experts.moe_act_recompute = False
    try:
        yield
    finally:
        experts.moe_act_recompute = previous


@dataclass
class _BackwardChunk:
    idx: int
    start: int
    end: int
    x: torch.Tensor
    scores: torch.Tensor | None
    handle: Any
    row_id_map: torch.Tensor
    prob_flat_indices: torch.Tensor
    recv_hidden_scratch: torch.Tensor | None
    recv_probs_scratch: torch.Tensor | None
    dispatched: torch.Tensor
    probs: torch.Tensor | None
    expert_out: torch.Tensor | None
    scores_edge: Any | None = None
    scores_shape: torch.Size | None = None
    scores_dtype: torch.dtype | None = None
    expert_out_edge: Any | None = None
    expert_out_shape: torch.Size | None = None
    expert_out_dtype: torch.dtype | None = None


class EPChunkOverlapOperator:
    """Replaceable DeepEP chunk-overlap schedule over model-owned MoE modules."""

    def __init__(
        self,
        *,
        router: nn.Module,
        experts: Experts,
        dispatcher: TokenDispatcher,
        forward_dispatchers: tuple[TokenDispatcher, ...],
        router_forward: (
            Callable[
                [nn.Module, torch.Tensor, torch.Tensor | None],
                tuple[torch.Tensor, torch.Tensor],
            ]
            | None
        ) = None,
    ):
        self._router_forward = router_forward
        self._active_routing_input: torch.Tensor | None = None

        self.router = router
        self.experts = experts
        self.dispatcher = dispatcher
        self._forward_dispatchers = forward_dispatchers
        if (
            not self.dispatcher.use_deepep
            or len(self._forward_dispatchers) != 2
            or not all(item.use_deepep for item in self._forward_dispatchers)
        ):
            raise RuntimeError("EP chunk overlap requires available DeepEP transport.")

    def _streams(
        self, device: torch.device
    ) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        return torch.cuda.current_stream(device), _shared_comm_stream(device)

    @contextmanager
    def _routing_context(self, routing_input: torch.Tensor | None):
        previous = getattr(self, "_active_routing_input", None)
        self._active_routing_input = routing_input
        try:
            yield
        finally:
            self._active_routing_input = previous

    def _route(
        self, x: torch.Tensor, start: int = 0, end: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_forward = getattr(self, "_router_forward", None)
        if router_forward is None:
            return self.router(x)
        routing_input = self._active_routing_input
        if routing_input is not None:
            routing_input = routing_input.reshape(-1)[start:end]
        return router_forward(self.router, x, routing_input)

    def forward(
        self, x: torch.Tensor, routing_input: torch.Tensor | None = None
    ) -> torch.Tensor:
        with self._routing_context(routing_input):
            return self._forward_impl(x, routing_input)

    __call__ = forward

    def _forward_impl(
        self, x: torch.Tensor, routing_input: torch.Tensor | None
    ) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.view(-1, x.size(-1)) if x.dim() == 3 else x
        chunks = 2
        if torch.is_grad_enabled():
            params = tuple(self.router.parameters()) + tuple(self.experts.parameters())
            return _FullRecomputeFused.apply(
                x_2d, routing_input, self, input_shape, x.dtype, chunks, chunks, *params
            )
        ranges = ep_chunk_ranges(
            x_2d.size(0), chunks, weights_env="MEGATRON_LITE_EP_CHUNK_WEIGHTS"
        )
        return self._forward_output_async(
            x_2d, ranges, input_shape, x.dtype, disable_expert_act_recompute=False
        )

    def _forward_dispatcher(self, idx: int) -> TokenDispatcher:
        return self._forward_dispatchers[idx]

    def _forward_full(self, x_2d: torch.Tensor) -> torch.Tensor:
        scores, indices = self._route(x_2d, 0, x_2d.size(0))
        dispatched, local_tpe, probs = self.dispatcher.dispatch(x_2d, scores, indices)
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            local_tpe,
            probs,
            getattr(self.dispatcher, "_local_tpe_list", None),
        )
        return self.dispatcher.combine(expert_out)

    def _forward_full_skip_combine_autograd(self, x_2d: torch.Tensor) -> torch.Tensor:
        scores, indices = self._route(x_2d, 0, x_2d.size(0))
        dispatched, local_tpe, probs = self.dispatcher.dispatch(x_2d, scores, indices)
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            local_tpe,
            probs,
            getattr(self.dispatcher, "_local_tpe_list", None),
        )
        return self.dispatcher.combine_deepep_backward_only(
            expert_out, tuple(x_2d.shape)
        )

    def _forward_output_only(
        self,
        x_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        input_shape: torch.Size,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._forward_output_async(
            x_2d, ranges, input_shape, input_dtype, disable_expert_act_recompute=True
        )

    def _forward_output_async(
        self,
        x_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        input_shape: torch.Size,
        input_dtype: torch.dtype,
        *,
        disable_expert_act_recompute: bool,
    ) -> torch.Tensor:
        if not ranges:
            return x_2d.new_empty(input_shape).to(input_dtype)
        if len(ranges) == 1:
            expert_ctx = (
                _expert_act_recompute_disabled(self.experts)
                if disable_expert_act_recompute
                else nullcontext()
            )
            with torch.no_grad(), expert_ctx:
                return (
                    self._forward_full(x_2d).view(input_shape).to(input_dtype).detach()
                )

        compute_stream, comm_stream = self._streams(x_2d.device)
        caller_stream = torch.cuda.current_stream(x_2d.device)
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)

        def submit_dispatch(chunk_idx: int):
            start, end = ranges[chunk_idx]
            x_chunk = x_2d[start:end]
            dispatcher = self._forward_dispatcher(chunk_idx)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(input_ready)
                scores, indices = self._route(x_chunk, start, end)
                with _ep_chunk_nvtx("forward.dispatch", chunk_idx):
                    state = dispatcher.submit_deepep_dispatch(
                        x_chunk,
                        scores,
                        indices,
                        num_worst_tokens=x_chunk.size(0) * dispatcher.ep_size,
                    )
            return chunk_idx, dispatcher, state

        def finish_dispatch_expert(pending):
            chunk_idx, dispatcher, state = pending
            with torch.cuda.stream(compute_stream):
                with _ep_chunk_nvtx("forward.dispatch.finish", chunk_idx):
                    dispatched, tpe, probs = dispatcher.finish_deepep_dispatch(state)
                recv_hidden = state.get("recv_hidden")
                _record_ep_chunk_shape(
                    chunk_idx=chunk_idx,
                    phase="forward",
                    recv_rows=(
                        dispatched.size(0)
                        if recv_hidden is None
                        else recv_hidden.size(0)
                    ),
                    expert_rows=dispatched.size(0),
                    device=dispatched.device,
                )
                state.pop("recv_hidden", None)
                state.pop("recv_indices", None)
                state.pop("recv_probs", None)
                state.pop("recv_per_expert", None)
                expert_ctx = (
                    _expert_act_recompute_disabled(self.experts)
                    if disable_expert_act_recompute
                    else nullcontext()
                )
                with expert_ctx:
                    with _ep_chunk_nvtx("forward.expert", chunk_idx):
                        expert_out = self.experts(
                            dispatched,
                            tpe,
                            probs,
                            tokens_per_expert_list=getattr(
                                dispatcher, "_local_tpe_list", None
                            ),
                        )
                rank_grouped, handle = dispatcher.prepare_deepep_combine(expert_out)
                ready = torch.cuda.Event()
                ready.record(compute_stream)
            del dispatched, probs, expert_out
            return chunk_idx, dispatcher, rank_grouped, handle, ready

        def submit_combine(prepared):
            chunk_idx, dispatcher, rank_grouped, handle, ready = prepared
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ready)
                with _ep_chunk_nvtx("forward.combine", chunk_idx):
                    combine_state = dispatcher.submit_deepep_combine_prepared(
                        rank_grouped, handle
                    )
            return dispatcher, combine_state

        pending_combines = []
        with torch.no_grad():
            current_state = submit_dispatch(0)
            for loop_idx in range(len(ranges)):
                prepared = finish_dispatch_expert(current_state)
                if loop_idx + 1 < len(ranges):
                    current_state = submit_dispatch(loop_idx + 1)
                pending_combines.append(submit_combine(prepared))

        done = torch.cuda.Event()
        done.record(compute_stream)
        caller_stream.wait_event(done)
        output_2d = x_2d.new_empty(x_2d.shape)
        offset = 0
        for chunk_idx, (dispatcher, state) in enumerate(pending_combines):
            with _ep_chunk_nvtx("forward.combine.finish", chunk_idx):
                chunk_out = dispatcher.finish_deepep_combine(state)
            next_offset = offset + chunk_out.size(0)
            output_2d[offset:next_offset].copy_(chunk_out)
            offset = next_offset
            del chunk_out
        return output_2d.view(input_shape).to(input_dtype).detach()

    def _full_recompute_fused_backward(
        self, x_saved: torch.Tensor, grad_2d: torch.Tensor, num_chunks: int
    ):
        ranges = ep_chunk_ranges(
            x_saved.size(0),
            num_chunks,
            weights_env=(
                "MEGATRON_LITE_EP_CHUNK_BWD_WEIGHTS",
                "MEGATRON_LITE_EP_CHUNK_WEIGHTS",
            ),
        )
        router_params = tuple(self.router.parameters())
        expert_params = tuple(self.experts.parameters())
        if len(ranges) == 1:
            return self._full_recompute_skip_combine_backward(
                x_saved, grad_2d, router_params, expert_params
            )
        return self._full_recompute_fused_backward_v6(
            x_saved, grad_2d, ranges, router_params, expert_params
        )

    def _full_recompute_skip_combine_backward(
        self,
        x_2d: torch.Tensor,
        grad_2d: torch.Tensor,
        router_params: tuple[torch.Tensor, ...],
        expert_params: tuple[torch.Tensor, ...],
    ):
        x_recompute = x_2d.detach().requires_grad_(True)
        output = self._forward_full_skip_combine_autograd(x_recompute)
        torch.autograd.backward(output, grad_2d)
        grad_x = x_recompute.grad
        if grad_x is None:
            grad_x = torch.zeros_like(x_recompute)
        return (grad_x, [None for _ in router_params], [None for _ in expert_params])

    def _full_recompute_fused_backward_v6(
        self,
        x_2d: torch.Tensor,
        grad_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        router_params: tuple[torch.Tensor, ...],
        expert_params: tuple[torch.Tensor, ...],
    ):
        if not ranges:
            return (
                grad_2d.new_zeros(grad_2d.shape),
                [torch.zeros_like(param) for param in router_params],
                [torch.zeros_like(param) for param in expert_params],
            )

        compute_stream, comm_stream = self._streams(grad_2d.device)
        wgrad_stream = _shared_wgrad_stream(grad_2d.device)
        input_ready = torch.cuda.Event()
        input_ready.record(torch.cuda.current_stream(grad_2d.device))
        dispatcher = self.dispatcher
        grad_x_chunks: list[torch.Tensor | None] = [None for _ in ranges]
        router_accum: list[torch.Tensor | None] = [None for _ in router_params]
        pending_dispatch_bwd: list[tuple[_BackwardChunk, dict[str, Any]]] = []
        last_deepep_event: Any | None = None

        def chain_deepep_event() -> None:
            if last_deepep_event is not None:
                _event_current_stream_wait(last_deepep_event)

        def remember_deepep_event(state: dict[str, Any]):
            nonlocal last_deepep_event
            last_deepep_event = state.get("event")
            return state

        def submit_recompute_dispatch(chunk_idx: int):
            start, end = ranges[chunk_idx]
            x_chunk = x_2d[start:end].detach().requires_grad_(True)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(input_ready)
                scores, indices = self._route(x_chunk, start, end)
                router_ready = torch.cuda.Event()
                router_ready.record(compute_stream)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(router_ready)
                chain_deepep_event()
                with _ep_chunk_nvtx("backward.dispatch", chunk_idx):
                    state = remember_deepep_event(
                        dispatcher.submit_deepep_dispatch(
                            x_chunk,
                            scores,
                            indices,
                            num_worst_tokens=x_chunk.size(0) * dispatcher.ep_size,
                        )
                    )
            return chunk_idx, start, end, x_chunk, scores, state

        def submit_combine_bwd(chunk_idx: int, start: int, end: int, handle: Any):
            with torch.cuda.stream(comm_stream):
                grad_chunk = grad_2d[start:end].contiguous()
                chain_deepep_event()
                with _ep_chunk_nvtx("backward.combine", chunk_idx):
                    return remember_deepep_event(
                        dispatcher.submit_deepep_combine_backward(grad_chunk, handle)
                    )

        def finish_recompute_expert(chunk_idx: int, state: dict[str, Any]):
            with torch.cuda.stream(compute_stream):
                state["recv_hidden"] = (
                    state["recv_hidden"].detach().requires_grad_(True)
                )
                state["recv_probs"] = state["recv_probs"].detach().requires_grad_(True)
                dispatched, local_tpe, probs, metadata = (
                    dispatcher.finish_deepep_dispatch_external_with_options(
                        state, force_manual_map=True, force_direct_permute=True
                    )
                )
                with _expert_act_recompute_disabled(self.experts):
                    expert_input = dispatched.detach().requires_grad_(True)
                    expert_probs = (
                        None if probs is None else probs.detach().requires_grad_(True)
                    )
                    with _ep_chunk_nvtx("backward.expert", chunk_idx):
                        expert_out = self.experts(
                            expert_input,
                            local_tpe,
                            expert_probs,
                            tokens_per_expert_list=metadata["local_tpe_list"],
                        )
                _record_ep_chunk_shape(
                    chunk_idx=chunk_idx,
                    phase="backward",
                    recv_rows=state["recv_hidden"].size(0),
                    expert_rows=dispatched.size(0),
                    device=dispatched.device,
                )
            return (
                dispatched,
                local_tpe,
                probs,
                metadata,
                expert_input,
                expert_probs,
                expert_out,
            )

        with torch.enable_grad():
            next_state = submit_recompute_dispatch(len(ranges) - 1)
            for rev_idx in range(len(ranges) - 1, -1, -1):
                chunk_idx, start, end, x_chunk, scores, state = next_state
                combine_state = submit_combine_bwd(
                    chunk_idx, start, end, state["handle"]
                )
                (
                    dispatched,
                    local_tpe,
                    probs,
                    metadata,
                    expert_input,
                    expert_probs,
                    expert_out,
                ) = finish_recompute_expert(chunk_idx, state)

                row_id_map = metadata["manual_row_id_map"]
                prob_flat_indices = metadata["manual_prob_flat_indices"]
                if row_id_map is None or prob_flat_indices is None:
                    raise RuntimeError(
                        "EP chunk overlap fused backward requires manual dgrad metadata."
                    )

                scores_edge = None
                scores_ref: torch.Tensor | None = scores
                expert_out_edge = None
                expert_out_ref: torch.Tensor | None = expert_out
                if hasattr(torch.autograd.graph, "get_gradient_edge"):
                    scores_edge = torch.autograd.graph.get_gradient_edge(scores)
                    scores_ref = None
                    expert_out_edge = torch.autograd.graph.get_gradient_edge(expert_out)
                    expert_out_ref = None

                chunk = _BackwardChunk(
                    idx=chunk_idx,
                    start=start,
                    end=end,
                    x=x_chunk,
                    scores=scores_ref,
                    handle=state["handle"],
                    row_id_map=row_id_map.detach(),
                    prob_flat_indices=prob_flat_indices.detach(),
                    recv_hidden_scratch=state["recv_hidden"].detach(),
                    recv_probs_scratch=state["recv_probs"].detach(),
                    dispatched=expert_input,
                    probs=expert_probs,
                    expert_out=expert_out_ref,
                    scores_edge=scores_edge,
                    scores_shape=scores.shape,
                    scores_dtype=scores.dtype,
                    expert_out_edge=expert_out_edge,
                    expert_out_shape=expert_out.shape,
                    expert_out_dtype=expert_out.dtype,
                )
                state.pop("recv_hidden", None)
                state.pop("recv_indices", None)
                state.pop("recv_probs", None)
                del dispatched, probs, expert_out, scores, local_tpe

                local_state: dict[str, Any] = {}
                with torch.cuda.stream(compute_stream):
                    grad_rank_grouped = dispatcher.finish_deepep_combine_backward(
                        combine_state
                    )
                    combine_state.pop("grad_rank_grouped", None)
                    combine_state.pop("event", None)
                    local_state["grad_expert_out"] = _manual_unpermute_backward(
                        chunk, grad_rank_grouped
                    )
                    del grad_rank_grouped

                if rev_idx > 0:
                    next_state = submit_recompute_dispatch(rev_idx - 1)

                with torch.cuda.stream(compute_stream):
                    expert_output = (
                        chunk.expert_out_edge
                        if chunk.expert_out_edge is not None
                        else chunk.expert_out
                    )
                    if chunk.dispatched is None or expert_output is None:
                        raise RuntimeError(
                            "EP chunk overlap expert graph was released."
                        )
                    expert_inputs = _expert_grad_inputs(
                        chunk.dispatched, chunk.probs
                    )
                    expert_grads = torch.autograd.grad(
                        expert_output,
                        expert_inputs,
                        local_state["grad_expert_out"],
                        allow_unused=True,
                    )
                    grad_dispatched = expert_grads[0]
                    if grad_dispatched is None:
                        grad_dispatched = torch.zeros_like(chunk.dispatched)
                    if chunk.probs is None:
                        grad_probs = None
                    else:
                        grad_probs = expert_grads[1]
                        if grad_probs is None:
                            grad_probs = torch.zeros_like(chunk.probs)
                    chunk.dispatched = None
                    chunk.probs = None
                    chunk.expert_out = None
                    chunk.expert_out_edge = None
                    local_state.pop("grad_expert_out", None)
                    grad_recv_hidden, grad_recv_probs = _dispatch_local_backward(
                        chunk, grad_dispatched, grad_probs
                    )
                    local_state["grad_recv_hidden"] = grad_recv_hidden
                    local_state["grad_recv_probs"] = grad_recv_probs
                    local_bwd_ready = torch.cuda.Event()
                    local_bwd_ready.record(compute_stream)
                    del grad_dispatched, grad_probs

                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_event(local_bwd_ready)
                    chain_deepep_event()
                    with _ep_chunk_nvtx("backward.dispatch", chunk.idx):
                        local_state["dispatch_bwd_state"] = remember_deepep_event(
                            dispatcher.submit_deepep_dispatch_backward(
                                local_state["grad_recv_hidden"],
                                local_state["grad_recv_probs"],
                                chunk.handle,
                            )
                        )
                    _release_dispatch_scratch(chunk, comm_stream)
                    local_state.pop("grad_recv_hidden", None)
                    local_state.pop("grad_recv_probs", None)

                pending_dispatch_bwd.append((chunk, local_state))

        wgrad_ready = torch.cuda.Event()
        wgrad_ready.record(compute_stream)
        with torch.cuda.stream(wgrad_stream):
            wgrad_stream.wait_event(wgrad_ready)
            with _ep_chunk_nvtx("backward.wgrad"):
                self.experts.flush_delayed_weight_grads(
                    num_contexts=len(pending_dispatch_bwd)
                )
            wgrad_done = torch.cuda.Event()
            wgrad_done.record(wgrad_stream)
        _queue_backward_stream_wait(wgrad_done, grad_2d.device)

        for chunk, local_state in pending_dispatch_bwd:
            with torch.cuda.stream(compute_stream):
                grad_hidden, grad_scores = dispatcher.finish_deepep_dispatch_backward(
                    local_state["dispatch_bwd_state"]
                )
                if grad_scores is None:
                    if chunk.scores_shape is None or chunk.scores_dtype is None:
                        raise RuntimeError("Missing router score metadata.")
                    grad_scores = torch.zeros(
                        chunk.scores_shape,
                        device=grad_2d.device,
                        dtype=chunk.scores_dtype,
                    )
                router_output = (
                    chunk.scores_edge if chunk.scores_edge is not None else chunk.scores
                )
                if router_output is None:
                    raise RuntimeError("EP chunk overlap router graph was released.")
                router_grads = torch.autograd.grad(
                    router_output,
                    (chunk.x, *router_params),
                    grad_scores.to(chunk.scores_dtype),
                    allow_unused=True,
                )
                grad_score_x = router_grads[0]
                if grad_score_x is None:
                    grad_score_x = torch.zeros_like(chunk.x)
                grad_x_chunks[chunk.idx] = grad_hidden.to(chunk.x.dtype) + grad_score_x
                _accumulate(router_accum, router_params, router_grads[1:])
                chunk.scores = None
                chunk.scores_edge = None
                local_state.clear()

        done = torch.cuda.Event()
        done.record(compute_stream)
        torch.cuda.current_stream(grad_2d.device).wait_event(done)

        grad_x = torch.cat(
            [
                torch.zeros_like(x_2d[start:end]) if grad is None else grad
                for (start, end), grad in zip(ranges, grad_x_chunks, strict=True)
            ],
            dim=0,
        ).view_as(grad_2d)
        router_grads_out = _materialize(router_params, router_accum)
        return grad_x, router_grads_out, [None for _ in expert_params]


class _FullRecomputeFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_2d: torch.Tensor,
        routing_input: torch.Tensor | None,
        operator: EPChunkOverlapOperator,
        input_shape: torch.Size,
        input_dtype: torch.dtype,
        fwd_chunks: int,
        bwd_chunks: int,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        del params, input_dtype
        ctx.operator = operator
        ctx.bwd_chunks = bwd_chunks
        ctx.num_router_params = len(tuple(operator.router.parameters()))
        ctx.has_routing_input = routing_input is not None
        saved_routing = (
            routing_input.detach()
            if routing_input is not None
            else x_2d.new_empty(0, dtype=torch.long)
        )
        ctx.save_for_backward(x_2d.detach(), saved_routing)
        ranges = ep_chunk_ranges(
            x_2d.size(0), fwd_chunks, weights_env="MEGATRON_LITE_EP_CHUNK_WEIGHTS"
        )
        output = operator._forward_output_only(x_2d, ranges, input_shape, x_2d.dtype)
        return output.detach().view(input_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        operator = ctx.operator
        x_saved, routing_saved = ctx.saved_tensors
        grad_2d = grad_output.contiguous().view(-1, grad_output.size(-1))
        routing_input = routing_saved if ctx.has_routing_input else None
        with operator._routing_context(routing_input), torch.enable_grad():
            grad_x, router_grads, expert_grads = operator._full_recompute_fused_backward(
                x_saved, grad_2d, ctx.bwd_chunks
            )
        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,
            None,
            *router_grads,
            *expert_grads,
        )


def _manual_unpermute_backward(
    chunk: _BackwardChunk, grad_rank_grouped: torch.Tensor
) -> torch.Tensor:
    if chunk.expert_out_shape is None or chunk.expert_out_dtype is None:
        raise RuntimeError("Missing expert output metadata.")
    return (
        grad_rank_grouped.index_select(0, chunk.row_id_map.reshape(-1).to(torch.long))
        .contiguous()
        .view(chunk.expert_out_shape)
        .to(chunk.expert_out_dtype)
    )


def _dispatch_local_backward(
    chunk: _BackwardChunk,
    grad_dispatched: torch.Tensor,
    grad_probs: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_id_map = chunk.row_id_map.reshape(-1).to(torch.long)
    grad_recv_hidden = chunk.recv_hidden_scratch
    if grad_recv_hidden is None:
        raise RuntimeError("EP chunk overlap recv-hidden scratch was released early.")
    grad_recv_hidden.zero_()
    grad_recv_hidden.scatter_add_(
        0,
        row_id_map.unsqueeze(1).expand(-1, grad_dispatched.size(1)),
        grad_dispatched.to(grad_recv_hidden.dtype),
    )
    grad_recv_probs = chunk.recv_probs_scratch
    if grad_recv_probs is None:
        raise RuntimeError("EP chunk overlap recv-probs scratch was released early.")
    grad_recv_probs.zero_()
    if grad_probs is not None:
        flat = chunk.prob_flat_indices.reshape(-1).to(grad_probs.device, torch.long)
        grad_recv_probs.reshape(-1).index_copy_(
            0, flat, grad_probs.reshape(-1).to(grad_recv_probs.dtype)
        )
    return grad_recv_hidden, grad_recv_probs


def _release_dispatch_scratch(
    chunk: _BackwardChunk, stream: torch.cuda.Stream
) -> None:
    for name in ("recv_hidden_scratch", "recv_probs_scratch"):
        tensor = getattr(chunk, name)
        if tensor is not None:
            tensor.record_stream(stream)
            setattr(chunk, name, None)


def _expert_grad_inputs(
    dispatched: torch.Tensor, probs: torch.Tensor | None
) -> tuple[torch.Tensor, ...]:
    return (dispatched,) if probs is None else (dispatched, probs)


def _accumulate(
    accum: list[torch.Tensor | None],
    params: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
) -> None:
    for idx, (param, grad) in enumerate(zip(params, grads, strict=True)):
        if grad is None:
            continue
        grad = grad.to(param.dtype)
        if accum[idx] is None:
            accum[idx] = grad
        else:
            accum[idx].add_(grad)


def _materialize(
    params: tuple[torch.Tensor, ...], accum: list[torch.Tensor | None]
) -> list[torch.Tensor]:
    return [
        torch.zeros_like(param) if grad is None else grad
        for param, grad in zip(params, accum, strict=True)
    ]


__all__ = ["EPChunkOverlapOperator"]

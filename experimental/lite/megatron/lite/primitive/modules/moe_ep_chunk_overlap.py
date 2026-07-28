# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepEP MoE EP chunk-overlap primitive."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    ChunkSpec,
    ep_chunk_ranges,
    parse_ep_chunk_spec,
    resolve_ep_chunk_overlap_chunks,
)
from megatron.lite.primitive.modules.router import TopKRouter
from megatron.lite.primitive.parallel import ParallelState


def _make_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    if not torch.cuda.is_available():
        raise RuntimeError("EP chunk overlap requires CUDA streams.")
    try:
        return torch.cuda.Stream(device=device)
    except TypeError:
        with torch.cuda.device(device):
            return torch.cuda.Stream()


_EP_CHUNK_STREAMS: dict[int, tuple[torch.cuda.Stream, torch.cuda.Stream]] = {}


def _cuda_device_index(device: torch.device | int | str) -> int:
    if isinstance(device, int):
        return device
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise RuntimeError("EP chunk overlap requires CUDA tensors.")
    return (
        torch.cuda.current_device() if cuda_device.index is None else cuda_device.index
    )


def _shared_streams(
    device: torch.device | int | str,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    device_index = _cuda_device_index(device)
    streams = _EP_CHUNK_STREAMS.get(device_index)
    if streams is None:
        streams = (_make_stream(device_index), _make_stream(device_index))
        _EP_CHUNK_STREAMS[device_index] = streams
    return streams


def _max_deepep_chunks(chunk_spec: ChunkSpec) -> int:
    return int(chunk_spec)


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
    recv_hidden_shape: torch.Size
    recv_hidden_dtype: torch.dtype
    recv_probs_shape: torch.Size
    recv_probs_dtype: torch.dtype
    dispatched: torch.Tensor
    probs: torch.Tensor | None
    expert_out: torch.Tensor | None
    scores_edge: Any | None = None
    scores_shape: torch.Size | None = None
    scores_dtype: torch.dtype | None = None
    expert_out_edge: Any | None = None
    expert_out_shape: torch.Size | None = None
    expert_out_dtype: torch.dtype | None = None


class EPChunkOverlapMoELayer(nn.Module):
    """Product DeepEP chunk-overlap MoE layer."""

    def __init__(
        self,
        config: Any,
        ps: ParallelState,
        *,
        num_chunks_ep_a2a_overlap: ChunkSpec = 1,
        use_deepep: bool = True,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        moe_full_recompute: bool = True,
        lora_config: Any | None = None,
        layer_idx: int | None = None,
        router: nn.Module | None = None,
        experts: Experts | None = None,
        dispatcher_factory: (
            Callable[[tuple[str, str, int, int]], TokenDispatcher] | None
        ) = None,
        router_forward: (
            Callable[
                [nn.Module, torch.Tensor, torch.Tensor | None],
                tuple[torch.Tensor, torch.Tensor],
            ]
            | None
        ) = None,
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        self.chunk_spec = parse_ep_chunk_spec(num_chunks_ep_a2a_overlap)
        if self.chunk_spec == 2 and not use_deepep:
            raise ValueError("EP chunk overlap requires use_deepep=True.")
        self.use_deepep = use_deepep
        self.moe_full_recompute = moe_full_recompute
        self.layer_idx = layer_idx
        self._max_chunks = _max_deepep_chunks(self.chunk_spec)
        self._deepep_layer_buffer_slots = 8
        self._dispatcher_factory = dispatcher_factory
        self._router_forward = router_forward
        self._active_routing_input: torch.Tensor | None = None

        self.router = (
            router
            if router is not None
            else TopKRouter(
                config, ps, router_bias_rate=router_bias_rate, compute_aux_loss=False
            )
        )
        self.experts = (
            experts
            if experts is not None
            else Experts(
                config,
                ps,
                fp8=fp8,
                moe_act_recompute=moe_act_recompute,
                lora_config=lora_config,
            )
        )
        self.dispatcher = self._new_dispatcher("main", 0)
        if self.chunk_spec == 2 and not self.dispatcher.use_deepep:
            raise RuntimeError("EP chunk overlap requires available DeepEP transport.")

        self._forward_dispatchers: list[TokenDispatcher] = []

    def _deepep_buffer_slot_for(self, role: str, idx: int) -> tuple[str, str, int, int]:
        layer_slot = id(self)
        if self.layer_idx is not None:
            layer_slot = int(self.layer_idx) % self._deepep_layer_buffer_slots
        return ("ep_chunk_overlap", role, layer_slot, idx % self._max_chunks)

    def _new_dispatcher(self, role: str, idx: int) -> TokenDispatcher:
        slot = self._deepep_buffer_slot_for(role, idx)
        if self._dispatcher_factory is not None:
            return self._dispatcher_factory(slot)
        return TokenDispatcher(
            self.config.num_experts,
            self.config.hidden_size,
            self.ps,
            use_deepep=self.use_deepep,
            buffer_slot=slot,
        )

    def _streams(
        self, device: torch.device
    ) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        return _shared_streams(device)

    def _num_chunks(self, num_tokens: int) -> int:
        return resolve_ep_chunk_overlap_chunks(
            num_tokens,
            ep_size=self.dispatcher.ep_size,
            hidden_size=self.config.hidden_size,
            spec=self.chunk_spec,
        )

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

    def _forward_impl(
        self, x: torch.Tensor, routing_input: torch.Tensor | None
    ) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.view(-1, x.size(-1)) if x.dim() == 3 else x
        chunks = self._num_chunks(x_2d.size(0))
        if chunks > 1 and torch.is_grad_enabled() and self.moe_full_recompute:
            params = tuple(self.router.parameters()) + tuple(self.experts.parameters())
            return _FullRecomputeFused.apply(
                x_2d, routing_input, self, input_shape, x.dtype, chunks, chunks, *params
            )
        if torch.is_grad_enabled() and chunks > 1:
            raise RuntimeError(
                "EP chunk-overlap training requires moe_full_recompute=True."
            )
        if chunks <= 1:
            return self._forward_full(x_2d).view(input_shape)
        ranges = ep_chunk_ranges(
            x_2d.size(0), chunks, weights_env="MEGATRON_LITE_EP_CHUNK_WEIGHTS"
        )
        return self._forward_output_async(
            x_2d, ranges, input_shape, x.dtype, disable_expert_act_recompute=False
        )

    def _forward_dispatcher(self, idx: int) -> TokenDispatcher:
        while len(self._forward_dispatchers) <= idx:
            self._forward_dispatchers.append(
                self._new_dispatcher("forward", len(self._forward_dispatchers))
            )
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
                state = dispatcher.submit_deepep_dispatch(x_chunk, scores, indices)
            return chunk_idx, dispatcher, state

        def finish_dispatch_expert_submit_combine(pending):
            chunk_idx, dispatcher, state = pending
            del chunk_idx
            with torch.cuda.stream(compute_stream):
                dispatched, tpe, probs = dispatcher.finish_deepep_dispatch(state)
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
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ready)
                combine_state = dispatcher.submit_deepep_combine_prepared(
                    rank_grouped, handle
                )
            del dispatched, probs, expert_out
            return dispatcher, combine_state

        pending_combines = []
        with torch.no_grad():
            next_state = submit_dispatch(0)
            for loop_idx in range(len(ranges)):
                current_state = next_state
                if loop_idx + 1 < len(ranges):
                    next_state = submit_dispatch(loop_idx + 1)
                pending_combines.append(
                    finish_dispatch_expert_submit_combine(current_state)
                )

        done = torch.cuda.Event()
        done.record(compute_stream)
        caller_stream.wait_event(done)
        output_2d = x_2d.new_empty(x_2d.shape)
        offset = 0
        for dispatcher, state in pending_combines:
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
        input_ready = torch.cuda.Event()
        input_ready.record(torch.cuda.current_stream(grad_2d.device))
        dispatcher = self.dispatcher
        grad_x_chunks: list[torch.Tensor | None] = [None for _ in ranges]
        router_accum: list[torch.Tensor | None] = [None for _ in router_params]
        expert_accum: list[torch.Tensor | None] = [None for _ in expert_params]
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
                state = remember_deepep_event(
                    dispatcher.submit_deepep_dispatch(x_chunk, scores, indices)
                )
            return chunk_idx, start, end, x_chunk, scores, state

        def submit_combine_bwd(chunk_idx: int, start: int, end: int, handle: Any):
            del chunk_idx
            with torch.cuda.stream(comm_stream):
                grad_chunk = grad_2d[start:end].contiguous()
                chain_deepep_event()
                return remember_deepep_event(
                    dispatcher.submit_deepep_combine_backward(grad_chunk, handle)
                )

        def finish_recompute_expert(chunk_idx: int, state: dict[str, Any]):
            del chunk_idx
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
                    expert_out = self.experts(
                        expert_input,
                        local_tpe,
                        expert_probs,
                        tokens_per_expert_list=metadata["local_tpe_list"],
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
                    recv_hidden_shape=state["recv_hidden"].shape,
                    recv_hidden_dtype=state["recv_hidden"].dtype,
                    recv_probs_shape=state["recv_probs"].shape,
                    recv_probs_dtype=state["recv_probs"].dtype,
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
                    if chunk.probs is None:
                        expert_inputs = (chunk.dispatched, *expert_params)
                    else:
                        expert_inputs = (chunk.dispatched, chunk.probs, *expert_params)
                    expert_grads = torch.autograd.grad(
                        expert_output,
                        expert_inputs,
                        local_state["grad_expert_out"],
                        allow_unused=True,
                        retain_graph=False,
                    )
                    grad_dispatched = expert_grads[0]
                    if grad_dispatched is None:
                        grad_dispatched = torch.zeros_like(chunk.dispatched)
                    if chunk.probs is None:
                        grad_probs = None
                        param_grads = expert_grads[1:]
                    else:
                        grad_probs = expert_grads[1]
                        if grad_probs is None:
                            grad_probs = torch.zeros_like(chunk.probs)
                        param_grads = expert_grads[2:]
                    _accumulate(expert_accum, expert_params, param_grads)
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
                    local_state["dispatch_bwd_state"] = remember_deepep_event(
                        dispatcher.submit_deepep_dispatch_backward(
                            local_state["grad_recv_hidden"],
                            local_state["grad_recv_probs"],
                            chunk.handle,
                        )
                    )
                    local_state.pop("grad_recv_hidden", None)
                    local_state.pop("grad_recv_probs", None)

                pending_dispatch_bwd.append((chunk, local_state))

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
        expert_grads = _materialize(expert_params, expert_accum)
        router_grads_out = _materialize(router_params, router_accum)
        return grad_x, router_grads_out, expert_grads


class _FullRecomputeFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_2d: torch.Tensor,
        routing_input: torch.Tensor | None,
        layer: EPChunkOverlapMoELayer,
        input_shape: torch.Size,
        input_dtype: torch.dtype,
        fwd_chunks: int,
        bwd_chunks: int,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        del params, input_dtype
        ctx.layer = layer
        ctx.bwd_chunks = bwd_chunks
        ctx.num_router_params = len(tuple(layer.router.parameters()))
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
        output = layer._forward_output_only(x_2d, ranges, input_shape, x_2d.dtype)
        return output.detach().view(input_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        layer = ctx.layer
        x_saved, routing_saved = ctx.saved_tensors
        grad_2d = grad_output.contiguous().view(-1, grad_output.size(-1))
        routing_input = routing_saved if ctx.has_routing_input else None
        with layer._routing_context(routing_input), torch.enable_grad():
            grad_x, router_grads, expert_grads = layer._full_recompute_fused_backward(
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
    grad_recv_hidden = torch.zeros(
        chunk.recv_hidden_shape,
        dtype=chunk.recv_hidden_dtype,
        device=grad_dispatched.device,
    )
    grad_recv_hidden.scatter_add_(
        0,
        row_id_map.unsqueeze(1).expand(-1, grad_dispatched.size(1)),
        grad_dispatched.to(grad_recv_hidden.dtype),
    )
    grad_recv_probs = torch.zeros(
        chunk.recv_probs_shape,
        dtype=chunk.recv_probs_dtype,
        device=grad_dispatched.device,
    )
    if grad_probs is not None:
        flat = chunk.prob_flat_indices.reshape(-1).to(grad_probs.device, torch.long)
        grad_recv_probs.reshape(-1).index_copy_(
            0, flat, grad_probs.reshape(-1).to(grad_recv_probs.dtype)
        )
    return grad_recv_hidden, grad_recv_probs


def _accumulate(
    accum: list[torch.Tensor | None],
    params: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
) -> None:
    for idx, (param, grad) in enumerate(zip(params, grads, strict=True)):
        if grad is None:
            continue
        grad = grad.to(param.dtype)
        accum[idx] = grad if accum[idx] is None else accum[idx] + grad


def _materialize(
    params: tuple[torch.Tensor, ...], accum: list[torch.Tensor | None]
) -> list[torch.Tensor]:
    return [
        torch.zeros_like(param) if grad is None else grad
        for param, grad in zip(params, accum, strict=True)
    ]


__all__ = ["EPChunkOverlapMoELayer", "resolve_ep_chunk_overlap_chunks"]

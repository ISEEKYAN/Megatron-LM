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
    ep_chunk_ranges,
)


def _make_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    if not torch.cuda.is_available():
        raise RuntimeError("EP chunk overlap requires CUDA streams.")
    try:
        return torch.cuda.Stream(device=device)
    except TypeError:
        with torch.cuda.device(device):
            return torch.cuda.Stream()


_EP_CHUNK_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}
_EP_CHUNK_WORKSPACES: dict[tuple[int, torch.dtype, str], torch.Tensor] = {}


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


def _workspace_like(reference: torch.Tensor, shape: torch.Size, role: str) -> torch.Tensor:
    """Return a grow-only shared workspace view for one non-overlapping schedule phase."""
    required = int(torch.Size(shape).numel())
    key = (_cuda_device_index(reference.device), reference.dtype, role)
    workspace = _EP_CHUNK_WORKSPACES.get(key)
    if workspace is None or workspace.numel() < required:
        workspace = torch.empty(required, device=reference.device, dtype=reference.dtype)
        _EP_CHUNK_WORKSPACES[key] = workspace
    return workspace[:required].view(shape)


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


def _merge_expert_chunks(
    chunks: list[torch.Tensor],
    counts: list[list[int]],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert chunk-major expert tokens to one expert-major tensor."""
    if len(chunks) != len(counts) or not chunks:
        raise ValueError("Expert chunks and token counts must be non-empty and aligned.")
    num_experts = len(counts[0])
    if any(len(item) != num_experts for item in counts):
        raise ValueError("All expert token-count lists must have the same length.")
    offsets = [[0] for _ in counts]
    for chunk_offsets, chunk_counts in zip(offsets, counts, strict=True):
        for count in chunk_counts:
            chunk_offsets.append(chunk_offsets[-1] + count)
    pieces = []
    for expert_idx in range(num_experts):
        for chunk, chunk_offsets in zip(chunks, offsets, strict=True):
            pieces.append(chunk[chunk_offsets[expert_idx] : chunk_offsets[expert_idx + 1]])
    if out is None:
        return torch.cat(pieces, dim=0)
    expected_shape = (sum(sum(item) for item in counts), *chunks[0].shape[1:])
    if out.shape != expected_shape:
        raise ValueError(
            f"Expert merge output has shape {out.shape}, expected {expected_shape}."
        )
    offset = 0
    with torch.no_grad():
        for piece in pieces:
            next_offset = offset + piece.size(0)
            out[offset:next_offset].copy_(piece)
            offset = next_offset
    return out


def _split_expert_chunks(
    merged: torch.Tensor,
    counts: list[list[int]],
    *,
    out: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Restore one expert-major tensor to chunk-major expert-token tensors."""
    num_chunks = len(counts)
    num_experts = len(counts[0])
    pieces: list[list[torch.Tensor]] = [[] for _ in counts]
    offset = 0
    for expert_idx in range(num_experts):
        for chunk_idx in range(num_chunks):
            count = counts[chunk_idx][expert_idx]
            pieces[chunk_idx].append(merged[offset : offset + count])
            offset += count
    if offset != merged.size(0):
        raise ValueError("Expert token counts do not cover the merged tensor.")
    if out is None:
        return [torch.cat(chunk_pieces, dim=0) for chunk_pieces in pieces]
    if len(out) != num_chunks:
        raise ValueError("Expert split outputs and token counts must be aligned.")
    with torch.no_grad():
        for output, chunk_pieces in zip(out, pieces, strict=True):
            expected_shape = (sum(piece.size(0) for piece in chunk_pieces), *merged.shape[1:])
            if output.shape != expected_shape:
                raise ValueError(
                    f"Expert split output has shape {output.shape}, expected {expected_shape}."
                )
            offset = 0
            for piece in chunk_pieces:
                next_offset = offset + piece.size(0)
                output[offset:next_offset].copy_(piece)
                offset = next_offset
    return out


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
        if len(ranges) == 2:
            return self._full_recompute_fused_backward_batched(
                x_saved, grad_2d, ranges, router_params, expert_params
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

    def _full_recompute_fused_backward_batched(
        self,
        x_2d: torch.Tensor,
        grad_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        router_params: tuple[torch.Tensor, ...],
        expert_params: tuple[torch.Tensor, ...],
    ):
        """Run both token chunks through one grouped-expert forward/backward."""
        compute_stream, comm_stream = self._streams(grad_2d.device)
        caller_stream = torch.cuda.current_stream(grad_2d.device)
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)
        dispatcher = self.dispatcher
        last_deepep_event: Any | None = None

        def chain_deepep_event() -> None:
            if last_deepep_event is not None:
                _event_current_stream_wait(last_deepep_event)

        def remember_deepep_event(state: dict[str, Any]):
            nonlocal last_deepep_event
            last_deepep_event = state.get("event")
            return state

        work = []
        with torch.enable_grad():
            for chunk_idx in range(len(ranges) - 1, -1, -1):
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
                    dispatch_state = remember_deepep_event(
                        dispatcher.submit_deepep_dispatch(x_chunk, scores, indices)
                    )
                    grad_chunk = grad_2d[start:end].contiguous()
                    chain_deepep_event()
                    combine_state = remember_deepep_event(
                        dispatcher.submit_deepep_combine_backward(
                            grad_chunk, dispatch_state["handle"]
                        )
                    )
                work.append(
                    {
                        "idx": chunk_idx,
                        "start": start,
                        "end": end,
                        "x": x_chunk,
                        "scores": scores,
                        "dispatch": dispatch_state,
                        "combine": combine_state,
                    }
                )

            work.sort(key=lambda item: item["idx"])
            chunks: list[_BackwardChunk] = []
            counts: list[list[int]] = []
            expert_inputs = []
            expert_probs = []
            for item in work:
                state = item["dispatch"]
                with torch.cuda.stream(compute_stream):
                    state["recv_hidden"] = (
                        state["recv_hidden"].detach().requires_grad_(True)
                    )
                    state["recv_probs"] = (
                        state["recv_probs"].detach().requires_grad_(True)
                    )
                    dispatched, _local_tpe, probs, metadata = (
                        dispatcher.finish_deepep_dispatch_external_with_options(
                            state, force_manual_map=True, force_direct_permute=True
                        )
                    )
                    expert_input = dispatched.detach().requires_grad_(True)
                    expert_prob = (
                        None if probs is None else probs.detach().requires_grad_(True)
                    )
                local_counts = metadata["local_tpe_list"]
                row_id_map = metadata["manual_row_id_map"]
                prob_flat_indices = metadata["manual_prob_flat_indices"]
                if (
                    local_counts is None
                    or row_id_map is None
                    or prob_flat_indices is None
                ):
                    raise RuntimeError(
                        "EP chunk overlap batched backward requires local expert metadata."
                    )
                scores = item["scores"]
                scores_edge = None
                scores_ref: torch.Tensor | None = scores
                if hasattr(torch.autograd.graph, "get_gradient_edge"):
                    scores_edge = torch.autograd.graph.get_gradient_edge(scores)
                    scores_ref = None
                chunks.append(
                    _BackwardChunk(
                        idx=item["idx"],
                        start=item["start"],
                        end=item["end"],
                        x=item["x"],
                        scores=scores_ref,
                        handle=state["handle"],
                        row_id_map=row_id_map.detach(),
                        prob_flat_indices=prob_flat_indices.detach(),
                        recv_hidden_scratch=state["recv_hidden"].detach(),
                        recv_probs_scratch=state["recv_probs"].detach(),
                        dispatched=expert_input,
                        probs=expert_prob,
                        expert_out=None,
                        scores_edge=scores_edge,
                        scores_shape=scores.shape,
                        scores_dtype=scores.dtype,
                    )
                )
                counts.append(list(local_counts))
                expert_inputs.append(expert_input)
                expert_probs.append(expert_prob)
                state.pop("recv_hidden", None)
                state.pop("recv_indices", None)
                state.pop("recv_probs", None)

            with torch.cuda.stream(compute_stream):
                merged_shape = (
                    sum(sum(item) for item in counts),
                    *expert_inputs[0].shape[1:],
                )
                merged_input = _merge_expert_chunks(
                    expert_inputs,
                    counts,
                    out=_workspace_like(
                        expert_inputs[0], torch.Size(merged_shape), "expert_input"
                    ),
                ).detach().requires_grad_(True)
                probs_present = [value is not None for value in expert_probs]
                if any(probs_present) != all(probs_present):
                    raise RuntimeError(
                        "DeepEP chunks disagree on expert probability routing."
                    )
                merged_probs = (
                    _merge_expert_chunks(
                        expert_probs,
                        counts,
                        out=_workspace_like(
                            expert_probs[0],
                            torch.Size(
                                (
                                    merged_shape[0],
                                    *expert_probs[0].shape[1:],
                                )
                            ),
                            "expert_probs",
                        ),
                    ).detach().requires_grad_(True)
                    if all(probs_present)
                    else None
                )
                merged_counts = [
                    sum(chunk_counts[idx] for chunk_counts in counts)
                    for idx in range(len(counts[0]))
                ]
                merged_tpe = torch.tensor(
                    merged_counts,
                    dtype=torch.int64,
                    device=merged_input.device,
                )
                with _expert_act_recompute_disabled(self.experts):
                    merged_output = self.experts(
                        merged_input,
                        merged_tpe,
                        merged_probs,
                        tokens_per_expert_list=merged_counts,
                    )
                for chunk, chunk_counts in zip(chunks, counts, strict=True):
                    chunk.expert_out_shape = torch.Size(
                        (sum(chunk_counts), *merged_output.shape[1:])
                    )
                    chunk.expert_out_dtype = merged_output.dtype

                grad_outputs = []
                for chunk, item in zip(chunks, work, strict=True):
                    grad_rank_grouped = dispatcher.finish_deepep_combine_backward(
                        item["combine"]
                    )
                    grad_outputs.append(
                        _manual_unpermute_backward(chunk, grad_rank_grouped)
                    )
                    item["combine"].pop("grad_rank_grouped", None)
                    item["combine"].pop("event", None)
                merged_grad_output = _merge_expert_chunks(
                    grad_outputs,
                    counts,
                    out=_workspace_like(
                        grad_outputs[0],
                        torch.Size(merged_output.shape),
                        "expert_grad_output",
                    ),
                )

                grad_inputs = _expert_grad_inputs(merged_input, merged_probs)
                expert_grads = torch.autograd.grad(
                    merged_output,
                    grad_inputs,
                    merged_grad_output,
                    allow_unused=True,
                )
                grad_merged_input = expert_grads[0]
                if grad_merged_input is None:
                    grad_merged_input = torch.zeros_like(merged_input)
                if merged_probs is None:
                    grad_merged_probs = None
                else:
                    grad_merged_probs = expert_grads[1]
                    if grad_merged_probs is None:
                        grad_merged_probs = torch.zeros_like(merged_probs)
                grad_input_chunks = _split_expert_chunks(
                    grad_merged_input,
                    counts,
                    out=[chunk.dispatched for chunk in chunks],
                )
                grad_prob_chunks = (
                    _split_expert_chunks(
                        grad_merged_probs,
                        counts,
                        out=[chunk.probs for chunk in chunks],
                    )
                    if grad_merged_probs is not None
                    else [None for _ in chunks]
                )

            dispatch_bwd_states = []
            for chunk, grad_input, grad_prob in zip(
                chunks, grad_input_chunks, grad_prob_chunks, strict=True
            ):
                with torch.cuda.stream(compute_stream):
                    grad_recv_hidden, grad_recv_probs = _dispatch_local_backward(
                        chunk, grad_input, grad_prob
                    )
                    local_ready = torch.cuda.Event()
                    local_ready.record(compute_stream)
                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_event(local_ready)
                    chain_deepep_event()
                    dispatch_bwd_states.append(
                        remember_deepep_event(
                            dispatcher.submit_deepep_dispatch_backward(
                                grad_recv_hidden,
                                grad_recv_probs,
                                chunk.handle,
                            )
                        )
                    )
                    _release_dispatch_scratch(chunk, comm_stream)

        with torch.cuda.stream(compute_stream):
            self.experts.flush_delayed_weight_grads(num_contexts=1)

        grad_x_chunks: list[torch.Tensor | None] = [None for _ in chunks]
        router_accum: list[torch.Tensor | None] = [None for _ in router_params]
        for chunk, dispatch_bwd_state in zip(
            chunks, dispatch_bwd_states, strict=True
        ):
            with torch.cuda.stream(compute_stream):
                grad_hidden, grad_scores = dispatcher.finish_deepep_dispatch_backward(
                    dispatch_bwd_state
                )
                if grad_scores is None:
                    grad_scores = torch.zeros(
                        chunk.scores_shape,
                        device=grad_2d.device,
                        dtype=chunk.scores_dtype,
                    )
                router_output = (
                    chunk.scores_edge
                    if chunk.scores_edge is not None
                    else chunk.scores
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
                grad_x_chunks[chunk.idx] = (
                    grad_hidden.to(chunk.x.dtype) + grad_score_x
                )
                _accumulate(router_accum, router_params, router_grads[1:])

        done = torch.cuda.Event()
        done.record(compute_stream)
        caller_stream.wait_event(done)
        grad_x = torch.cat(
            [
                torch.zeros_like(x_2d[start:end]) if grad is None else grad
                for (start, end), grad in zip(
                    ranges, grad_x_chunks, strict=True
                )
            ],
            dim=0,
        ).view_as(grad_2d)
        return (
            grad_x,
            _materialize(router_params, router_accum),
            [None for _ in expert_params],
        )

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

        with torch.cuda.stream(compute_stream):
            self.experts.flush_delayed_weight_grads(
                num_contexts=len(pending_dispatch_bwd)
            )

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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepEP MoE EP chunk-overlap primitive."""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import (
    ep_chunk_ranges,
)


EP_CHUNK_COUNT = 2
EPChunkOpName = Literal["forward", "backward", "fused_forward_backward"]


@dataclass(frozen=True)
class EPChunkWorkspaceKey:
    """Cross-layer workspace identity; layer and chunk are deliberately absent."""

    op: EPChunkOpName
    device_type: str
    device_index: int | None
    ep_group_id: int
    dtype: torch.dtype
    shape_profile: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.op not in {"forward", "backward", "fused_forward_backward"}:
            raise ValueError(f"Unsupported EP chunk op: {self.op!r}")
        if len(self.shape_profile) != 3 or any(
            int(value) <= 0 for value in self.shape_profile
        ):
            raise ValueError(
                "EP chunk shape_profile must be (max_rows, hidden_size, topk)"
            )


@dataclass
class _WorkspaceSlot:
    dispatcher: TokenDispatcher
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    in_use: bool = False
    consumer_event: Any | None = None


class EPChunkWorkspaceLease:
    def __init__(self, workspace: "EPChunkWorkspace", slot: int):
        self.workspace = workspace
        self.slot = slot
        self.dispatcher = workspace.dispatcher(slot)
        self._active = True

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        if not self._active:
            raise RuntimeError("EP chunk workspace lease has already been released")
        return self.workspace._lease_tensor(
            self.slot, name, shape, dtype=dtype, device=device
        )

    def release(self, consumer_event: Any) -> None:
        if not self._active:
            raise RuntimeError("EP chunk workspace lease has already been released")
        if consumer_event is None:
            raise RuntimeError("EP chunk workspace release requires a consumer event")
        slot = self.workspace._slots[self.slot]
        slot.consumer_event = consumer_event
        slot.in_use = False
        self._active = False


class EPChunkWorkspace:
    """Exactly two stable dispatcher/tensor slots owned by one explicit EP op."""

    def __init__(
        self,
        key: EPChunkWorkspaceKey,
        dispatcher_factory: Callable[[int], TokenDispatcher],
    ):
        self.key = key
        self._slots = [
            _WorkspaceSlot(dispatcher_factory(slot)) for slot in range(EP_CHUNK_COUNT)
        ]
        if len({id(slot.dispatcher) for slot in self._slots}) != EP_CHUNK_COUNT:
            raise RuntimeError("EP chunk workspace requires two distinct dispatchers")
        self._allocations = 0
        self._waits = 0

    def dispatcher(self, slot: int) -> TokenDispatcher:
        self._validate_slot(slot)
        return self._slots[slot].dispatcher

    def acquire(self, slot: int, *, stream: Any | None = None) -> EPChunkWorkspaceLease:
        self._validate_slot(slot)
        state = self._slots[slot]
        if state.in_use:
            raise RuntimeError(f"EP chunk workspace slot {slot} is already leased")
        event = state.consumer_event
        if event is not None:
            ready = bool(event.query()) if hasattr(event, "query") else False
            if not ready:
                if stream is not None and hasattr(stream, "wait_event"):
                    stream.wait_event(event)
                elif hasattr(event, "current_stream_wait"):
                    event.current_stream_wait()
                else:
                    raise RuntimeError(
                        "Pending EP chunk consumer event is not stream-waitable"
                    )
                self._waits += 1
            state.consumer_event = None
        state.in_use = True
        return EPChunkWorkspaceLease(self, slot)

    def warmup_tensor(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        for slot in range(EP_CHUNK_COUNT):
            self._reserve_tensor(slot, name, shape, dtype=dtype, device=device)

    def tensor(self, slot: int, name: str) -> torch.Tensor:
        self._validate_slot(slot)
        try:
            return self._slots[slot].tensors[name]
        except KeyError as exc:
            raise RuntimeError(
                f"EP chunk workspace tensor {name!r} is not warm"
            ) from exc

    def _lease_tensor(
        self,
        slot: int,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        tensor = self._reserve_tensor(slot, name, shape, dtype=dtype, device=device)
        requested = tuple(int(dim) for dim in shape)
        slices = tuple(slice(0, dim) for dim in requested)
        return tensor[slices].view(requested).detach().zero_()

    def _reserve_tensor(
        self,
        slot: int,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        self._validate_slot(slot)
        requested = tuple(int(dim) for dim in shape)
        if not requested or any(dim < 0 for dim in requested):
            raise ValueError("EP chunk workspace tensor shape must be non-negative")
        existing = self._slots[slot].tensors.get(name)
        if existing is None:
            existing = torch.empty(requested, dtype=dtype, device=device)
            self._slots[slot].tensors[name] = existing
            self._allocations += 1
            return existing
        capacity = tuple(existing.shape)
        if (
            existing.dtype != dtype
            or existing.device != torch.device(device)
            or len(capacity) != len(requested)
            or any(want > have for want, have in zip(requested, capacity, strict=True))
        ):
            raise RuntimeError(
                f"EP chunk tensor {name!r} shape {requested} exceeds the fixed "
                f"workspace shape {capacity}"
            )
        return existing

    def metrics(self) -> dict[str, int]:
        return {
            "allocations": self._allocations,
            "waits": self._waits,
            "grows": 0,
            "fallbacks": 0,
        }

    @staticmethod
    def _validate_slot(slot: int) -> None:
        if not isinstance(slot, int) or not 0 <= slot < EP_CHUNK_COUNT:
            raise IndexError(f"EP chunk slot must be 0 or 1, got {slot!r}")


class EPChunkWorkspaceRegistry:
    def __init__(self):
        self._workspaces: dict[EPChunkWorkspaceKey, EPChunkWorkspace] = {}

    def get_or_create(
        self,
        key: EPChunkWorkspaceKey,
        dispatcher_factory: Callable[[int], TokenDispatcher],
    ) -> EPChunkWorkspace:
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = EPChunkWorkspace(key, dispatcher_factory)
            self._workspaces[key] = workspace
        return workspace


_EP_CHUNK_WORKSPACES = EPChunkWorkspaceRegistry()


def get_ep_chunk_workspace(
    key: EPChunkWorkspaceKey,
    dispatcher_factory: Callable[[int], TokenDispatcher],
) -> EPChunkWorkspace:
    """Return the process-local workspace shared by every matching model layer."""
    return _EP_CHUNK_WORKSPACES.get_or_create(key, dispatcher_factory)


def _make_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    if not torch.cuda.is_available():
        raise RuntimeError("EP chunk overlap requires CUDA streams.")
    try:
        return torch.cuda.Stream(device=device)
    except TypeError:
        with torch.cuda.device(device):
            return torch.cuda.Stream()


_EP_CHUNK_SHARED_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}
_EP_CHUNK_WGRAD_STREAMS: dict[int, torch.cuda.Stream] = {}
_EP_CHUNK_RECV_ACTIVE: dict[tuple[str, EPChunkOpName, int], dict[str, int]] = {}
_EP_CHUNK_RECV_STATS: dict[str, int] = {}


def _tensor_numel_and_bytes(tensor: torch.Tensor | None) -> tuple[int, int]:
    if tensor is None:
        return 0, 0
    numel = int(tensor.numel())
    return numel, numel * tensor.element_size()


def _record_ep_chunk_recv_tensors(
    *,
    action: str,
    phase: str,
    workspace: EPChunkOpName,
    chunk_idx: int,
    recv_hidden: torch.Tensor | None = None,
    recv_probs: torch.Tensor | None = None,
) -> None:
    key = (phase, workspace, int(chunk_idx))
    if action == "acquire":
        if key in _EP_CHUNK_RECV_ACTIVE:
            raise RuntimeError(f"EP chunk recv block is already active: {key!r}")
        hidden_numel, hidden_bytes = _tensor_numel_and_bytes(recv_hidden)
        probs_numel, probs_bytes = _tensor_numel_and_bytes(recv_probs)
        values = {
            "hidden_numel": hidden_numel,
            "hidden_bytes": hidden_bytes,
            "probs_numel": probs_numel,
            "probs_bytes": probs_bytes,
        }
        _EP_CHUNK_RECV_ACTIVE[key] = values
        prefix = f"recv_{phase}_chunk_{chunk_idx}"
        for name, value in values.items():
            _EP_CHUNK_RECV_STATS[f"{prefix}_{name}"] = value
        _EP_CHUNK_RECV_STATS["active_blocks"] = len(_EP_CHUNK_RECV_ACTIVE)
        _EP_CHUNK_RECV_STATS["active_blocks_peak"] = max(
            _EP_CHUNK_RECV_STATS.get("active_blocks_peak", 0),
            len(_EP_CHUNK_RECV_ACTIVE),
        )
        active_bytes = sum(
            value["hidden_bytes"] + value["probs_bytes"]
            for value in _EP_CHUNK_RECV_ACTIVE.values()
        )
        _EP_CHUNK_RECV_STATS["active_bytes"] = active_bytes
        _EP_CHUNK_RECV_STATS["active_bytes_peak"] = max(
            _EP_CHUNK_RECV_STATS.get("active_bytes_peak", 0), active_bytes
        )
    elif action == "release":
        values = _EP_CHUNK_RECV_ACTIVE.pop(key, None)
        if values is None:
            raise RuntimeError(f"EP chunk recv block is not active: {key!r}")
        _EP_CHUNK_RECV_STATS["active_blocks"] = len(_EP_CHUNK_RECV_ACTIVE)
        _EP_CHUNK_RECV_STATS["active_bytes"] = sum(
            value["hidden_bytes"] + value["probs_bytes"]
            for value in _EP_CHUNK_RECV_ACTIVE.values()
        )
    else:
        raise ValueError(
            f"EP chunk recv action must be acquire or release, got {action!r}"
        )

    if os.environ.get("MEGATRON_LITE_EP_CHUNK_SCRATCH_TRACE", "0") not in {
        "0",
        "false",
        "False",
    }:
        values = (
            _EP_CHUNK_RECV_ACTIVE.get(key, values)
            if action == "release"
            else _EP_CHUNK_RECV_ACTIVE[key]
        )
        print(
            "[EPCHUNK_RECV_TRACE] "
            f"action={action} phase={phase} workspace={workspace} chunk={chunk_idx} "
            f"hidden_numel={values['hidden_numel']} hidden_bytes={values['hidden_bytes']} "
            f"probs_numel={values['probs_numel']} probs_bytes={values['probs_bytes']} "
            f"active_blocks={_EP_CHUNK_RECV_STATS.get('active_blocks', 0)} "
            f"active_bytes={_EP_CHUNK_RECV_STATS.get('active_bytes', 0)}",
            flush=True,
        )


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
    stream = _EP_CHUNK_SHARED_COMM_STREAMS.get(device_index)
    if stream is None:
        stream = _make_stream(device_index)
        _EP_CHUNK_SHARED_COMM_STREAMS[device_index] = stream
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


def _event_current_stream_wait(event: Any) -> None:
    if event is None:
        return
    if hasattr(event, "current_stream_wait"):
        event.current_stream_wait()
    else:
        torch.cuda.current_stream().wait_event(event)


def _record_state_tensors_current_stream(state: dict[str, Any]) -> None:
    for value in state.values():
        if torch.is_tensor(value) and value.is_cuda:
            value.record_stream(torch.cuda.current_stream(value.device))
        elif isinstance(value, dict):
            _record_state_tensors_current_stream(value)


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
    dispatcher: TokenDispatcher
    workspace_lease: EPChunkWorkspaceLease
    scores_edge: Any | None = None
    scores_shape: torch.Size | None = None
    scores_dtype: torch.dtype | None = None
    expert_out_edge: Any | None = None
    expert_out_shape: torch.Size | None = None
    expert_out_dtype: torch.dtype | None = None


class _EPChunkOperationBase:
    """Shared schedule mechanics; each public operation owns its own workspace."""

    def __init__(
        self,
        *,
        router: nn.Module,
        experts: Experts,
        workspace: EPChunkWorkspace,
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
        self.workspace = workspace
        chunk_dispatchers = tuple(
            workspace.dispatcher(slot) for slot in range(EP_CHUNK_COUNT)
        )
        for chunk_idx, chunk_dispatcher in enumerate(chunk_dispatchers):
            if not chunk_dispatcher.use_deepep:
                raise RuntimeError(
                    "EP chunk overlap "
                    f"chunk dispatcher {chunk_idx} has DeepEP disabled."
                )
        if len({id(item) for item in chunk_dispatchers}) != EP_CHUNK_COUNT:
            raise RuntimeError(
                "EP chunk overlap requires distinct per-chunk dispatchers."
            )

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

    def _forward_output_only(
        self,
        x_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        input_shape: torch.Size,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._forward_output_async(
            x_2d,
            ranges,
            input_shape,
            input_dtype,
            disable_expert_act_recompute=True,
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
        if len(ranges) != EP_CHUNK_COUNT:
            raise RuntimeError("EP chunk overlap requires two non-empty token chunks")

        compute_stream, comm_stream = self._streams(x_2d.device)
        caller_stream = torch.cuda.current_stream(x_2d.device)
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)

        def submit_dispatch(chunk_idx: int):
            start, end = ranges[chunk_idx]
            x_chunk = x_2d[start:end]
            lease = self.workspace.acquire(chunk_idx, stream=comm_stream)
            dispatcher = lease.dispatcher
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(input_ready)
                scores, indices = self._route(x_chunk, start, end)
                with _ep_chunk_nvtx("forward.dispatch", chunk_idx):
                    state = dispatcher.submit_deepep_dispatch(
                        x_chunk,
                        scores,
                        indices,
                    )
            return chunk_idx, dispatcher, state, lease

        def finish_dispatch_expert(pending):
            chunk_idx, dispatcher, state, lease = pending
            with torch.cuda.stream(compute_stream):
                with _ep_chunk_nvtx("forward.dispatch.finish", chunk_idx):
                    dispatched, tpe, probs = dispatcher.finish_deepep_dispatch(state)
                _record_state_tensors_current_stream(state)
                recv_hidden = state.get("recv_hidden")
                _record_ep_chunk_recv_tensors(
                    action="acquire",
                    phase="forward",
                    workspace=self.workspace.key.op,
                    chunk_idx=chunk_idx,
                    recv_hidden=recv_hidden,
                    recv_probs=state.get("recv_probs"),
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
                _record_ep_chunk_recv_tensors(
                    action="release",
                    phase="forward",
                    workspace=self.workspace.key.op,
                    chunk_idx=chunk_idx,
                )
            del dispatched, probs, expert_out
            return chunk_idx, dispatcher, rank_grouped, handle, ready, lease

        def submit_combine(prepared):
            chunk_idx, dispatcher, rank_grouped, handle, ready, lease = prepared
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ready)
                with _ep_chunk_nvtx("forward.combine", chunk_idx):
                    combine_state = dispatcher.submit_deepep_combine_prepared(
                        rank_grouped, handle
                    )
            return chunk_idx, dispatcher, combine_state, lease

        output_2d = x_2d.new_empty(x_2d.shape)

        def finish_combine(pending) -> None:
            chunk_idx, dispatcher, state, lease = pending
            with _ep_chunk_nvtx("forward.combine.finish", chunk_idx):
                chunk_out = dispatcher.finish_deepep_combine(state)
            start, end = ranges[chunk_idx]
            output_2d[start:end].copy_(chunk_out)
            consumed = torch.cuda.Event()
            consumed.record(torch.cuda.current_stream(output_2d.device))
            lease.release(consumed)

        pending_combine = None
        with torch.no_grad():
            current_state = submit_dispatch(0)
            for loop_idx in range(len(ranges)):
                prepared = finish_dispatch_expert(current_state)
                if loop_idx + 1 < len(ranges):
                    current_state = submit_dispatch(loop_idx + 1)
                if pending_combine is not None:
                    finish_combine(pending_combine)
                pending_combine = submit_combine(prepared)

        done = torch.cuda.Event()
        done.record(compute_stream)
        caller_stream.wait_event(done)
        assert pending_combine is not None
        finish_combine(pending_combine)
        return output_2d.view(input_shape).to(input_dtype).detach()

    def _full_recompute_fused_backward(
        self,
        x_saved: torch.Tensor,
        grad_2d: torch.Tensor,
    ):
        ranges = ep_chunk_ranges(x_saved.size(0))
        router_params = tuple(self.router.parameters())
        expert_params = tuple(self.experts.parameters())
        return self._full_recompute_fused_backward_v6(
            x_saved,
            grad_2d,
            ranges,
            router_params,
            expert_params,
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
        wgrad_stream = _shared_wgrad_stream(grad_2d.device)
        input_ready = torch.cuda.Event()
        input_ready.record(torch.cuda.current_stream(grad_2d.device))
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
            lease = self.workspace.acquire(chunk_idx, stream=comm_stream)
            dispatcher = lease.dispatcher
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
                        )
                    )
            return chunk_idx, start, end, x_chunk, scores, dispatcher, state, lease

        def submit_combine_bwd(
            chunk_idx: int,
            start: int,
            end: int,
            dispatcher: TokenDispatcher,
            handle: Any,
        ):
            with torch.cuda.stream(comm_stream):
                grad_chunk = grad_2d[start:end].contiguous()
                chain_deepep_event()
                with _ep_chunk_nvtx("backward.combine", chunk_idx):
                    return remember_deepep_event(
                        dispatcher.submit_deepep_combine_backward(grad_chunk, handle)
                    )

        def finish_recompute_expert(
            chunk_idx: int,
            dispatcher: TokenDispatcher,
            state: dict[str, Any],
        ):
            with torch.cuda.stream(compute_stream):
                state["recv_hidden"] = (
                    state["recv_hidden"].detach().requires_grad_(True)
                )
                state["recv_probs"] = state["recv_probs"].detach().requires_grad_(True)
                _record_ep_chunk_recv_tensors(
                    action="acquire",
                    phase="backward",
                    workspace=self.workspace.key.op,
                    chunk_idx=chunk_idx,
                    recv_hidden=state["recv_hidden"],
                    recv_probs=state["recv_probs"],
                )
                dispatched, local_tpe, probs, metadata = (
                    dispatcher.finish_deepep_dispatch_external_with_options(
                        state, force_manual_map=True, force_direct_permute=True
                    )
                )
                _record_state_tensors_current_stream(state)
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
                (
                    chunk_idx,
                    start,
                    end,
                    x_chunk,
                    scores,
                    dispatcher,
                    state,
                    workspace_lease,
                ) = next_state
                combine_state = submit_combine_bwd(
                    chunk_idx, start, end, dispatcher, state["handle"]
                )
                (
                    dispatched,
                    local_tpe,
                    probs,
                    metadata,
                    expert_input,
                    expert_probs,
                    expert_out,
                ) = finish_recompute_expert(chunk_idx, dispatcher, state)

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
                    dispatcher=dispatcher,
                    workspace_lease=workspace_lease,
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
                    expert_inputs = _expert_grad_inputs(chunk.dispatched, chunk.probs)
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
                    _record_ep_chunk_recv_tensors(
                        action="release",
                        phase="backward",
                        workspace=self.workspace.key.op,
                        chunk_idx=chunk.idx,
                    )
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
                grad_hidden, grad_scores = (
                    chunk.dispatcher.finish_deepep_dispatch_backward(
                        local_state["dispatch_bwd_state"]
                    )
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
                consumed = torch.cuda.Event()
                consumed.record(compute_stream)
                chunk.workspace_lease.release(consumed)
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


class _FusedEPChunkFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_2d: torch.Tensor,
        routing_input: torch.Tensor | None,
        fused_op: "EPChunkFusedForwardBackwardOp",
        input_shape: torch.Size,
        input_dtype: torch.dtype,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        del params, input_dtype
        ctx.backward_op = fused_op.backward_op
        ctx.num_router_params = len(tuple(fused_op.router.parameters()))
        ctx.has_routing_input = routing_input is not None
        saved_routing = (
            routing_input.detach()
            if routing_input is not None
            else x_2d.new_empty(0, dtype=torch.long)
        )
        ctx.save_for_backward(x_2d.detach(), saved_routing)
        ranges = ep_chunk_ranges(x_2d.size(0))
        output = fused_op._forward_output_only(
            x_2d,
            ranges,
            input_shape,
            x_2d.dtype,
        )
        return output.detach().view(input_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_saved, routing_saved = ctx.saved_tensors
        routing_input = routing_saved if ctx.has_routing_input else None
        grad_x, router_grads, expert_grads = ctx.backward_op.backward(
            x_saved, grad_output, routing_input
        )
        return (
            grad_x,
            None,
            None,
            None,
            None,
            *router_grads,
            *expert_grads,
        )


class EPChunkForwardOp(_EPChunkOperationBase):
    """Forward-only two-chunk DeepEP operation."""

    def forward(
        self, x: torch.Tensor, routing_input: torch.Tensor | None = None
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "EPChunkForwardOp is forward-only; use fused_forward_backward "
                "when gradients are enabled"
            )
        input_shape = x.shape
        x_2d = x.view(-1, x.size(-1)) if x.dim() == 3 else x
        ranges = ep_chunk_ranges(x_2d.size(0))
        with self._routing_context(routing_input):
            return self._forward_output_async(
                x_2d,
                ranges,
                input_shape,
                x.dtype,
                disable_expert_act_recompute=False,
            )

    __call__ = forward


class EPChunkBackwardOp(_EPChunkOperationBase):
    """Backward recompute operation with a workspace separate from forward."""

    def backward(
        self,
        x_saved: torch.Tensor,
        grad_output: torch.Tensor,
        routing_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor | None]]:
        grad_2d = grad_output.contiguous().view(-1, grad_output.size(-1))
        with self._routing_context(routing_input), torch.enable_grad():
            return self._full_recompute_fused_backward(x_saved, grad_2d)


class EPChunkFusedForwardBackwardOp(_EPChunkOperationBase):
    """Autograd composition of fused forward and the explicit backward op."""

    def __init__(self, *, backward_op: EPChunkBackwardOp, **kwargs):
        super().__init__(**kwargs)
        if (
            backward_op.router is not self.router
            or backward_op.experts is not self.experts
        ):
            raise RuntimeError(
                "Fused EP chunk forward/backward must share model-owned router and experts"
            )
        self.backward_op = backward_op

    def forward(
        self, x: torch.Tensor, routing_input: torch.Tensor | None = None
    ) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.view(-1, x.size(-1)) if x.dim() == 3 else x
        params = tuple(self.router.parameters()) + tuple(self.experts.parameters())
        with self._routing_context(routing_input):
            return _FusedEPChunkFunction.apply(
                x_2d,
                routing_input,
                self,
                input_shape,
                x.dtype,
                *params,
            )

    __call__ = forward


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
    grad_recv_hidden = chunk.workspace_lease.tensor(
        "grad_recv_hidden",
        chunk.recv_hidden_shape,
        dtype=chunk.recv_hidden_dtype,
        device=grad_dispatched.device,
    )
    grad_recv_hidden.scatter_add_(
        0,
        row_id_map.unsqueeze(1).expand(-1, grad_dispatched.size(1)),
        grad_dispatched.to(grad_recv_hidden.dtype),
    )
    grad_recv_probs = chunk.workspace_lease.tensor(
        "grad_recv_probs",
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


__all__ = [
    "EP_CHUNK_COUNT",
    "EPChunkBackwardOp",
    "EPChunkForwardOp",
    "EPChunkFusedForwardBackwardOp",
    "EPChunkWorkspace",
    "EPChunkWorkspaceKey",
    "EPChunkWorkspaceRegistry",
    "get_ep_chunk_workspace",
]

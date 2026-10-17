# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Route-preserving normal DeepEP: per-route expert rows back to their source.

Ported from the DeepSeek-V4 vLLM-aligned dispatcher (Slime-style route
alignment). Tokens travel once through normal DeepEP; a second, top-1
metadata dispatch of per-route fingerprints yields a handle whose combine
returns every (token, top-k slot) expert row to its source rank unreduced, so
the caller owns the top-k reduction (e.g. an ordered FMA).
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from megatron.lite.primitive.modules.dispatcher import (
    TokenDispatcher,
    _DeepEPCombine,
    _DeepEPDispatch,
    deep_ep,
)

_HOT_PATH_ASSERTS = os.getenv("MLITE_VLLM_HOT_PATH_ASSERTS") == "1"


@contextmanager
def _moe_nvtx_range(name: str):
    if os.environ.get("MLITE_STEP_NVTX") != "1" or not torch.cuda.is_available():
        yield
        return
    with torch.cuda.nvtx.range(name):
        yield


@triton.jit
def _route_hash_kernel(
    fingerprint_words, indices, weight_bits, hashes, FINGERPRINT_WORDS: tl.constexpr
):
    row = tl.program_id(0) + tl.arange(0, 1)
    value = tl.full((1,), 1469598103934665603, tl.int64)
    for column in range(FINGERPRINT_WORDS):
        word = tl.load(fingerprint_words + row * FINGERPRINT_WORDS + column).to(
            tl.int64
        )
        value = (value ^ (word & 0xFFFF)) * 1099511628211
    value = (value ^ tl.load(indices + row).to(tl.int64)) * 1099511628211
    bits = tl.load(weight_bits + row).to(tl.int64) & 0xFFFFFFFF
    tl.store(hashes + row, (value ^ bits) * 1099511628211)


@triton.jit
def _scatter_routes_forward_kernel(
    hidden_states,
    output_index,
    output,
    num_slots,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    first_slot = tl.program_id(1)
    num_slot_programs = tl.num_programs(1)
    hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < hidden_size

    for slot_i32 in range(first_slot, num_slots, num_slot_programs):
        slot = slot_i32.to(tl.int64)
        destination_i32 = tl.load(output_index + slot)
        destination = destination_i32.to(tl.int64)
        token = slot // topk
        valid = destination_i32 >= 0
        values = tl.load(
            hidden_states + token * hidden_size + hidden_offsets,
            mask=hidden_mask & valid,
            other=0.0,
        )
        tl.store(
            output + destination * hidden_size + hidden_offsets,
            values,
            mask=hidden_mask & valid,
        )


@triton.jit
def _scatter_routes_backward_kernel(
    grad_output,
    output_index,
    grad_input,
    num_tokens,
    grad_output_rows,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    first_token = tl.program_id(1)
    num_token_programs = tl.num_programs(1)
    hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < hidden_size

    for token_i32 in range(first_token, num_tokens, num_token_programs):
        token = token_i32.to(tl.int64)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for route_slot in range(topk):
            route_i32 = tl.load(output_index + token * topk + route_slot)
            valid = (route_i32 >= 0) & (route_i32 < grad_output_rows)
            safe_route = tl.maximum(route_i32, 0).to(tl.int64)
            route_grad = tl.load(
                grad_output + safe_route * hidden_size + hidden_offsets,
                mask=hidden_mask & valid,
                other=0.0,
            ).to(tl.float32)
            accumulator = (accumulator + route_grad).to(tl.bfloat16).to(tl.float32)
        tl.store(
            grad_input + token * hidden_size + hidden_offsets,
            accumulator,
            mask=hidden_mask,
        )


@triton.jit
def _compact_route_positions_kernel(
    valid, prefix, positions, num_slots, topk: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    flat_slot = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = flat_slot < num_slots
    is_valid = tl.load(valid + flat_slot, mask=in_bounds, other=0).to(tl.int1)
    compact_row = tl.load(prefix + flat_slot, mask=in_bounds, other=0).to(tl.int64) - 1
    write_mask = in_bounds & is_valid
    tl.store(positions + compact_row * 2, flat_slot // topk, mask=write_mask)
    tl.store(positions + compact_row * 2 + 1, flat_slot % topk, mask=write_mask)


def scatter_routes_forward(
    hidden_states: torch.Tensor, output_index: torch.Tensor, output: torch.Tensor
) -> None:
    if output_index.numel() == 0:  # an EP rank may receive no routes
        return
    hidden_size = hidden_states.shape[1]
    block_d = 1024 if hidden_size >= 1024 else triton.next_power_of_2(hidden_size)
    num_slot_programs = min(output_index.numel(), 8192)
    grid = (triton.cdiv(hidden_size, block_d), num_slot_programs)
    _scatter_routes_forward_kernel[grid](
        hidden_states,
        output_index,
        output,
        output_index.numel(),
        hidden_size=hidden_size,
        topk=output_index.shape[1],
        BLOCK_D=block_d,
        num_warps=4,
    )


def scatter_routes_backward(
    grad_output: torch.Tensor, output_index: torch.Tensor, grad_input: torch.Tensor
) -> None:
    if output_index.shape[0] == 0:
        return
    hidden_size = grad_output.shape[1]
    block_d = 1024 if hidden_size >= 1024 else triton.next_power_of_2(hidden_size)
    num_token_programs = min(output_index.shape[0], 2048)
    grid = (triton.cdiv(hidden_size, block_d), num_token_programs)
    _scatter_routes_backward_kernel[grid](
        grad_output,
        output_index,
        grad_input,
        output_index.shape[0],
        grad_output.shape[0],
        hidden_size=hidden_size,
        topk=output_index.shape[1],
        BLOCK_D=block_d,
        num_warps=4,
    )


def compact_route_positions(valid: torch.Tensor, num_routes: int) -> torch.Tensor:
    flat_valid = valid.reshape(-1).contiguous()
    prefix = torch.cumsum(flat_valid, dim=0, dtype=torch.int32)
    positions = torch.empty((num_routes, 2), device=valid.device, dtype=torch.long)
    if flat_valid.numel() and _HOT_PATH_ASSERTS:
        torch._assert_async(
            prefix[-1] == num_routes,
            "DeepEP compact route count differs from its metadata handle",
        )
    if num_routes:
        block_size = 256
        grid = (triton.cdiv(flat_valid.numel(), block_size),)
        _compact_route_positions_kernel[grid](
            flat_valid,
            prefix,
            positions,
            flat_valid.numel(),
            topk=valid.shape[1],
            BLOCK_SIZE=block_size,
        )
    return positions


_BACKWARD_CHUNK_ROWS = 1024


class _DeepEPScatterWithDeterministicBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, hidden_states: torch.Tensor, output_index: torch.Tensor, total_rows: int
    ) -> torch.Tensor:
        output = hidden_states.new_zeros((int(total_rows), hidden_states.shape[1]))
        if hidden_states.is_cuda:
            scatter_routes_forward(
                hidden_states.contiguous(), output_index.contiguous(), output
            )
        else:
            valid_positions = torch.nonzero(output_index >= 0, as_tuple=False)
            for start in range(0, valid_positions.shape[0], _BACKWARD_CHUNK_ROWS):
                positions = valid_positions[start : start + _BACKWARD_CHUNK_ROWS]
                token_rows = positions[:, 0]
                topk_columns = positions[:, 1]
                destination_rows = output_index[token_rows, topk_columns].to(
                    dtype=torch.long
                )
                output.index_copy_(
                    0, destination_rows, hidden_states.index_select(0, token_rows)
                )

        ctx.input_shape = tuple(hidden_states.shape)
        ctx.input_dtype = hidden_states.dtype
        ctx.input_device = hidden_states.device
        ctx.save_for_backward(output_index)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None, None

        (output_index,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.input_shape, dtype=ctx.input_dtype, device=ctx.input_device
        )
        if grad_output.is_cuda:
            with _moe_nvtx_range("moe_bwd/route_scatter"):
                scatter_routes_backward(
                    grad_output.contiguous(), output_index.contiguous(), grad_input
                )
        else:
            for start in range(0, output_index.shape[0], _BACKWARD_CHUNK_ROWS):
                end = min(start + _BACKWARD_CHUNK_ROWS, output_index.shape[0])
                grad_chunk = grad_input[start:end]
                for column in range(output_index.shape[1]):
                    route_rows = output_index[start:end, column].to(dtype=torch.long)
                    valid_rows = route_rows >= 0
                    safe_route_rows = route_rows.clamp(
                        min=0, max=max(grad_output.shape[0] - 1, 0)
                    )
                    if grad_output.shape[0] == 0:
                        continue
                    selected = grad_output.index_select(0, safe_route_rows)
                    selected.masked_fill_(~valid_rows.unsqueeze(1), 0)
                    grad_chunk.add_(selected)
        return grad_input, None, None


def _scatter_deepep_routes_with_padding(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    expected_route_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("DeepEP scatter expects [tokens, hidden] and [tokens, topk]")
    if topk_indices.shape != topk_weights.shape:
        raise ValueError("DeepEP top-k indices and weights must have identical shapes")
    if hidden_states.shape[0] != topk_indices.shape[0]:
        raise ValueError("DeepEP hidden-state and top-k token dimensions differ")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"DeepEP top-k indices must be integer, got {topk_indices.dtype}"
        )
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"DeepEP top-k weights must be FP32, got {topk_weights.dtype}")

    if tokens_per_expert.device.type == "cpu":
        count_values = tuple(
            int(value) for value in tokens_per_expert.reshape(-1).tolist()
        )
        total_rows = sum(count_values)
    else:
        count_values = None

    num_experts = tokens_per_expert.numel()
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    sanitized_indices = topk_indices.masked_fill(~valid, -1)
    real_counts = torch.bincount(
        sanitized_indices.masked_select(valid).to(dtype=torch.long),
        minlength=num_experts,
    )

    counts_are_exact_unpadded = (
        count_values is not None
        and expected_route_count is not None
        and total_rows == expected_route_count
    )
    if counts_are_exact_unpadded:
        counts = real_counts
        if _HOT_PATH_ASSERTS:
            torch._assert_async(
                real_counts.sum() == expected_route_count,
                "DeepEP received route count differs from its route-preserving metadata",
            )
    else:
        counts = tokens_per_expert.to(
            device=topk_indices.device, dtype=torch.long
        ).reshape(-1)
    if _HOT_PATH_ASSERTS:
        torch._assert_async(
            torch.all(real_counts <= counts),
            "DeepEP real route count exceeds its aligned expert count",
        )

    if count_values is None:
        total_rows = int(counts.sum().item())
    permuted_probs = topk_weights.new_zeros((total_rows,))
    output_index = torch.full_like(topk_indices, -1)
    expert_offsets = torch.cumsum(counts, dim=0) - counts

    if expected_route_count is not None and valid.is_cuda:
        occurrences = compact_route_positions(valid.contiguous(), expected_route_count)
    else:
        occurrences = torch.nonzero(valid, as_tuple=False)
    if occurrences.numel():
        token_rows = occurrences[:, 0]
        topk_columns = occurrences[:, 1]
        route_experts = sanitized_indices[token_rows, topk_columns].to(dtype=torch.long)
        expert_order = torch.argsort(route_experts, stable=True)
        token_rows = token_rows.index_select(0, expert_order)
        topk_columns = topk_columns.index_select(0, expert_order)
        route_experts = route_experts.index_select(0, expert_order)

        real_offsets = torch.cumsum(real_counts, dim=0) - real_counts
        within_expert = torch.arange(
            route_experts.numel(), device=hidden_states.device, dtype=torch.long
        ) - real_offsets.index_select(0, route_experts)
        destination_rows = expert_offsets.index_select(0, route_experts) + within_expert
        permuted_probs.index_copy_(
            0, destination_rows, topk_weights[token_rows, topk_columns]
        )
        output_index[token_rows, topk_columns] = destination_rows.to(
            dtype=output_index.dtype
        )

    if _HOT_PATH_ASSERTS:
        torch._assert_async(
            torch.all(~valid | (output_index >= 0)),
            "A valid DeepEP route has no expert-major output row",
        )
    permuted_hidden = _DeepEPScatterWithDeterministicBackward.apply(
        hidden_states, output_index, total_rows
    )
    return (
        permuted_hidden,
        permuted_probs,
        output_index,
        sanitized_indices,
        occurrences,
    )


def _compact_route_preserving_metadata_inputs(
    hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise TypeError("Route metadata requires BF16 [tokens, hidden]")
    if hidden_states.shape[1] < 16:
        raise ValueError("Route fingerprints require hidden size >= 16")
    if topk_indices.ndim != 2 or topk_weights.shape != topk_indices.shape:
        raise ValueError("Route top-k IDs and weights must align")
    if topk_indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("Route token counts must align")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Route IDs must be integer")
    if topk_weights.dtype != torch.float32:
        raise TypeError("Route weights must be FP32")

    flat_indices = topk_indices.reshape(-1)
    valid_positions = torch.nonzero(flat_indices >= 0, as_tuple=False).reshape(-1)
    if valid_positions.numel() == 0:
        raise RuntimeError("Route-preserving DeepEP received no valid expert routes")
    compact_indices = flat_indices.index_select(0, valid_positions)
    compact_weights = topk_weights.detach().reshape(-1).index_select(0, valid_positions)
    token_rows = torch.div(
        valid_positions, topk_indices.shape[1], rounding_mode="floor"
    )
    fingerprints = (
        hidden_states.detach().narrow(1, 0, 16).index_select(0, token_rows).contiguous()
    )
    output_index = torch.full_like(topk_indices, -1, dtype=torch.long)
    output_index.reshape(-1).index_copy_(
        0,
        valid_positions,
        torch.arange(
            valid_positions.numel(), device=valid_positions.device, dtype=torch.long
        ),
    )
    return (
        compact_indices.reshape(-1, 1).contiguous(),
        compact_weights.reshape(-1, 1).contiguous(),
        fingerprints,
        output_index,
        valid_positions.numel() == flat_indices.numel(),
    )


def _deepep_route_handle_received_rows(handle: tuple) -> int:
    if not isinstance(handle, tuple):
        raise TypeError(
            f"DeepEP route handle must be a tuple, got {type(handle).__name__}"
        )
    if len(handle) == 6:
        received_metadata = handle[3]
    elif len(handle) == 10:
        received_metadata = handle[7]
    else:
        raise ValueError(
            f"Unsupported normal DeepEP route handle length: {len(handle)}"
        )
    if not isinstance(received_metadata, torch.Tensor) or received_metadata.ndim < 1:
        raise TypeError("DeepEP route handle has invalid received-source metadata")
    return received_metadata.shape[0]


def _route_hashes(
    fingerprints: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    words = fingerprints.contiguous().view(torch.int16).flatten(1)
    indices = indices.contiguous()
    weight_bits = weights.contiguous().view(torch.int32)
    if words.shape[0] == 0:
        return torch.empty((0,), dtype=torch.int64, device=words.device)
    if words.is_cuda:
        hashes = torch.empty((words.shape[0],), dtype=torch.int64, device=words.device)
        _route_hash_kernel[(words.shape[0],)](
            words,
            indices,
            weight_bits,
            hashes,
            FINGERPRINT_WORDS=words.shape[1],
            num_warps=1,
        )
        return hashes
    hashes = torch.full(
        (words.shape[0],), 1469598103934665603, dtype=torch.int64, device=words.device
    )
    for column in range(words.shape[1]):
        hashes = (hashes ^ (words[:, column].to(torch.int64) & 0xFFFF)) * 1099511628211
    hashes = (hashes ^ indices.to(torch.int64)) * 1099511628211
    return (hashes ^ (weight_bits.to(torch.int64) & 0xFFFFFFFF)) * 1099511628211


def _validate_and_order_route_preserving_outputs(
    received_tokens: torch.Tensor,
    received_topk_indices: torch.Tensor,
    received_topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    route_fingerprints: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    route_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expert-major row of each metadata route, matched by fingerprint and ID."""
    if received_tokens.ndim != 2:
        raise ValueError("Route-preserving DeepEP expects 2D hidden tensors")
    if received_topk_indices.shape != received_topk_weights.shape:
        raise ValueError("Received DeepEP IDs and weights do not align")
    if output_index.shape != received_topk_indices.shape:
        raise ValueError("Received DeepEP route mapping does not align")

    positions = (
        torch.nonzero(output_index >= 0, as_tuple=False)
        if route_positions is None
        else route_positions
    )
    if positions.shape[0] != route_indices.numel():
        raise RuntimeError(
            "Route-preserving DeepEP route count mismatch: "
            f"primary={positions.shape[0]} metadata={route_indices.numel()}"
        )
    token_rows = positions[:, 0]
    topk_slots = positions[:, 1]
    expected_indices = received_topk_indices[token_rows, topk_slots].reshape(-1)
    expected_fingerprints = received_tokens.narrow(1, 0, 16).index_select(0, token_rows)
    if route_fingerprints.shape != expected_fingerprints.shape:
        raise RuntimeError(
            "Route-preserving DeepEP fingerprint shape mismatch: "
            f"{tuple(route_fingerprints.shape)} != {tuple(expected_fingerprints.shape)}"
        )

    expected_weights = received_topk_weights[token_rows, topk_slots].reshape(-1)
    # The hidden-state and metadata dispatches are separate DeepEP operations.
    # At full-model scale they may preserve the same routes while choosing a
    # different arrival order within an expert. Match the two streams by a
    # bitwise route fingerprint before validating and building combine rows.
    expected_hashes = _route_hashes(
        expected_fingerprints, expected_indices, expected_weights
    )
    route_hashes = _route_hashes(
        route_fingerprints,
        route_indices.to(dtype=expected_indices.dtype),
        route_weights.to(dtype=expected_weights.dtype),
    )
    expected_order = torch.argsort(expected_hashes, stable=True)
    route_order = torch.argsort(route_hashes, stable=True)
    expected_for_route = torch.empty_like(expected_order)
    expected_for_route.scatter_(0, route_order, expected_order)
    expected_indices = expected_indices.index_select(0, expected_for_route)
    expected_weights = expected_weights.index_select(0, expected_for_route)
    expected_fingerprints = expected_fingerprints.index_select(0, expected_for_route)
    if _HOT_PATH_ASSERTS:
        torch._assert_async(
            torch.all(
                expected_indices == route_indices.to(dtype=expected_indices.dtype)
            ),
            "Route-preserving DeepEP metadata changed local expert order",
        )
        torch._assert_async(
            torch.all(
                expected_weights.contiguous().view(torch.int32)
                == route_weights.to(dtype=expected_weights.dtype)
                .contiguous()
                .view(torch.int32)
            ),
            "Route-preserving DeepEP metadata changed route probability order",
        )
        torch._assert_async(
            torch.all(expected_fingerprints == route_fingerprints),
            "Route-preserving DeepEP metadata changed source-token order",
        )
    return (
        output_index[token_rows, topk_slots]
        .to(dtype=torch.long)
        .index_select(0, expected_for_route)
    )


_deepep_buffer = None


def _configure_deepep_deterministic_allocator() -> None:
    """Prevent deterministic debug-fill from racing normal DeepEP writes."""
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        torch.utils.deterministic.fill_uninitialized_memory = False


def _get_deepep_buffer(group: dist.ProcessGroup, hidden_bytes: int):
    """Reuse the process-wide normal-DeepEP buffer used by MCore and Slime."""
    if deep_ep is None:
        raise RuntimeError("DeepEP is required for vLLM-aligned EP>1")
    _configure_deepep_deterministic_allocator()
    global _deepep_buffer
    group_size = dist.get_world_size(group=group)
    num_nvl_bytes = 0
    num_rdma_bytes = 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group_size),
        deep_ep.Buffer.get_combine_config(group_size),
    ):
        num_nvl_bytes = max(
            num_nvl_bytes, config.get_nvl_buffer_size_hint(hidden_bytes, group_size)
        )
        if group_size > torch.cuda.device_count():
            num_rdma_bytes = max(
                num_rdma_bytes,
                config.get_rdma_buffer_size_hint(hidden_bytes, group_size),
            )
    if (
        _deepep_buffer is None
        or getattr(_deepep_buffer, "runtime", None) is None
        or _deepep_buffer.group != group
        or _deepep_buffer.num_nvl_bytes < num_nvl_bytes
        or _deepep_buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _deepep_buffer = deep_ep.Buffer(
            group=group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            explicitly_destroy=True,
        )
    return _deepep_buffer


def _dispatch_route_metadata(
    buffer,
    fingerprints: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    num_experts: int,
):
    layout = buffer.get_dispatch_layout(
        route_indices,
        num_experts=num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    return buffer.dispatch(
        fingerprints.contiguous(),
        topk_idx=route_indices.contiguous(),
        topk_weights=route_weights.float().contiguous(),
        num_tokens_per_rank=layout[0],
        num_tokens_per_rdma_rank=layout[1],
        num_tokens_per_expert=layout[2],
        is_token_in_rank=layout[3],
        previous_event=layout[4],
        async_finish=False,
        allocate_on_comm_stream=False,
    )


class _RoutedBackwardDispatch(torch.autograd.Function):
    """Dispatch whose VJP reduces each token's route gradients as all-to-all does.

    Forward is the dispatcher's own route-preserving dispatch. Backward returns
    every route's input and probability gradient unreduced to its source over
    the route-metadata handle (one row per route, so DeepEP sums nothing), then
    orders the routes as the unfused all-to-all permute does (global expert,
    then token) and applies the same ``index_select`` VJP, so dX is bitwise the
    all-to-all dispatcher's under the same determinism mode.
    """

    @staticmethod
    def forward(ctx, hidden_states, topk_scores, topk_indices, dispatcher):
        expert_hidden, counts, expert_probs = dispatcher._dispatch_routes(
            hidden_states, topk_scores, topk_indices
        )
        tokens, topk = topk_indices.shape
        flat = topk_indices.reshape(-1).long()
        route_slots = torch.nonzero(flat >= 0, as_tuple=False).reshape(-1)
        route_tokens = torch.div(route_slots, topk, rounding_mode="floor")
        experts = flat.index_select(0, route_slots)
        ordered = torch.sort(topk_indices.long(), dim=1).values
        torch._assert_async(
            torch.all(ordered[:, 1:] != ordered[:, :-1]),
            "Routed DeepEP backward needs distinct experts per token",
        )
        order = torch.argsort(experts * tokens + route_tokens, stable=True)
        ctx.route_rows = dispatcher._metadata_route_rows
        ctx.route_handle = dispatcher._route_handle
        ctx.buffer = dispatcher.buffer if dispatcher.ep_size > 1 else None
        ctx.route_slots = route_slots
        ctx.order = order
        ctx.order_tokens = route_tokens.index_select(0, order)
        ctx.hidden_shape = tuple(hidden_states.shape)
        ctx.scores_shape = tuple(topk_scores.shape)
        ctx.scores_dtype = topk_scores.dtype
        ctx.mark_non_differentiable(counts)
        return expert_hidden, counts, expert_probs

    @staticmethod
    def backward(ctx, grad_hidden, _grad_counts, grad_probs):
        rows = ctx.route_rows
        if grad_hidden is None:
            grad_hidden = torch.zeros(
                (rows.numel(), ctx.hidden_shape[1]),
                dtype=torch.bfloat16,
                device=rows.device,
            )
        else:
            grad_hidden = grad_hidden.index_select(0, rows)
        grad_probs = (
            torch.zeros(rows.numel(), dtype=torch.float32, device=rows.device)
            if grad_probs is None
            else grad_probs.float().index_select(0, rows)
        )
        if ctx.buffer is not None:
            with _moe_nvtx_range("moe_bwd/deepep/routed_combine"):
                grad_hidden, grad_probs, _ = ctx.buffer.combine(
                    grad_hidden.contiguous(),
                    ctx.route_handle,
                    topk_weights=grad_probs.unsqueeze(1).contiguous(),
                )
            grad_probs = grad_probs.reshape(-1)
        # One row per source route, token-major slot order (as combine_unreduced).
        grad_input = torch.ops.aten.index_select_backward(
            grad_hidden.index_select(0, ctx.order),
            ctx.hidden_shape,
            0,
            ctx.order_tokens,
        )
        grad_scores = grad_probs.new_zeros(math.prod(ctx.scores_shape))
        grad_scores.index_copy_(0, ctx.route_slots, grad_probs)
        grad_scores = grad_scores.reshape(ctx.scores_shape).to(ctx.scores_dtype)
        return grad_input, grad_scores, None, None


class RoutePreservingDeepEPDispatcher(TokenDispatcher):
    """Normal DeepEP transport that returns each route's expert row unreduced.

    ``dispatch`` yields expert-major local rows, CPU per-expert counts and one
    FP32 routing weight per row. ``combine_unreduced`` returns
    ``(rows, row_token, row_expert)`` with one row per valid (token, slot)
    route, in source token-major slot order. Duplicate slots stay separate.

    ``routed_backward`` makes the dispatch VJP reduce route gradients exactly
    as the unfused all-to-all dispatcher does (see ``_RoutedBackwardDispatch``)
    at the cost of returning one gradient row per route instead of one per
    (token, destination rank).
    """

    def __init__(self, *args, routed_backward=False, **kwargs):
        self.routed_backward = bool(routed_backward)
        kwargs["use_deepep"] = False
        super().__init__(*args, **kwargs)
        self.use_deepep = self.ep_size > 1
        self.buffer = None
        if self.use_deepep:
            if deep_ep is None:
                raise RuntimeError("Route-preserving DeepEP requires deep_ep at EP>1")
            if self.ps.tp_ep_group is None:
                raise RuntimeError(
                    "Route-preserving DeepEP at EP>1 requires an EP group"
                )
            self._deepep_group = self.ps.tp_ep_group
            deep_ep.Buffer.set_num_sms(20)

    def _ensure_deepep_buffer(self, hidden_states: torch.Tensor):
        self.buffer = _get_deepep_buffer(
            self._deepep_group,
            hidden_states.shape[1] * max(hidden_states.element_size(), 2),
        )
        return self.buffer

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.routed_backward and torch.is_grad_enabled():
            return _RoutedBackwardDispatch.apply(
                hidden_states, topk_scores, topk_indices, self
            )
        return self._dispatch_routes(hidden_states, topk_scores, topk_indices)

    def _dispatch_routes(self, hidden_states, topk_scores, topk_indices):
        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise TypeError("Route-preserving DeepEP requires BF16 [tokens, hidden]")
        if hidden_states.shape[1] < 16:
            raise ValueError("Route-preserving DeepEP requires hidden size >= 16")
        if topk_indices.shape != topk_scores.shape:
            raise ValueError("top-k IDs and scores must have identical shapes")
        topk_indices = topk_indices.long().contiguous()
        topk_scores = topk_scores.float().contiguous()
        (route_indices, route_weights, route_fingerprints, source_output_index, _) = (
            _compact_route_preserving_metadata_inputs(
                hidden_states, topk_indices, topk_scores
            )
        )
        if self.ep_size > 1:
            buffer = self._ensure_deepep_buffer(hidden_states)
            with _moe_nvtx_range("moe/deepep/primary_dispatch"):
                (
                    received_hidden,
                    received_indices,
                    received_weights,
                    received_per_expert,
                    _,
                ) = _DeepEPDispatch.apply(
                    buffer,
                    hidden_states,
                    topk_indices,
                    topk_scores,
                    self.num_experts,
                    False,
                    False,
                )
            if received_per_expert.device.type != "cpu":
                raise RuntimeError("DeepEP expert counts must remain CPU metadata")
            with _moe_nvtx_range("moe/deepep/metadata_dispatch"):
                (
                    received_fingerprints,
                    received_route_indices,
                    received_route_weights,
                    _,
                    route_handle,
                    _,
                ) = _dispatch_route_metadata(
                    buffer,
                    route_fingerprints,
                    route_indices,
                    route_weights,
                    self.num_experts,
                )
            expected_route_count = _deepep_route_handle_received_rows(route_handle)
        else:
            received_hidden = hidden_states
            received_indices = topk_indices
            received_weights = topk_scores
            received_fingerprints = route_fingerprints
            received_route_indices = route_indices
            received_route_weights = route_weights
            route_handle = None
            received_per_expert = torch.bincount(
                route_indices.reshape(-1).cpu(), minlength=self.num_local_experts
            )
            expected_route_count = route_indices.numel()
        self._local_tpe_list = received_per_expert.tolist()
        with _moe_nvtx_range("moe/deepep/scatter_routes"):
            (
                expert_hidden,
                expert_probs,
                output_index,
                sanitized_indices,
                positions,
            ) = _scatter_deepep_routes_with_padding(
                received_hidden,
                received_indices,
                received_weights,
                received_per_expert,
                expected_route_count=expected_route_count,
            )
        with _moe_nvtx_range("moe/deepep/order_routes"):
            self._metadata_route_rows = _validate_and_order_route_preserving_outputs(
                received_hidden,
                sanitized_indices,
                received_weights,
                output_index,
                received_fingerprints,
                received_route_indices.reshape(-1),
                received_route_weights.reshape(-1),
                route_positions=positions,
            )
        self._route_handle = route_handle
        self._source_indices = topk_indices
        self._source_output_index = source_output_index
        return expert_hidden, received_per_expert, expert_probs

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "RoutePreservingDeepEPDispatcher returns unreduced rows; "
            "use combine_unreduced and reduce the top-k rows in the caller"
        )

    def combine_unreduced(self, expert_output: torch.Tensor):
        route_outputs = expert_output.index_select(0, self._metadata_route_rows)
        if self.ep_size > 1:
            with _moe_nvtx_range("moe/deepep/combine"):
                rows = _DeepEPCombine.apply(
                    self.buffer, route_outputs, self._route_handle, True, False
                )
        else:
            rows = route_outputs
        # Compact route r is the r-th valid (token, slot) in token-major order.
        topk = self._source_indices.shape[1]
        valid_slots = torch.nonzero(
            self._source_output_index.reshape(-1) >= 0, as_tuple=False
        ).reshape(-1)
        row_token = torch.div(valid_slots, topk, rounding_mode="floor")
        row_expert = self._source_indices.reshape(-1).index_select(0, valid_slots)
        for name in (
            "_metadata_route_rows",
            "_route_handle",
            "_source_indices",
            "_source_output_index",
        ):
            delattr(self, name)
        self._local_tpe_list = None
        return rows, row_token, row_expert


__all__ = ["RoutePreservingDeepEPDispatcher"]

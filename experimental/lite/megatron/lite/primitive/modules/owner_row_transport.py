# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lossless fixed-capacity transport for uneven contiguous row owners."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


def _fixed_capacity_all_to_all(
    input_tensor: torch.Tensor,
    input_split_sizes: tuple[int, ...],
    output_split_sizes: tuple[int, ...],
    capacity: int,
    process_group: dist.ProcessGroup,
    *,
    fill_value: int | float,
) -> torch.Tensor:
    """Exchange compact rank segments through an equal-split All-to-All."""
    world_size = dist.get_world_size(process_group)
    if len(input_split_sizes) != world_size or len(output_split_sizes) != world_size:
        raise ValueError(
            "Fixed-capacity All-to-All split metadata must contain one entry per process-group rank"
        )
    if capacity < 0:
        raise ValueError(
            f"Fixed-capacity All-to-All capacity must be non-negative, got {capacity}"
        )
    if any(
        count < 0 or count > capacity
        for count in (*input_split_sizes, *output_split_sizes)
    ):
        raise ValueError(
            f"Fixed-capacity All-to-All counts must lie in [0, {capacity}]: "
            f"input={input_split_sizes}, output={output_split_sizes}"
        )
    if input_tensor.shape[0] != sum(input_split_sizes):
        raise ValueError(
            "Fixed-capacity All-to-All input rows do not match its split metadata: "
            f"{input_tensor.shape[0]} != {sum(input_split_sizes)}"
        )

    output_shape = (sum(output_split_sizes), *input_tensor.shape[1:])
    if capacity == 0:
        return input_tensor.new_empty(output_shape)

    padded_shape = (world_size, capacity, *input_tensor.shape[1:])
    padded_input = input_tensor.new_full(padded_shape, fill_value)
    input_offset = 0
    for destination_rank, count in enumerate(input_split_sizes):
        if count:
            padded_input[destination_rank, :count].copy_(
                input_tensor[input_offset : input_offset + count]
            )
        input_offset += count

    # A sentinel-initialized receive buffer makes a transport that returns
    # without writing deterministic.  The ID route validates these sentinels
    # before local indexing; padding itself is never exposed in compact output.
    padded_output = input_tensor.new_full(padded_shape, fill_value)
    contiguous_input = padded_input.contiguous()
    if contiguous_input.is_cuda:
        torch.cuda.synchronize(contiguous_input.device)
    dist.all_to_all_single(padded_output, contiguous_input, group=process_group)
    if padded_output.is_cuda:
        torch.cuda.synchronize(padded_output.device)

    compact_output = input_tensor.new_empty(output_shape)
    output_offset = 0
    for source_rank, count in enumerate(output_split_sizes):
        if count:
            compact_output[output_offset : output_offset + count].copy_(
                padded_output[source_rank, :count]
            )
        output_offset += count
    return compact_output


class _FixedCapacityAllToAll(torch.autograd.Function):
    """Autograd-aware equal-split All-to-All for compact routed values."""

    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: torch.Tensor,
        input_split_sizes: tuple[int, ...],
        output_split_sizes: tuple[int, ...],
        capacity: int,
        process_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Exchange compact rows through fixed-capacity peer segments."""
        ctx.process_group = process_group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.capacity = capacity
        return _fixed_capacity_all_to_all(
            input_tensor,
            input_split_sizes,
            output_split_sizes,
            capacity,
            process_group,
            fill_value=0,
        )

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """Route output gradients back to the ranks that supplied the rows."""
        grad_input = _fixed_capacity_all_to_all(
            grad_output,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            ctx.capacity,
            ctx.process_group,
            fill_value=0,
        )
        return grad_input, None, None, None, None


class OwnerRowTransport:
    transport_label = 'Row'

    """Transport mixin; caller provides group, rank, size and row boundaries."""

    def _exchange_ids(
        self, sorted_global_ids: torch.Tensor, send_counts: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int]:
        """Send global row IDs to their contiguous row owners."""
        if self.process_group is None:
            counts = (sorted_global_ids.numel(),)
            return sorted_global_ids, counts, counts, sorted_global_ids.numel()
        # Split sizes are Python host metadata used to pack compact segments.
        # Make the CUDA-to-host boundary explicit before materializing them.
        if send_counts.is_cuda:
            torch.cuda.synchronize(send_counts.device)
        input_split_sizes = tuple(int(count) for count in send_counts.cpu().tolist())

        # AllGather exposes one send-count row from every request rank.  Each
        # owner reads its column to obtain source-ordered receive splits.  At
        # EP64 this fixed-shape exchange is only 64 * 64 int64 values
        # (32 KiB/rank).
        gathered_counts = send_counts.new_empty(
            self.owner_world_size * self.owner_world_size
        )
        dist.all_gather_into_tensor(
            gathered_counts, send_counts, group=self.process_group
        )
        if gathered_counts.is_cuda:
            torch.cuda.synchronize(gathered_counts.device)
        count_matrix = gathered_counts.view(
            self.owner_world_size, self.owner_world_size
        )

        # Validate both the rank-local contribution and cross-rank agreement
        # before count metadata can size a payload buffer.  MIN/MAX reductions
        # are cheap for this 32 KiB EP64 matrix and make a plausible but
        # inconsistent AllGather result fail symmetrically instead of causing
        # a payload overread on just one peer.
        minimum_count_matrix = count_matrix.clone()
        maximum_count_matrix = count_matrix.clone()
        dist.all_reduce(
            minimum_count_matrix, op=dist.ReduceOp.MIN, group=self.process_group
        )
        dist.all_reduce(
            maximum_count_matrix, op=dist.ReduceOp.MAX, group=self.process_group
        )
        local_row_matches = torch.equal(count_matrix[self.owner_rank], send_counts)
        local_row_sum_matches = (
            int(count_matrix[self.owner_rank].sum().item()) == sorted_global_ids.numel()
        )
        matrices_match = torch.equal(minimum_count_matrix, maximum_count_matrix)
        count_metadata_valid = (
            local_row_matches and local_row_sum_matches and matrices_match
        )
        valid_tensor = torch.tensor(
            int(count_metadata_valid), device=send_counts.device, dtype=torch.int32
        )
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN, group=self.process_group)
        if not bool(valid_tensor.item()):
            raise RuntimeError(
                f"{self.transport_label} count AllGather produced inconsistent route metadata; "
                "refusing to size the fixed-capacity payload exchange"
            )

        receive_counts = count_matrix[:, self.owner_rank].contiguous()
        output_split_sizes = tuple(
            int(count) for count in receive_counts.cpu().tolist()
        )
        capacity = int(maximum_count_matrix.max().item())
        received_ids = _fixed_capacity_all_to_all(
            sorted_global_ids,
            input_split_sizes,
            output_split_sizes,
            capacity,
            self.process_group,
            fill_value=-1,
        )
        return received_ids, input_split_sizes, output_split_sizes, capacity

    def _validate_sorted_send_ids(
        self, sorted_global_ids: torch.Tensor, send_counts: torch.Tensor
    ) -> None:
        """Symmetrically verify compact destination segments before routing."""
        if self.process_group is None:
            return
        if sorted_global_ids.is_cuda:
            torch.cuda.synchronize(sorted_global_ids.device)
        expected_owners = torch.repeat_interleave(
            torch.arange(
                self.owner_world_size, device=sorted_global_ids.device, dtype=torch.long
            ),
            send_counts,
        )
        actual_owners = torch.bucketize(
            sorted_global_ids,
            torch.tensor(self.boundaries[1:-1], device=sorted_global_ids.device),
            right=True,
        )
        locally_valid = expected_owners.shape == actual_owners.shape and torch.equal(
            expected_owners, actual_owners
        )
        valid_tensor = torch.tensor(
            int(locally_valid), device=sorted_global_ids.device, dtype=torch.int32
        )
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN, group=self.process_group)
        if not bool(valid_tensor.item()):
            raise RuntimeError(
                f"{self.transport_label} sorted ID segments do not match their destination owners; refusing the payload All-to-All"
            )


@dataclass
class _Route:
    group: object
    order: torch.Tensor
    local_ids: torch.Tensor | None
    send_counts: list[int]
    recv_counts: list[int]
    capacity: int

    def exchange(self, tensor, send_counts, recv_counts):
        if self.group is None:
            return tensor
        return _FixedCapacityAllToAll.apply(
            tensor, send_counts, recv_counts, self.capacity, self.group
        )

    def return_rows(self, rows):
        ordered = self.exchange(rows, self.recv_counts, self.send_counts)
        result = torch.empty_like(ordered)
        result[self.order] = ordered
        return result


def _gather_rows(tensor, ids, route=None):
    if tensor is None:
        return None
    # Index FP8 storage as bytes; floating masters retain their autograd edge.
    rows = tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor
    rows = rows[ids if route is None else route.local_ids]
    if route is not None:
        rows = route.return_rows(rows)
    if tensor.element_size() == 1:
        rows = rows.view(tensor.dtype)
    return rows.reshape(*ids.shape, tensor.shape[1])

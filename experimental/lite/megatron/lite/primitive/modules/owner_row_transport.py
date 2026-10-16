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
    dist.all_to_all_single(padded_output, contiguous_input, group=process_group)

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
        if self.group is None:
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
        gathered_counts = send_counts.new_empty(self.size * self.size)
        dist.all_gather_into_tensor(gathered_counts, send_counts, group=self.group)
        count_matrix = gathered_counts.view(self.size, self.size)

        local_row_sum_matches = (
            int(count_matrix[self.rank].sum().item()) == sorted_global_ids.numel()
        )
        if not local_row_sum_matches:
            raise RuntimeError(
                f"{self.transport_label} count AllGather produced inconsistent route metadata; "
                "refusing to size the fixed-capacity payload exchange"
            )

        receive_counts = count_matrix[:, self.rank].contiguous()
        output_split_sizes = tuple(
            int(count) for count in receive_counts.cpu().tolist()
        )
        capacity = int(count_matrix.max().item())
        received_ids = _fixed_capacity_all_to_all(
            sorted_global_ids,
            input_split_sizes,
            output_split_sizes,
            capacity,
            self.group,
            fill_value=-1,
        )
        return received_ids, input_split_sizes, output_split_sizes, capacity


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

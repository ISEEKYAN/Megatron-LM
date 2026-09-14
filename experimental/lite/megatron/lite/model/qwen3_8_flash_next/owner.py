# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Floating owner-row transport ported from NVIDIA-NeMo/Automodel.

Reference: f7ccd6f7902634af34c2f31b3294ac250dc97670 (main), identical owner
implementation in PR3690 5cfe13b160eb7e23ac5a4868bbf611707cdf98fb.
Retain fixed-capacity a2a, source-rank order, stable sorting, synchronization,
metadata validation and native embedding arithmetic. MLite owns local Parameters
through EP/expert-DP instead of Automodel's FSDP-excluded DTensor wrapper.
"""

from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F


def _fixed_capacity_all_to_all(
    input_tensor: torch.Tensor,
    input_split_sizes: tuple[int, ...],
    output_split_sizes: tuple[int, ...],
    capacity: int,
    process_group: dist.ProcessGroup,
    *,
    fill_value: int | float,
) -> torch.Tensor:
    """Exchange compact rank segments through an equal-split All-to-All.

    Padding every peer segment to one globally agreed capacity removes
    backend-specific uneven-split behavior and gives forward and backward the
    same symmetric exchange metadata.  The compact, source-ordered result
    expected by the owner lookup is restored after the collective.  Transport
    provider selection remains an independent runtime concern.

    Args:
        input_tensor: Compact tensor of shape ``[sum(input_split_sizes), ...]``
            whose axis-0 segments are ordered by destination rank.
        input_split_sizes: Number of rows sent to every destination rank.
        output_split_sizes: Number of rows received from every source rank.
        capacity: Globally agreed maximum of every source/destination count.
        process_group: Process group whose rank order defines both count tuples.
        fill_value: Value used for padded and initially untouched output rows.

    Returns:
        A compact tensor of shape ``[sum(output_split_sizes), ...]`` whose
        axis-0 segments are ordered by source rank.
    """
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
        """Exchange compact rows through fixed-capacity peer segments.

        Args:
            ctx: PyTorch autograd context.
            input_tensor: Tensor of shape ``[input_rows, ...]``. Axis 0 contains
                contiguous per-destination segments described by
                ``input_split_sizes``; arbitrary trailing dimensions are kept.
            input_split_sizes: Number of rows sent to every destination rank.
            output_split_sizes: Number of rows received from every source rank.
            capacity: Global maximum peer-segment row count.
            process_group: Process group whose rank order defines both split tuples.

        Returns:
            Tensor of shape ``[output_rows, ...]``, where
            ``output_rows = sum(output_split_sizes)`` and axis 0 is grouped by
            source rank.
        """
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
        """Route output gradients back to the ranks that supplied the rows.

        Args:
            ctx: PyTorch autograd context populated by :meth:`forward`.
            grad_output: Tensor of shape ``[output_rows, ...]`` with the same
                source-rank segmentation as the forward output.

        Returns:
            A gradient tensor of shape ``[input_rows, ...]`` followed by four
            ``None`` entries for the non-tensor split and process-group inputs.
        """
        grad_input = _fixed_capacity_all_to_all(
            grad_output,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            ctx.capacity,
            ctx.process_group,
            fill_value=0,
        )
        return grad_input, None, None, None, None


class OwnerRowLookup:
    def __init__(self, rows, process_group):
        self.process_group = process_group
        self.owner_world_size = dist.get_world_size(process_group)
        self.owner_rank = dist.get_rank(process_group)
        if rows % self.owner_world_size:
            raise ValueError('PLE_OWNER_ROWS_DIVISIBLE')
        self.num_embeddings = rows
        self.num_embeddings_per_rank = rows // self.owner_world_size
        self.vocab_start_index = self.owner_rank * self.num_embeddings_per_rank
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_rank

    def _require_all(self, valid, device, tag):
        flag = torch.tensor(int(valid), device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.process_group)
        if not bool(flag.item()):
            raise ValueError(tag)

    def _validate_global_ids(self, ids):
        self._require_all(
            ids.dtype == torch.int64
            and bool(((ids >= 0) & (ids < self.num_embeddings)).all()),
            ids.device,
            'PLE_OWNER_GLOBAL_IDS',
        )

    def _validate_received_ids(self, ids, output_split_sizes):
        self._require_all(
            ids.numel() == sum(output_split_sizes)
            and bool(
                ((ids >= self.vocab_start_index) & (ids < self.vocab_end_index)).all()
            ),
            ids.device,
            'PLE_OWNER_RECEIVED_ROW_RANGE',
        )

    def _exchange_ids(
        self, sorted_global_ids: torch.Tensor, send_counts: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int]:
        """Send global row IDs to their contiguous row owners.

        Args:
            sorted_global_ids: Tensor of shape ``[request_rows]`` grouped by
                destination owner rank.
            send_counts: Tensor of shape ``[owner_world_size]`` containing the
                number of IDs sent to each owner.

        Returns:
            A tuple containing owner-local received IDs of shape
            ``[owned_requests]``, the per-destination send counts, the
            per-source receive counts, and the globally agreed padded route
            capacity.
        """
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
                "Engram count AllGather produced inconsistent route metadata; "
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
        """Symmetrically verify compact destination segments before routing.

        Args:
            sorted_global_ids: Tensor of shape ``[request_rows]`` containing
                global IDs grouped by contiguous owner rank.
            send_counts: Tensor of shape ``[owner_world_size]`` containing one
                request count per destination owner.

        Raises:
            RuntimeError: If any request rank's segment contains an ID owned by
                a different destination rank.
        """
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
        actual_owners = torch.div(
            sorted_global_ids, self.num_embeddings_per_rank, rounding_mode="floor"
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
                "Engram sorted ID segments do not match their destination owners; refusing the payload All-to-All"
            )

    def gather_rows(
        self, weight: torch.Tensor, global_ids: torch.Tensor
    ) -> torch.Tensor:
        """Look up arbitrary global rows while keeping weights on row owners.

        Args:
            global_ids: Integer tensor of shape ``[...]`` containing global row
                IDs in the packed multi-head table.

        Returns:
            Tensor of shape ``[..., embedding_dim]`` in the original request
            order. The returned tensor does not alias ``global_ids`` or the
            rank-local table weight.
        """
        self._validate_global_ids(global_ids)
        original_shape = global_ids.shape
        flattened_ids = global_ids.reshape(-1).to(dtype=torch.long)
        owners = torch.div(
            flattened_ids, self.num_embeddings_per_rank, rounding_mode="floor"
        )
        send_counts = torch.bincount(owners, minlength=self.owner_world_size).to(
            torch.int64
        )
        sort_indices = torch.argsort(owners, stable=True)
        sorted_global_ids = flattened_ids[sort_indices]
        unsort_indices = torch.empty_like(sort_indices)
        unsort_indices[sort_indices] = torch.arange(
            sort_indices.numel(), device=sort_indices.device
        )
        self._validate_sorted_send_ids(sorted_global_ids, send_counts)

        received_ids, input_split_sizes, output_split_sizes, capacity = (
            self._exchange_ids(sorted_global_ids, send_counts)
        )
        self._validate_received_ids(received_ids, output_split_sizes)
        local_ids = received_ids - self.vocab_start_index
        local_weight = weight
        owned_values = F.embedding(local_ids, local_weight)
        returned_values = _FixedCapacityAllToAll.apply(
            owned_values,
            output_split_sizes,
            input_split_sizes,
            capacity,
            self.process_group,
        )
        return returned_values[unsort_indices].reshape(*original_shape, weight.shape[1])

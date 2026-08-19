"""Route-preserving AllToAll adapter for the vLLM DeepEP-LL contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from megatron.lite.primitive.modules.moe import _AllToAll


@dataclass
class AllToAllRouteState:
    input_splits: list[int]
    output_splits: list[int]
    received_route_count: int
    source_output_index: torch.Tensor
    source_all_routes_valid: bool
    group: dist.ProcessGroup


@dataclass
class AllToAllDispatchResult:
    hidden: torch.Tensor
    local_expert_indices: torch.Tensor
    weights: torch.Tensor
    tokens_per_expert: torch.Tensor
    state: AllToAllRouteState


def dispatch_routes(
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int,
    ep_size: int,
    group: dist.ProcessGroup,
) -> AllToAllDispatchResult:
    """Move every ``(token, top-k slot)`` route without deduplication."""

    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    positions = torch.nonzero(valid, as_tuple=False)
    token_rows = positions[:, 0]
    topk_slots = positions[:, 1]
    route_experts = topk_indices[token_rows, topk_slots].to(dtype=torch.long)
    target_ranks = torch.div(
        route_experts, num_local_experts, rounding_mode="floor"
    )
    order = torch.argsort(target_ranks, stable=True)
    ordered_positions = positions.index_select(0, order)
    ordered_token_rows = token_rows.index_select(0, order)
    ordered_experts = route_experts.index_select(0, order)
    ordered_weights = topk_scores[
        ordered_positions[:, 0], ordered_positions[:, 1]
    ].float()

    input_split_tensor = torch.bincount(
        target_ranks, minlength=ep_size
    ).to(dtype=torch.int64)
    gathered_splits = input_split_tensor.new_empty(ep_size * ep_size)
    dist.all_gather_into_tensor(
        gathered_splits, input_split_tensor, group=group
    )
    ep_rank = dist.get_rank(group=group)
    split_matrix = gathered_splits.view(ep_size, ep_size)
    input_splits = input_split_tensor.tolist()
    output_splits = split_matrix[:, ep_rank].tolist()

    send_hidden = hidden_states.index_select(0, ordered_token_rows).contiguous()
    recv_hidden = _AllToAll.apply(
        send_hidden, input_splits, output_splits, group
    )
    recv_global_experts = _AllToAll.apply(
        ordered_experts.unsqueeze(1),
        input_splits,
        output_splits,
        group,
    ).reshape(-1)
    recv_weights = _AllToAll.apply(
        ordered_weights.unsqueeze(1),
        input_splits,
        output_splits,
        group,
    ).reshape(-1)

    local_start = ep_rank * num_local_experts
    recv_local_experts = recv_global_experts - local_start
    torch._assert_async(
        torch.all(
            (recv_local_experts >= 0)
            & (recv_local_experts < num_local_experts)
        ),
        "alltoall delivered a route to a non-local expert",
    )
    actual_tokens_per_expert = torch.bincount(
        recv_local_experts, minlength=num_local_experts
    ).to(dtype=torch.int64)
    received_route_count = recv_hidden.shape[0]
    received_per_expert = torch.where(
        actual_tokens_per_expert > 0,
        ((actual_tokens_per_expert + 127) // 128) * 128,
        actual_tokens_per_expert,
    )
    padding_rows = int(
        (received_per_expert.sum() - actual_tokens_per_expert.sum()).item()
    )
    if padding_rows:
        recv_hidden = torch.cat(
            (
                recv_hidden,
                recv_hidden.new_zeros((padding_rows, recv_hidden.shape[1])),
            ),
            dim=0,
        )
        recv_local_experts = torch.cat(
            (
                recv_local_experts,
                recv_local_experts.new_full((padding_rows,), -1),
            ),
            dim=0,
        )
        recv_weights = torch.cat(
            (recv_weights, recv_weights.new_zeros((padding_rows,))),
            dim=0,
        )

    source_output_index = torch.full_like(
        topk_indices, -1, dtype=torch.long
    )
    source_output_index[
        ordered_positions[:, 0], ordered_positions[:, 1]
    ] = torch.arange(
        ordered_positions.shape[0],
        device=topk_indices.device,
        dtype=torch.long,
    )
    state = AllToAllRouteState(
        input_splits=input_splits,
        output_splits=output_splits,
        received_route_count=received_route_count,
        source_output_index=source_output_index,
        source_all_routes_valid=(
            positions.shape[0] == topk_indices.numel()
        ),
        group=group,
    )
    return AllToAllDispatchResult(
        hidden=recv_hidden,
        local_expert_indices=recv_local_experts.reshape(-1, 1),
        weights=recv_weights.reshape(-1, 1),
        tokens_per_expert=received_per_expert,
        state=state,
    )


def combine_routes(
    route_outputs: torch.Tensor, state: AllToAllRouteState
) -> torch.Tensor:
    return _AllToAll.apply(
        route_outputs,
        state.output_splits,
        state.input_splits,
        state.group,
    )

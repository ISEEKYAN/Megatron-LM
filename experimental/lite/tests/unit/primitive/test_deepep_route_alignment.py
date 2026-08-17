from __future__ import annotations

import torch

from megatron.lite.primitive.alignment.deepep_route import (
    _validate_and_order_route_preserving_outputs,
)


def test_route_fingerprint_recovers_metadata_receive_order() -> None:
    received_tokens = torch.arange(4 * 16, dtype=torch.bfloat16).reshape(4, 16)
    received_indices = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    received_weights = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    output_index = torch.tensor([[0], [2], [1], [3]], dtype=torch.int64)
    expert_outputs = torch.tensor([[10.0], [30.0], [20.0], [40.0]])

    # Metadata dispatch receives the same route multiset in a payload-dependent
    # order different from the primary hidden dispatch.
    metadata_to_primary = torch.tensor([2, 0, 3, 1])
    route_fingerprints = received_tokens.index_select(0, metadata_to_primary)
    route_indices = received_indices.reshape(-1).index_select(
        0, metadata_to_primary
    )
    route_weights = received_weights.reshape(-1).index_select(
        0, metadata_to_primary
    )

    route_rows = _validate_and_order_route_preserving_outputs(
        expert_outputs,
        received_tokens,
        received_indices,
        received_weights,
        output_index,
        route_fingerprints,
        route_indices,
        route_weights,
        return_route_rows=True,
    )
    assert torch.equal(route_rows, torch.tensor([1, 0, 3, 2]))
    ordered = _validate_and_order_route_preserving_outputs(
        expert_outputs,
        received_tokens,
        received_indices,
        received_weights,
        output_index,
        route_fingerprints,
        route_indices,
        route_weights,
    )
    assert torch.equal(ordered, torch.tensor([[30.0], [10.0], [40.0], [20.0]]))


def test_metadata_is_authoritative_when_primary_route_weights_are_undefined() -> None:
    received_tokens = torch.arange(4 * 16, dtype=torch.bfloat16).reshape(4, 16)
    received_indices = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    received_weights = torch.full((4, 1), float("nan"))
    output_index = torch.tensor([[0], [2], [1], [3]], dtype=torch.int64)
    expert_outputs = torch.tensor([[10.0], [30.0], [20.0], [40.0]])

    metadata_to_primary = torch.tensor([2, 0, 3, 1])
    route_fingerprints = received_tokens.index_select(0, metadata_to_primary)
    route_indices = received_indices.reshape(-1).index_select(0, metadata_to_primary)
    route_weights = torch.tensor([0.3, 0.1, 0.4, 0.2])

    ordered = _validate_and_order_route_preserving_outputs(
        expert_outputs,
        received_tokens,
        received_indices,
        received_weights,
        output_index,
        route_fingerprints,
        route_indices,
        route_weights,
    )
    assert torch.equal(ordered, torch.tensor([[30.0], [10.0], [40.0], [20.0]]))

from __future__ import annotations

import torch

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    _compact_route_dispatch_inputs,
    _ordered_route_backward,
)


def test_ordered_route_backward_ignores_padded_slots() -> None:
    route_values = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    topk_weights = torch.tensor([[0.25, 9.0], [0.5, 11.0]])
    output_index = torch.tensor([[0, -1], [1, -1]])
    grad_output = torch.tensor([[13.0, 17.0], [19.0, 23.0]])
    grad_routes = torch.zeros_like(route_values)
    grad_weights = torch.zeros_like(topk_weights)

    _ordered_route_backward(
        route_values=route_values,
        topk_weights=topk_weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=grad_routes,
        grad_weights=grad_weights,
        static_mapping_valid=False,
    )

    torch.testing.assert_close(
        grad_routes,
        torch.stack((grad_output[0] * 0.25, grad_output[1] * 0.5)),
        rtol=0,
        atol=0,
    )
    assert torch.equal(grad_weights[:, 1], torch.zeros(2))
    assert grad_weights[0, 0] == torch.dot(grad_output[0], route_values[0])
    assert grad_weights[1, 0] == torch.dot(grad_output[1], route_values[1])


def test_route_slot_compaction_preserves_duplicate_order_and_weight_bits() -> None:
    hidden = torch.arange(3 * 16, dtype=torch.bfloat16).reshape(3, 16)
    indices = torch.tensor([[1, 1], [-1, 0], [1, 9]], dtype=torch.int64)
    weight_bits = torch.tensor(
        [0x3E800001, 0x3F000001, 0x7FC00001, 0x3F400001, 0x3F600001, 0x3F700001],
        dtype=torch.int32,
    )
    weights = weight_bits.view(torch.float32).reshape(3, 2)

    route_indices, route_weights, route_hidden, output_index, all_valid = (
        _compact_route_dispatch_inputs(hidden, indices, weights, num_experts=2)
    )

    assert not all_valid
    assert route_indices.reshape(-1).tolist() == [1, 1, 0, 1]
    assert output_index.tolist() == [[0, 1], [-1, 2], [3, -1]]
    assert torch.equal(route_hidden, hidden.index_select(0, torch.tensor([0, 0, 1, 2])))
    assert torch.equal(
        route_weights.reshape(-1).view(torch.int32),
        weight_bits.index_select(0, torch.tensor([0, 1, 3, 4])),
    )

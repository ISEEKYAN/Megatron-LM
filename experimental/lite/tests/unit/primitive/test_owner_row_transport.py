# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.primitive.modules import engram_lookup as transport


@pytest.fixture
def route(monkeypatch):
    owner = transport.OwnerRowTransport()
    owner.process_group = object()
    owner.owner_world_size, owner.owner_rank = 2, 0
    owner.boundaries = (0, 2, 5)
    monkeypatch.setattr(transport.dist, 'get_world_size', lambda group: 2)
    monkeypatch.setattr(transport.dist, 'all_reduce', lambda *a, **k: None)
    return owner


def test_rejects_wrong_owner_segment(route):
    with pytest.raises(RuntimeError, match='sorted ID segments'):
        route._validate_sorted_send_ids(torch.tensor([4, 0]), torch.tensor([1, 1]))


def test_rejects_corrupt_count_matrix(route, monkeypatch):
    monkeypatch.setattr(
        transport.dist,
        'all_gather_into_tensor',
        lambda output, *a, **k: output.copy_(torch.tensor([0, 2, 0, 0])),
    )
    with pytest.raises(RuntimeError, match='inconsistent route metadata'):
        route._exchange_ids(torch.tensor([0, 4]), torch.tensor([1, 1]))


def test_rejects_wrong_input_rows(route):
    with pytest.raises(ValueError, match='input rows do not match'):
        transport._fixed_capacity_all_to_all(
            torch.ones(1, 3), (1, 1), (1, 1), 1, route.process_group, fill_value=0
        )


def test_uneven_owner_boundary_and_empty_segments(route, monkeypatch):
    route._validate_sorted_send_ids(torch.tensor([1, 2, 4]), torch.tensor([1, 2]))
    route._validate_sorted_send_ids(torch.tensor([2, 4]), torch.tensor([0, 2]))
    monkeypatch.setattr(
        transport.dist,
        'all_to_all_single',
        lambda output, value, **k: output.copy_(value),
    )
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    actual = transport._FixedCapacityAllToAll.apply(
        value, (0, 2), (0, 2), 2, route.process_group
    )
    assert torch.equal(actual, value)
    actual.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))
    empty = transport._fixed_capacity_all_to_all(
        value[:0], (0, 0), (0, 0), 0, route.process_group, fill_value=0
    )
    assert empty.shape == (0, 3)

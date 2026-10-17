# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Existing Engram hashes restart per document across contiguous CP cuts."""

import pytest
import torch


@pytest.fixture
def modules(v41_core_te):
    from megatron.lite.primitive.modules.attention.packed_geometry import (
        PackedCPGeometry,
    )
    from megatron.lite.primitive.modules.engram_lookup import NgramHash, prime_buckets
    from megatron.lite.primitive.modules.packed_row_memory import packed_sequence_hashes

    def make_hash(order):
        multipliers = torch.tensor([[3, 5, 7, 11, 13], [17, 19, 23, 29, 31]])
        return NgramHash(
            torch.arange(64) % 17,
            0,
            multipliers[:, :order],
            prime_buckets((1, 4), order, 2, 43),
        )

    return PackedCPGeometry, packed_sequence_hashes, make_hash


@pytest.mark.parametrize('cp', [1, 2, 3, 4])
@pytest.mark.parametrize('order', [3, 5])
@pytest.mark.parametrize('mask_kind', ['none', 'mixed', 'inactive'])
def test_packed_hashes_equal_document_hashes_then_cp_slice(
    modules, cp, order, mask_kind
):
    geometry_type, packed, make_hash = modules
    bounds = (0, 1, 5, 7, 16)
    ids = torch.arange(1, 17).view(1, -1)
    mask = None
    if mask_kind != 'none':
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[:, ::3] = False
        if mask_kind == 'inactive':
            mask.zero_()
    hasher = make_hash(order)
    expected = torch.cat(
        [
            hasher(ids[:, begin:end], None if mask is None else mask[:, begin:end])
            for begin, end in zip(bounds, bounds[1:])
        ],
        1,
    )
    for rank in range(cp):
        geometry = geometry_type(bounds, rank, cp)
        begin = geometry.global_start
        end = min(begin + geometry.width, geometry.total)
        actual = packed(
            hasher,
            ids[:, begin:end],
            None if mask is None else mask[:, begin:end],
            geometry,
            lookback_ids=ids[:, :begin],
            lookback_mask=None if mask is None else mask[:, :begin],
        )
        assert torch.equal(actual, expected[:, begin:end])


def test_pure_pad_rank_returns_empty_hash_without_requesting_history(modules):
    geometry_type, packed, make_hash = modules
    hasher = make_hash(5)
    actual = packed(
        hasher,
        torch.empty(1, 0, dtype=torch.int64),
        None,
        geometry_type((0, 1, 2), 3, 4),
    )
    assert actual.shape == (1, 0, 2, 8)


@pytest.mark.parametrize('missing', ['tokens', 'mask'])
def test_missing_cp_history_fails_closed(modules, missing):
    geometry_type, packed, make_hash = modules
    ids = torch.arange(1, 9).view(1, -1)
    options = {} if missing == 'tokens' else {'lookback_ids': ids[:, :4]}
    with pytest.raises(ValueError, match='history|left-neighbor'):
        packed(
            make_hash(5),
            ids[:, 4:],
            torch.ones(1, 4, dtype=torch.bool),
            geometry_type((0, 3, 8), 1, 2),
            **options
        )


def test_history_cannot_leak_across_a_document_boundary(modules):
    geometry_type, packed, make_hash = modules
    ids = torch.tensor([[1, 2, 3, 4]])
    hasher = make_hash(3)
    wanted = hasher(ids[:, 2:])
    assert not torch.equal(hasher(ids)[:, 2:], wanted)
    actual = packed(
        hasher,
        ids[:, 2:],
        None,
        geometry_type((0, 2, 4), 1, 2),
        lookback_ids=ids[:, :2],
    )
    assert torch.equal(actual, wanted)


@pytest.mark.parametrize('rank', [0, 1])
def test_fixed_zero_prefixed_halo_matches_full_document_hashes(modules, rank):
    geometry_type, packed, make_hash = modules
    geometry = geometry_type((0, 2, 7), rank, 2)
    ids = torch.arange(1, 8).view(1, -1)
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, 3] = False
    hasher = make_hash(5)
    expected = torch.cat(
        (hasher(ids[:, :2], mask[:, :2]), hasher(ids[:, 2:], mask[:, 2:])), 1
    )
    halo = torch.zeros(1, 6, dtype=ids.dtype)
    halo_mask = torch.zeros(1, 6, dtype=torch.bool)
    start = geometry.global_start
    end = min(start + geometry.width, geometry.total)
    first = max(0, start - halo.shape[1])
    count = start - first
    if count:
        halo[:, -count:] = ids[:, first:start]
        halo_mask[:, -count:] = mask[:, first:start]
    actual = packed(
        hasher,
        ids[:, start:end],
        mask[:, start:end],
        geometry,
        lookback_ids=halo,
        lookback_mask=halo_mask,
    )
    assert torch.equal(actual, expected[:, start:end])

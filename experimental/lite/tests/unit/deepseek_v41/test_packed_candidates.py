# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed candidate masks retain the deployed per-document block policy."""

import pytest
import torch


@pytest.fixture
def modules(v41_core_te):
    from megatron.lite.primitive.modules.attention.csa import candidate_blocks
    from megatron.lite.primitive.modules.attention.packed_candidates import (
        packed_candidate_blocks,
    )
    from megatron.lite.primitive.modules.attention.packed_geometry import (
        PackedCPGeometry,
    )

    return PackedCPGeometry, packed_candidate_blocks, candidate_blocks


@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('cp', [1, 2, 4])
@pytest.mark.parametrize('ties', [False, True])
def test_packed_candidate_mask_equals_full_document_then_cp_slice(
    modules, ratio, cp, ties
):
    geometry_type, packed, document_policy = modules
    bounds = (0, 5, 21, 24)
    kv_bounds = [0]
    for begin, end in zip(bounds, bounds[1:]):
        kv_bounds.append(kv_bounds[-1] + (end - begin) // ratio)
    torch.manual_seed(702)
    scores = torch.randn(1, bounds[-1], kv_bounds[-1])
    if ties:
        scores.fill_(1)
    expected = torch.zeros_like(scores, dtype=torch.bool)
    for doc, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        first, last = kv_bounds[doc : doc + 2]
        lengths = ((torch.arange(end - begin) + 1) // ratio)[:, None]
        expected[:, begin:end, first:last] = document_policy(
            scores[:, begin:end, first:last], lengths, topk_blocks=2, block_size=3
        )
    for rank in range(cp):
        geometry = geometry_type(bounds, rank, cp)
        query_bounds = [0]
        for doc in range(geometry.num_documents):
            begin, end = geometry.owned(doc)
            query_bounds.append(query_bounds[-1] + end - begin)
        start = geometry.global_start
        stop = min(start + geometry.width, geometry.total)
        actual = packed(
            scores[:, start:stop],
            geometry,
            ratio,
            kv_bounds=tuple(kv_bounds),
            query_bounds=tuple(query_bounds),
            topk_blocks=2,
            block_size=3,
        )
        assert torch.equal(actual, expected[:, start:stop])


def test_zero_compressed_groups_and_pure_pad_rank_have_empty_masks(modules):
    geometry_type, packed, _ = modules
    for rank in range(4):
        geometry = geometry_type((0, 1, 2), rank, 4)
        query_bounds = [0]
        for doc in range(2):
            begin, end = geometry.owned(doc)
            query_bounds.append(query_bounds[-1] + end - begin)
        scores = torch.empty(1, query_bounds[-1], 0)
        result = packed(
            scores, geometry, 2, kv_bounds=(0, 0, 0), query_bounds=tuple(query_bounds)
        )
        assert result.shape == scores.shape and result.dtype == torch.bool


@pytest.mark.parametrize('field', ['query_bounds', 'kv_bounds'])
def test_invalid_candidate_boundaries_are_rejected(modules, field):
    geometry_type, packed, _ = modules
    options = dict(kv_bounds=(0, 2, 5), query_bounds=(0, 2, 5))
    options[field] = (0, 1)
    with pytest.raises(ValueError, match='boundaries'):
        packed(torch.zeros(1, 5, 5), geometry_type((0, 2, 5)), 1, **options)

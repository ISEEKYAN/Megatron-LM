# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed page writes/gathers preserve independent request payloads on CPU."""

import pytest
import torch


@pytest.fixture
def modules(v41_core_te):
    from megatron.lite.primitive.modules.attention.packed_geometry import (
        PackedCPGeometry,
    )
    from megatron.lite.primitive.modules.deployment_math.packed_cache import (
        packed_cache_geometry,
    )

    return PackedCPGeometry, packed_cache_geometry


def prefixes(geometry, ratio):
    bounds, queries = [0], [0]
    for doc, (begin, end) in enumerate(
        zip(geometry.boundaries, geometry.boundaries[1:])
    ):
        bounds.append(bounds[-1] + ((end - begin) // ratio if ratio else end - begin))
        first, last = geometry.owned(doc)
        queries.append(queries[-1] + last - first)
    return tuple(bounds), tuple(queries)


@pytest.mark.parametrize('ratio', [0, 1, 2])
@pytest.mark.parametrize('cp', [1, 2, 4])
@pytest.mark.parametrize('block_size', [4, 64])
def test_document_pages_and_local_queries_match_independent_requests(
    modules, ratio, cp, block_size
):
    geometry_type, packed = modules
    previous_writer = None
    for rank in range(cp):
        geometry = geometry_type((0, 1, 65, 130), rank, cp)
        kv_bounds, query_bounds = prefixes(geometry, ratio)
        result = packed(
            geometry,
            query_bounds=query_bounds,
            kv_bounds=kv_bounds,
            ratio=ratio,
            block_size=block_size,
        )
        writer, query = result.writer, result.query
        assert writer.query_start_loc.tolist() == list(kv_bounds)
        assert writer.table.dtype == torch.int32 and writer.slots.dtype == torch.int64
        assert writer.slots.unique().numel() == kv_bounds[-1]
        payload = torch.stack((writer.requests.long(), writer.positions), -1)
        cache = torch.full((writer.num_pages * block_size, 2), -99, dtype=torch.int64)
        cache[writer.slots] = payload
        query_positions, query_requests, used_pages = [], [], set()
        for doc, (begin, end) in enumerate(
            zip(geometry.boundaries, geometry.boundaries[1:])
        ):
            first, last = kv_bounds[doc : doc + 2]
            rows = last - first
            n_pages = max(1, (rows + block_size - 1) // block_size)
            pages = set(writer.table[doc, :n_pages].tolist())
            assert not used_pages & pages
            used_pages |= pages
            local_rows = torch.arange(rows)
            addresses = (
                writer.table[doc, local_rows // block_size].long() * block_size
                + local_rows % block_size
            )
            expected_positions = local_rows * ratio + ratio - 1 if ratio else local_rows
            wanted = torch.stack(
                (torch.full_like(local_rows, doc), expected_positions), -1
            )
            assert torch.equal(cache[addresses], wanted)
            first, last = geometry.owned(doc)
            query_positions.extend(range(first - begin, last - begin))
            query_requests.extend([doc] * (last - first))
        assert query.query_start_loc.tolist() == list(query_bounds)
        assert query.lengths.tolist() == [
            b - a for a, b in zip(query_bounds, query_bounds[1:])
        ]
        assert query.positions.tolist() == query_positions
        assert query.requests.tolist() == query_requests
        if previous_writer is not None:
            for field in (
                'table',
                'slots',
                'lengths',
                'positions',
                'requests',
                'query_start_loc',
            ):
                assert torch.equal(
                    getattr(writer, field), getattr(previous_writer, field)
                )
        previous_writer = writer


@pytest.mark.parametrize('ratio', [0, 1, 2])
def test_pure_pad_rank_keeps_full_writer_and_no_query_rows(modules, ratio):
    geometry_type, packed = modules
    geometry = geometry_type((0, 1, 2), 3, 4)
    kv_bounds, query_bounds = prefixes(geometry, ratio)
    result = packed(
        geometry,
        query_bounds=query_bounds,
        kv_bounds=kv_bounds,
        ratio=ratio,
        block_size=4,
    )
    assert result.writer.lengths.tolist() == ([0, 0] if ratio == 2 else [1, 1])
    assert result.query.query_start_loc.tolist() == [0, 0, 0]
    assert result.query.positions.numel() == result.query.requests.numel() == 0


@pytest.mark.parametrize('kind', ['missing_kv', 'query_rows', 'prefix'])
def test_cache_metadata_rejects_missing_or_inconsistent_boundaries(modules, kind):
    geometry_type, packed = modules
    options = dict(query_bounds=(0, 2, 5), kv_bounds=(0, 2, 5), ratio=1)
    if kind == 'missing_kv':
        options['kv_bounds'] = None
    elif kind == 'query_rows':
        options['query_bounds'] = (0, 1, 5)
    else:
        options['kv_bounds'] = (0, 3, 2)
    with pytest.raises(ValueError, match='boundaries'):
        packed(geometry_type((0, 2, 5)), **options)

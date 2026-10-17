# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Document-private writer pages and CP-local query metadata."""

from dataclasses import dataclass

import torch

from .cache import cache_geometry


@dataclass(frozen=True)
class CacheRequestRows:
    query_start_loc: torch.Tensor
    lengths: torch.Tensor
    positions: torch.Tensor
    requests: torch.Tensor


@dataclass(frozen=True)
class CacheWriterRows(CacheRequestRows):
    table: torch.Tensor
    slots: torch.Tensor
    num_pages: int


@dataclass(frozen=True)
class PackedCacheGeometry:
    writer: CacheWriterRows
    query: CacheRequestRows


def packed_cache_geometry(
    geometry, *, query_bounds, kv_bounds=None, ratio=0, block_size=64
):
    """Keep full-document writers on every rank and omit padded query rows.

    CR1/2 prefixes come from AttentionState/Core. Writer positions are the
    original group's last token, matching cache.py's native insert calls;
    the unchanged deployment kernels apply RoPE at the group's first token.
    Query lengths and positions describe only this rank's real rows.
    """
    if ratio not in (0, 1, 2) or block_size < 1:
        raise ValueError('Packed cache requires CR0/1/2 and positive page size')
    bounds = geometry.boundaries if ratio == 0 else kv_bounds
    documents = geometry.num_documents
    if bounds is None or any(
        len(prefix) != documents + 1
        or prefix[0] != 0
        or any(b < a for a, b in zip(prefix, prefix[1:]))
        for prefix in (bounds, query_bounds)
    ):
        raise ValueError('Cache boundaries must be document prefixes from Core/state')
    expected_queries = [0]
    for doc in range(documents):
        first, last = geometry.owned(doc)
        expected_queries.append(expected_queries[-1] + last - first)
    if tuple(query_bounds) != tuple(expected_queries):
        raise ValueError('Cache query boundaries must contain real local rows')
    tables, slots, positions, requests = [], [], [], []
    page_offset = 0
    for doc, (first, last) in enumerate(zip(bounds, bounds[1:])):
        rows = last - first
        table, row_slots, _ = cache_geometry(
            1, rows, block_size, device=geometry.device
        )
        tables.append(table + page_offset)
        slots.append(row_slots.flatten() + page_offset * block_size)
        pos = torch.arange(rows, dtype=torch.int64, device=geometry.device)
        if ratio:
            pos = pos * ratio + ratio - 1
        positions.append(pos)
        requests.append(torch.full_like(pos, doc, dtype=torch.int32))
        page_offset += table.numel()
    max_pages = max(t.shape[1] for t in tables)
    table = torch.cat(
        [torch.nn.functional.pad(t, (0, max_pages - t.shape[1])) for t in tables]
    )
    writer_cu = torch.tensor(bounds, dtype=torch.int32, device=geometry.device)
    query_cu = torch.tensor(query_bounds, dtype=torch.int32, device=geometry.device)
    query_lengths = query_cu.diff()
    writer = CacheWriterRows(
        query_start_loc=writer_cu,
        lengths=writer_cu.diff(),
        positions=torch.cat(positions),
        requests=torch.cat(requests),
        table=table.contiguous(),
        slots=torch.cat(slots).contiguous(),
        num_pages=page_offset,
    )
    query = CacheRequestRows(
        query_start_loc=query_cu,
        lengths=query_lengths,
        positions=geometry.positions[: query_bounds[-1]].contiguous(),
        requests=torch.repeat_interleave(
            torch.arange(documents, dtype=torch.int32, device=geometry.device),
            query_lengths,
        ),
    )
    return PackedCacheGeometry(writer, query)

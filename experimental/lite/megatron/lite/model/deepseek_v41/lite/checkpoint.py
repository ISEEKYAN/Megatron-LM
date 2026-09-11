# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded row loading from a validated release-entry manifest."""

import hashlib

import torch


def _iter_rows(entry, begin, end, chunk_rows):
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('chunk_rows must be positive')
    width = entry.shape[1]  # Validated one-byte FP8/E8M0 entries only.
    if width <= 0 or entry.byte_length != entry.shape[0] * width or entry.offset < 0:
        raise ValueError('Invalid matrix byte interval')
    digest = hashlib.sha256()
    with open(entry.source_shard, 'rb') as source:
        source.seek(entry.offset)
        for first in range(0, entry.shape[0], chunk_rows):
            last = min(first + chunk_rows, entry.shape[0])
            raw = source.read((last - first) * width)
            if len(raw) != (last - first) * width:
                raise ValueError('Truncated matrix payload')
            digest.update(raw)
            lo, hi = max(first, begin), min(last, end)
            if lo < hi:
                yield lo, raw[(lo - first) * width : (hi - first) * width]
    if digest.hexdigest() != entry.payload_digest:
        raise ValueError('Payload digest mismatch')


def load_engram_rows(store, name, *, intervals, rank, device, chunk_rows=4096):
    """Load only the assigned FP8/E8M0 rows, with bounded host staging.

    CPU is supported for byte-level fixtures; runtime callers pass their CUDA
    device. No decoded full table or persistent host replica is created.
    """
    if not name.endswith('.engram.embed.weight'):
        raise ValueError('Require exact Engram weight family')
    weight = store.entries[name]
    scale_name = name[:-6] + 'scale'
    scale = store.entries[scale_name]
    if (
        weight.release_key != name
        or scale.release_key != scale_name
        or weight.dtype != 'F8_E4M3'
        or len(weight.shape) != 2
        or weight.shape[1] % 32
        or scale.dtype != 'F8_E8M0'
        or scale.shape != (weight.shape[0], weight.shape[1] // 32)
    ):
        raise ValueError(
            'Engram requires FP8 values and matching row/block32 E8M0 scales'
        )
    intervals = tuple(intervals)
    cursor = 0
    for begin, end in intervals:
        if (
            type(begin) is not int
            or type(end) is not int
            or begin != cursor
            or end < begin
        ):
            raise ValueError('Row coverage has gaps, overlaps or invalid intervals')
        cursor = end
    if cursor != weight.shape[0] or not 0 <= rank < len(intervals):
        raise ValueError('Row coverage or rank does not match logical table')
    begin, end = intervals[rank]
    tensors = []
    for key, entry in ((name, weight), (scale_name, scale)):
        result = torch.empty(
            (end - begin, entry.shape[1]), device=device, dtype=torch.uint8
        )
        for first, raw in _iter_rows(entry, begin, end, chunk_rows):
            chunk = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
                -1, entry.shape[1]
            )
            result[first - begin : first - begin + chunk.shape[0]].copy_(chunk)
        tensors.append(
            result.view(
                torch.float8_e4m3fn
                if entry.dtype == 'F8_E4M3'
                else torch.float8_e8m0fnu
            )
        )
    return tuple(tensors)


def load_engram_table(
    store,
    name,
    lookup,
    *,
    device,
    trainable=False,
    output_dtype=torch.bfloat16,
    chunk_rows=4096
):
    """Compose the model's release loader with the generic resident provider."""
    from megatron.lite.primitive.modules import engram_lookup

    if lookup.group is not None and torch.device(device).type != 'cuda':
        raise ValueError('Distributed Engram checkpoint load requires a CUDA device')
    values, scales = load_engram_rows(
        store,
        name,
        intervals=tuple(zip(lookup.boundaries, lookup.boundaries[1:])),
        rank=lookup.rank,
        device=device,
        chunk_rows=chunk_rows,
    )
    return engram_lookup.ShardedEngramTable(
        values, scales, lookup, trainable=trainable, output_dtype=output_dtype
    )

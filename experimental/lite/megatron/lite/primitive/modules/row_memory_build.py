# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build resident row memories from explicit vocabulary and projection dimensions."""
import torch

from . import engram_lookup as memory


def build_row_memories(
    *,
    layer_ids,
    row_counts,
    order,
    heads,
    vocabulary,
    width,
    compressed_vocabulary,
    pad_id,
    token_map,
    token_count,
    hidden_size,
    copies,
    eps,
    trainable,
    group,
    group_size,
    local_range,
    projection
):
    hash_module = None
    if layer_ids and token_map is not None:
        with torch.device('cpu'):
            primes = memory.prime_buckets(layer_ids, order, heads, vocabulary)
            if primes.flatten(1).sum(1).tolist() != row_counts:
                raise ValueError('Row counts disagree with prime layout')
            if (
                len(token_map) != token_count
                or min(token_map) < 0
                or max(token_map) >= compressed_vocabulary
            ):
                raise ValueError('Token map disagrees with compressed vocabulary')
            hash_module = memory.NgramHash(
                token_map,
                pad_id,
                memory.hash_multipliers(layer_ids, order, compressed_vocabulary),
                primes,
            )
    memories = {}
    for slot, index in enumerate(layer_ids):
        if not local_range[0] <= index < local_range[1]:
            continue
        rows, table_type, options = row_counts[slot], memory.EngramTable, {}
        if group is not None:
            boundaries = [rows * i // group_size for i in range(group_size + 1)]
            lookup = memory.RowLookup(boundaries, group)
            rows = boundaries[lookup.rank + 1] - boundaries[lookup.rank]
            table_type, options = memory.ShardedEngramTable, {'lookup': lookup}
        table = table_type(
            torch.zeros(rows, width, dtype=torch.float8_e4m3fn),
            torch.ones(rows, width // 32, dtype=torch.float8_e8m0fnu),
            trainable=trainable,
            **options
        )
        memories[index] = memory.Engram(
            hidden_size,
            copies,
            table,
            projection((order - 1) * heads * width, (copies + 1) * hidden_size),
            eps=eps,
        )
    return hash_module, memories


def sequence_hashes(hash_module, input_ids, token_mask, context):
    if hash_module is None:
        raise ValueError('Row lookup requires an explicit tokenizer token map')
    if context is not None:
        input_ids = context.gather(input_ids)
        if token_mask is not None:
            token_mask = context.gather(token_mask)
    hashes = hash_module(input_ids, token_mask)
    return hashes if context is None else context.slice(hashes)

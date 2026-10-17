# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Apply DS41's existing candidate policy within each packed document."""

import torch


def packed_candidate_blocks(
    scores,
    geometry,
    ratio,
    *,
    kv_bounds,
    query_bounds,
    topk_blocks=2048,
    block_size=8,
    phase='post-training',
):
    """Return a global-column mask while keeping document-local block IDs.

    Prefixes come from AttentionState v1.3. Core computes visible counts;
    the unchanged DS41 candidate policy sees exactly its per-document width,
    including its original tie and unfinished-last-block behavior.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils.packed_layout import (
        build_seq_lens,
    )

    from .csa import candidate_blocks

    documents = geometry.num_documents
    if ratio not in (1, 2):
        raise ValueError('Candidate blocks require DS41 CR1/2')
    if any(
        len(bounds) != documents + 1
        or bounds[0] != 0
        or any(b < a for a, b in zip(bounds, bounds[1:]))
        for bounds in (kv_bounds, query_bounds)
    ):
        raise ValueError('Candidate boundaries must be document prefixes')
    expected_queries = [0]
    offsets = []
    for doc, begin in enumerate(geometry.boundaries[:-1]):
        start, stop = geometry.owned(doc)
        expected_queries.append(expected_queries[-1] + stop - start)
        offsets.append(start - begin)
    if tuple(query_bounds) != tuple(expected_queries):
        raise ValueError('Query boundaries must cover real local document rows')
    if (
        scores.ndim != 3
        or scores.shape != (1, query_bounds[-1], kv_bounds[-1])
        or scores.device != geometry.device
    ):
        raise ValueError('Candidate scores must cover THD local queries and global KV')
    query_cu = torch.tensor(query_bounds, dtype=torch.int32, device=scores.device)
    key_cu = torch.tensor(kv_bounds, dtype=torch.int32, device=scores.device)
    offsets = torch.tensor(offsets, dtype=torch.int32, device=scores.device)
    lengths = build_seq_lens(query_cu, key_cu, scores.shape[1], ratio, offsets)
    result = torch.zeros_like(scores, dtype=torch.bool)
    for doc in range(documents):
        first, last = query_bounds[doc : doc + 2]
        key_first, key_last = kv_bounds[doc : doc + 2]
        result[:, first:last, key_first:key_last] = candidate_blocks(
            scores[:, first:last, key_first:key_last],
            lengths[first:last, None],
            topk_blocks=topk_blocks,
            block_size=block_size,
            phase=phase,
        )
    return result

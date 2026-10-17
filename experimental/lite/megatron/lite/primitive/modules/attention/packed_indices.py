# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS41's physical KV layout lowered by the upstream Core packed builder."""

import torch


def packed_sparse_metadata(
    geometry,
    selected=None,
    *,
    ratio=0,
    window_size,
    lookback_rows=None,
    width=None,
    compressed_cu_seqlens=None,
    compressed_rows=None,
    kv_bounds=None,
    query_bounds=None
):
    """Keep selected compressed slots, ascending SWA slots, then -1 padding.

    The physical buffer is compressed sequence-major rows followed by the
    left-neighbor window and real local rows. Selected IDs are document-relative,
    covering only real query rows (AttentionState interface v1.3). Core receives
    temporary tail padding; no padded row is returned to the attention kernel.
    Explicit lookback_rows retains Core's fixed, right-aligned halo, including
    its zero prefix on the first rank; Core masks it by document boundaries.
    Core owns document isolation, causality and tail-padding masks. Compressed
    prefixes come from Core compressor_input, and compressed_rows is the
    sequence-major KV buffer length; neither ownership nor visibility is
    recomputed here.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils.packed_layout import (
        build_attention_indices,
    )

    real_rows = max(0, min(geometry.width, geometry.total - geometry.global_start))
    if query_bounds is not None:
        expected = [0]
        for doc in range(geometry.num_documents):
            start, stop = geometry.owned(doc)
            expected.append(expected[-1] + stop - start)
        if tuple(query_bounds) != tuple(expected):
            raise ValueError('Query boundaries must cover real local document rows')
    if kv_bounds is not None:
        if (
            len(kv_bounds) != geometry.num_documents + 1
            or kv_bounds[0] != 0
            or any(b < a for a, b in zip(kv_bounds, kv_bounds[1:]))
        ):
            raise ValueError('KV boundaries must be Core compressed prefixes')
        if compressed_cu_seqlens is not None or compressed_rows is not None:
            raise ValueError('Pass either host KV boundaries or Core tensor metadata')
        compressed_cu_seqlens = torch.tensor(
            kv_bounds, dtype=torch.int32, device=geometry.device
        )
        compressed_rows = kv_bounds[-1]
    if ratio not in (0, 1, 2) or window_size < 1:
        raise ValueError('Expected DS41 CR0/1/2 and a positive window')
    required = min(geometry.global_start, window_size - 1)
    lookback = required if lookback_rows is None else lookback_rows
    if lookback < required:
        raise ValueError('Sparse window is missing left-neighbor rows')
    if ratio:
        if (
            compressed_cu_seqlens is None
            or compressed_rows is None
            or compressed_cu_seqlens.shape != geometry.cu_seqlens.shape
            or compressed_cu_seqlens.dtype != torch.int32
            or compressed_cu_seqlens.device != geometry.device
            or not isinstance(compressed_rows, int)
            or compressed_rows < 0
        ):
            raise ValueError('Compressed metadata must come from Core compressor_input')
        bounds = compressed_cu_seqlens
        total = compressed_rows
    else:
        bounds = torch.zeros_like(geometry.cu_seqlens)
        total = 0
    if selected is not None:
        if (
            not ratio
            or selected.ndim != 2
            or selected.shape[0] != real_rows
            or selected.device != geometry.cu_seqlens.device
            or selected.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                'Selected IDs must cover real local rows on the geometry device'
            )
        columns = selected.shape[1]
        selected = torch.nn.functional.pad(
            selected, (0, 0, 0, geometry.width - real_rows), value=-1
        )
    else:
        columns = (
            max(
                (b - a) // ratio
                for a, b in zip(geometry.boundaries, geometry.boundaries[1:])
            )
            if ratio
            else 0
        )
    logical_width = columns + window_size
    width = ((logical_width + 127) // 128) * 128 if width is None else width
    if width < logical_width:
        raise ValueError('Sparse width would truncate compressed or SWA slots')
    mapping = torch.arange(total, dtype=torch.int32, device=geometry.cu_seqlens.device)
    indices, _, _, _ = build_attention_indices(
        geometry.cu_seqlens_padded,
        geometry.global_start,
        geometry.width,
        lookback,
        window_size,
        ratio or 1,
        columns,
        selected,
        cu_seqlens_compressed=bounds,
        seq_to_rank_row=mapping,
        for_indexer_loss=True,
        compressed_rows=total,
        cu_seqlens_unpadded=geometry.cu_seqlens,
    )
    # Core's physical buffer is [window | compressed], but DS41's deployed
    # reduction order uses [compressed | window]. Translate bases, then move
    # invalid slots to the tail without changing any valid-slot order.
    indices = torch.cat(
        [
            torch.where(
                indices[:, :columns] >= 0,
                indices[:, :columns] - (lookback + geometry.width),
                -1,
            ),
            torch.where(indices[:, columns:] >= 0, indices[:, columns:] + total, -1),
        ],
        -1,
    )
    indices = indices[:real_rows]
    positions = torch.arange(logical_width, device=indices.device)[None, :]
    order = torch.where(indices >= 0, positions, positions + logical_width).argsort(-1)
    indices = indices.gather(1, order)
    lengths = (indices >= 0).sum(-1, dtype=torch.int32)
    return (
        torch.nn.functional.pad(
            indices, (0, width - logical_width), value=-1
        ).contiguous(),
        lengths,
    )

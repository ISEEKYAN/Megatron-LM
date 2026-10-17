# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core boundary transport and DS4 compressed-row gathering for packed CSA.

Hidden rows are first and contain only real local tokens. Core's fixed-width
transport padding is temporary inside the boundary helper. These helpers do not change
native cache writer geometry or perform document-local communication.
"""

import torch


def packed_boundary_hidden(hidden, geometry, *, window_size, ratio=0):
    """Exchange a fixed right-aligned halo; keep rank zero's zero prefix.

    Use Core's existing differentiable exchange with an explicit window, not
    its DS4-specific compressor halo rule. Packed metadata masks document
    boundaries; the transport must not trim or mask the received rows.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
        exchange_cp_boundary_hidden,
    )

    if ratio not in (0, 1, 2) or window_size < 0:
        raise ValueError('Expected DS41 CR0/1/2 and a nonnegative window')
    real_rows = max(0, min(geometry.width, geometry.total - geometry.global_start))
    if hidden.ndim < 2 or hidden.shape[0] != real_rows:
        raise ValueError('Boundary input must contain only the real local CP rows')
    if hidden.device != geometry.device:
        raise ValueError('Boundary input must be on the geometry device')
    rows = max(window_size, max(ratio - 1, 0))
    if not rows:
        # Core sends tensor[-d_window:]; -0 would send the whole input.
        return hidden[:0]
    if geometry.cp_size == 1:
        return hidden.new_zeros((rows, *hidden.shape[1:]))
    # This restriction is geometric and identical on all ranks. Reject it
    # before any communication (including ranks with no real token rows).
    if geometry.width < rows:
        raise ValueError(
            'Core boundary exchange requires local width >= halo: '
            f'T={geometry.total}, cp={geometry.cp_size}, '
            f'width={geometry.width}, d={rows}'
        )
    group = geometry.cp_group
    if (
        group is None
        or group.size() != geometry.cp_size
        or group.rank() != geometry.cp_rank
    ):
        raise ValueError('Boundary transport requires the geometry CP group')
    padded = torch.cat(
        (hidden, hidden.new_zeros((geometry.width - real_rows, *hidden.shape[1:]))), 0
    )
    return exchange_cp_boundary_hidden(padded, 1, rows, group)


def gather_packed_compressed_rows(local_rows, row_map, geometry):
    """DS4's rank-major gather and Core's sequence-major compressed row map.

    This is the gather_cp_compressed_rows path from DS4 attention/runtime.py.
    Core owns the row map; no owner or document ordering is recomputed here.
    """
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
    )

    if local_rows.ndim < 2 or local_rows.device != geometry.device:
        raise ValueError('Compressed rows must be row-first on the geometry device')
    if (
        row_map.ndim != 1
        or row_map.dtype not in (torch.int32, torch.int64)
        or row_map.device != geometry.device
    ):
        raise ValueError('Expected Core compressed row_map on the geometry device')
    if geometry.cp_size == 1:
        rank_major = local_rows
    else:
        group = geometry.cp_group
        if (
            group is None
            or group.size() != geometry.cp_size
            or group.rank() != geometry.cp_rank
        ):
            raise ValueError('Compressed transport requires the geometry CP group')
        rank_major = gather_from_sequence_parallel_region(local_rows, group=group)
    return rank_major, torch.index_select(rank_major, 0, row_map.clamp_min(0).long())

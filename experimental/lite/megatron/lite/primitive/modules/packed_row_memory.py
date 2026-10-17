# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Supply packed boundaries and CP history to the existing Engram hash."""

import torch


def packed_sequence_hashes(
    hash_module,
    input_ids,
    token_mask,
    geometry,
    *,
    lookback_ids=None,
    lookback_mask=None,
):
    """Hash real local rows after prepending the supplied order-1 history.

    History transport belongs to the caller. NgramHash owns token mapping,
    mask propagation, document resets and integer hash arithmetic. Extra left
    history is accepted so the caller may share a larger boundary exchange.
    """
    if hash_module is None:
        raise ValueError('Row lookup requires an explicit tokenizer token map')
    rows = max(0, min(geometry.width, geometry.total - geometry.global_start))
    if input_ids.shape != (1, rows) or input_ids.device != geometry.device:
        raise ValueError('Packed hash input must contain the real local THD rows')
    if token_mask is not None and (
        token_mask.shape != input_ids.shape
        or token_mask.dtype != torch.bool
        or token_mask.device != input_ids.device
    ):
        raise ValueError('Packed hash token mask must match local rows')
    if rows == 0:
        return hash_module(input_ids, token_mask)
    order = hash_module.multipliers.shape[1]
    needed = min(order - 1, geometry.global_start)
    if needed and lookback_ids is None:
        raise ValueError('Packed hash is missing CP left-neighbor token history')
    if lookback_ids is not None and (
        lookback_ids.ndim != 2
        or lookback_ids.shape[0] != 1
        or lookback_ids.shape[1] < needed
        or lookback_ids.dtype != input_ids.dtype
        or lookback_ids.device != input_ids.device
    ):
        raise ValueError('Packed hash left-neighbor token history has wrong shape')
    if lookback_mask is not None and (
        lookback_ids is None
        or lookback_mask.shape != lookback_ids.shape
        or lookback_mask.dtype != torch.bool
        or lookback_mask.device != input_ids.device
    ):
        raise ValueError('Packed hash history mask must match history tokens')
    if needed and token_mask is not None and lookback_mask is None:
        raise ValueError('Masked packed hash requires the left-neighbor mask')
    history = lookback_ids[:, -needed:] if needed else input_ids[:, :0]
    ids = torch.cat((history, input_ids), 1)
    mask = None
    if token_mask is not None or lookback_mask is not None:
        local_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if token_mask is not None:
            local_mask = token_mask
        left_mask = lookback_mask[:, -needed:] if needed else local_mask[:, :0]
        mask = torch.cat((left_mask, local_mask), 1)
    begin = geometry.global_start - needed
    end = geometry.global_start + rows
    boundaries = [0]
    boundaries.extend(p - begin for p in geometry.boundaries[1:-1] if begin < p < end)
    boundaries.append(end - begin)
    cu = torch.tensor(boundaries, dtype=torch.int32, device=input_ids.device)
    return hash_module(ids, mask, cu_seqlens=cu)[:, needed:]

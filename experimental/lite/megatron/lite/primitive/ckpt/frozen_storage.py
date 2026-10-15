# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded byte fingerprints for resident immutable storage."""
import hashlib

import torch


def storage_digest(*tensors, block_bytes=8 * 1024**2):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        if not tensor.is_contiguous():
            raise ValueError('Frozen storage must be contiguous')
        flat = tensor.detach().view(torch.uint8).reshape(-1)
        for offset in range(0, flat.numel(), block_bytes):
            digest.update(flat[offset : offset + block_bytes].cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_storage_census(records, total_rows=None):
    """Require an ordered, contiguous global partition with byte digests."""
    if not isinstance(records, list) or not records:
        raise ValueError('Invalid frozen Engram shard census')
    cursor = 0
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {'start', 'end', 'sha256'}
            or type(record['start']) is not int
            or type(record['end']) is not int
            or record['start'] != cursor
            or record['end'] <= cursor
            or not isinstance(record['sha256'], str)
            or len(record['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in record['sha256'])
        ):
            raise ValueError('Invalid frozen Engram shard census')
        cursor = record['end']
    if total_rows is not None and cursor != total_rows:
        raise ValueError('Frozen Engram shard census coverage mismatch')


def storage_census(weight, scale, *, start, boundaries):
    """Fingerprint owned pieces of a common global row partition."""
    end = start + weight.shape[0]
    cuts = [cut for cut in boundaries if start <= cut <= end]
    if not cuts or cuts[0] != start or cuts[-1] != end:
        raise ValueError('Frozen Engram shard boundaries do not align with storage')
    return [
        {
            'start': a,
            'end': b,
            'sha256': storage_digest(
                weight[a - start : b - start].view(torch.float8_e4m3fn),
                scale[a - start : b - start].view(torch.float8_e8m0fnu),
            ),
        }
        for a, b in zip(cuts, cuts[1:])
    ]


def verify_storage_census(records, weight, scale, *, start, total_rows):
    """Verify every installed row without gathering another rank's table."""
    validate_storage_census(records, total_rows)
    end = start + weight.shape[0]
    owned = [
        record for record in records if record['end'] > start and record['start'] < end
    ]
    boundaries = (
        [record['start'] for record in owned] + [owned[-1]['end']] if owned else []
    )
    actual = storage_census(weight, scale, start=start, boundaries=boundaries)
    if actual != owned:
        raise ValueError('Frozen Engram shard census digest mismatch')

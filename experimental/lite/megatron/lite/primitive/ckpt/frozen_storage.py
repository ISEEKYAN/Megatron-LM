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

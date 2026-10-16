# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contiguous-CP SWA cache: two kernel calls reproduce the CP1 call row for row.

The native joint Q/SWA writer is a CUDA kernel. A float64 stand-in with the same
contract (row-paired Q and KV, one position per row, row-wise outputs; the
window is a per-row function of KV and its position) checks the orchestration:
for every CP rank the split call must return the CP1 window and exactly the CP1
query rows of that rank. The real kernel is compared on GPU.
"""

import pytest
import torch


def _stand_in(query, values, positions, cos_sin, eps, block_size):
    angle = positions.to(torch.float64)[None, :, None, None]
    rotated = query * torch.cos(angle) + query.flip(-1) * torch.sin(angle) + eps
    window = values * (1.0 + positions.to(torch.float64)[None, :, None] / 7.0)
    return rotated, window


@pytest.mark.parametrize('cp', [2, 4])
def test_split_call_matches_cp1_rows(v41_core_te, monkeypatch, cp):
    from megatron.lite.primitive.modules.deployment_math import cache

    monkeypatch.setattr(cache, '_swa_insert', _stand_in)
    generator = torch.Generator().manual_seed(3)
    length, heads, width = 12, 3, 8
    q = torch.randn(1, length, heads, width, generator=generator, dtype=torch.float64)
    kv = torch.randn(1, length, width, generator=generator, dtype=torch.float64)
    positions = torch.arange(length)

    def reference(query, values):
        return query, values

    full_q, full_window = cache.swa_cache(q, kv, positions, None, 1e-6, reference)
    rows = length // cp
    for rank in range(cp):
        start = rank * rows
        local = positions[start : start + rows]
        local_q, window = cache.swa_cache(
            q[:, start : start + rows], kv, local, None, 1e-6, reference, start=start
        )
        assert torch.equal(window, full_window), rank
        assert torch.equal(local_q, full_q[:, start : start + rows]), rank


def test_kernel_call_still_requires_cuda_bf16(v41_core_te):
    from megatron.lite.primitive.modules.deployment_math import cache

    with pytest.raises(ValueError, match='CUDA BF16'):
        cache._swa_insert(
            torch.zeros(1, 2, 1, 8),
            torch.zeros(1, 2, 8),
            torch.arange(2),
            None,
            0.0,
            64,
        )

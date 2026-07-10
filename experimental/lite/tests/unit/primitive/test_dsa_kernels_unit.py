# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch


pytestmark = pytest.mark.mlite


def _patch_cpu_kernel_harness(monkeypatch, sparse_forward):
    from megatron.lite.primitive.kernels import dsa_kernels

    monkeypatch.setattr(dsa_kernels, "_flash_mla_sparse_fwd", sparse_forward)
    monkeypatch.setattr(dsa_kernels, "_get_topk_alignment", lambda: 64)
    monkeypatch.setattr(
        dsa_kernels.torch.cuda.nvtx, "range", lambda _name: nullcontext()
    )
    return dsa_kernels


def _inputs():
    return (
        torch.zeros(2, 4, 8),
        torch.zeros(3, 8),
        torch.zeros(2, 5, dtype=torch.int32),
    )


def test_flash_mla_sparse_default_path_matches_pinned_signature(monkeypatch):
    calls = []

    def pinned_sparse_forward(
        q,
        kv,
        indices,
        sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
        out=None,
    ):
        calls.append((q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out))
        shape = (q.shape[0], q.shape[1])
        return torch.zeros(*shape, d_v), torch.zeros(shape), torch.zeros(shape)

    dsa_kernels = _patch_cpu_kernel_harness(monkeypatch, pinned_sparse_forward)
    q, kv, indices = _inputs()

    out, lse, lse_indexer = dsa_kernels._dsa_fwd_flash_mla(
        q, kv, indices, 0.125, indexer_topk=0
    )

    assert len(calls) == 1
    assert calls[0][1].shape == (3, 1, 8)
    assert calls[0][2].shape == (2, 1, 64)
    assert out.shape == (2, 4, 512)
    assert lse.shape == (2, 4)
    assert lse_indexer is None


def test_flash_mla_sparse_extended_path_preserves_indexer_topk(monkeypatch):
    calls = []

    def extended_sparse_forward(*args, indexer_topk, **kwargs):
        calls.append(indexer_topk)
        q = args[0]
        shape = (q.shape[0], q.shape[1])
        return (
            torch.zeros(*shape, kwargs["d_v"]),
            torch.zeros(shape),
            torch.zeros(shape),
            torch.ones(shape),
        )

    dsa_kernels = _patch_cpu_kernel_harness(monkeypatch, extended_sparse_forward)
    q, kv, indices = _inputs()

    _out, _lse, lse_indexer = dsa_kernels._dsa_fwd_flash_mla(
        q, kv, indices, 0.125, indexer_topk=3
    )

    assert calls == [3]
    assert torch.equal(lse_indexer, torch.ones(2, 4))

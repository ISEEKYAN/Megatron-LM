# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Upstream Core regression for small-head FlashMLA padding and cropping."""

import pytest
import torch


@pytest.fixture(autouse=True)
def core_module(v41_core_te):
    global fused_csa
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        fused_sparse_attention as fused_csa,
    )


def test_flash_mla_adapter_pads_and_restores_small_head_count(monkeypatch):
    seen = {}

    def fake_flash(q, _kv, _indices, _scale, **kwargs):
        seen["q"] = q.detach().clone()
        seen["sink"] = kwargs["attn_sink"].detach().clone()
        rows, heads, _ = q.shape
        out = torch.arange(rows * heads * 3, dtype=q.dtype).reshape(rows, heads, 3)
        lse = torch.arange(rows * heads, dtype=torch.float32).reshape(rows, heads)
        lse_indexer = lse + 1000
        return out, torch.empty(0), lse, lse_indexer

    monkeypatch.setattr(fused_csa, "_flash_mla_sparse_fwd", fake_flash)
    monkeypatch.setattr(fused_csa, "_get_head_padding", lambda _heads: 64)
    monkeypatch.setattr(fused_csa, "_get_topk_alignment", lambda: 1)

    q = torch.randn(3, 2, 8)
    kv = torch.randn(5, 8)
    indices = torch.zeros(3, 4, dtype=torch.int32)
    sink = torch.tensor([0.25, -0.5])

    out, lse, lse_indexer = fused_csa._csa_fwd_flash_mla(
        q, kv, indices, 0.125, d_v=3, attn_sink=sink, indexer_topk=1
    )

    assert seen["q"].shape == (3, 64, 8)
    assert torch.equal(seen["q"][:, :2], q)
    assert torch.count_nonzero(seen["q"][:, 2:]) == 0
    assert torch.equal(seen["sink"][:2], sink)
    assert torch.all(seen["sink"][2:] == float("-inf"))
    assert out.shape == (3, 2, 3)
    assert lse.shape == (3, 2)
    assert lse_indexer.shape == (3, 2)
    expected_out = torch.arange(3 * 64 * 3, dtype=q.dtype).reshape(3, 64, 3)[:, :2]
    expected_lse = torch.arange(3 * 64, dtype=torch.float32).reshape(3, 64)[:, :2]
    assert torch.equal(out, expected_out)
    assert torch.equal(lse, expected_lse)
    assert torch.equal(lse_indexer, expected_lse + 1000)

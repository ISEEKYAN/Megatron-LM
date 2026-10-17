# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Standard sparse contracts; CPU mocks do not certify the CUDA kernels."""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def modules(v41_core_te):
    from megatron.lite.primitive.modules.attention.standard_sparse import (
        standard_indexer_scores,
        standard_sparse_attention,
    )

    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        fused_sparse_attention as core,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        packed_sparse_attention as packed,
    )

    return core, packed, standard_sparse_attention, standard_indexer_scores


@pytest.mark.parametrize('ratio', [0, 1, 2])
@pytest.mark.parametrize('batch', [1, 2])
def test_sparse_addresses_preserve_dense_document_outputs_and_gradients(
    modules, monkeypatch, ratio, batch
):
    _, packed, attention, _ = modules
    torch.manual_seed(19)
    rows, heads, dim, window = 7, 2, 512, 4
    columns = rows // ratio if ratio else 0
    q = (torch.randn(batch, rows, heads, dim) * 0.05).bfloat16().requires_grad_()
    w = (torch.randn(batch, rows, dim) * 0.05).bfloat16().requires_grad_()
    c = (torch.randn(batch, columns, dim) * 0.05).bfloat16().requires_grad_()
    sink = torch.randn(heads, requires_grad=True)
    qr, wr, cr, sr = [t.detach().clone().requires_grad_() for t in (q, w, c, sink)]
    selected = None
    if ratio:
        visible = (torch.arange(rows) + 1) // ratio
        selected = torch.arange(columns)[None].expand(rows, -1)
        selected = torch.where(selected < visible[:, None], selected, -1)
        selected = selected[None].expand(batch, -1, -1).int()

    def reference_backend(query, kv, bias, indices, scale, topk_length):
        # Substitute only Core's CUDA implementation; consume actual A2 indices.
        assert indices.dtype == topk_length.dtype == torch.int32
        assert indices.shape[-1] >= 128
        gathered = kv[indices.clamp_min(0).long()]
        scores = torch.einsum('shd,std->sht', query.float(), gathered.float()) * scale
        scores = scores.masked_fill((indices < 0)[:, None], -torch.inf)
        p = torch.cat((scores, bias[None, :, None].expand(rows, -1, 1)), -1).softmax(
            -1
        )[..., :-1]
        return (
            torch.einsum('sht,std->shd', p, gathered.float()).to(query.dtype).flatten(1)
        )

    monkeypatch.setattr(packed, 'csa_sparse_attn', reference_backend)
    actual = attention(
        q, w, c if ratio else None, sink, selected, ratio=ratio, window_size=window
    )
    kv = torch.cat((wr, cr), 1)
    row = torch.arange(rows)
    mask = (row[None] <= row[:, None]) & (row[None] > row[:, None] - window)
    mask = mask[None].expand(batch, -1, -1)
    if ratio:
        mask = torch.cat(
            (
                mask,
                (
                    torch.arange(columns)[None, None]
                    < ((row + 1) // ratio)[None, :, None]
                ).expand(batch, -1, -1),
            ),
            -1,
        )
    logits = torch.einsum('bshd,btd->bsht', qr.float(), kv.float()) * dim**-0.5
    logits = logits.masked_fill(~mask[:, :, None], -torch.inf)
    p = torch.cat(
        (logits, sr[None, None, :, None].expand(batch, rows, -1, 1)), -1
    ).softmax(-1)[..., :-1]
    expected = torch.einsum('bsht,btd->bshd', p, kv.float()).to(q.dtype)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for a, b in [(q.grad, qr.grad), (w.grad, wr.grad), (sink.grad, sr.grad)]:
        torch.testing.assert_close(a, b, rtol=5e-2, atol=5e-2)
    if ratio:
        torch.testing.assert_close(c.grad, cr.grad, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize('ratio', [1, 2])
def test_indexer_keeps_final_scores_for_existing_candidate_policy(
    modules, monkeypatch, ratio
):
    core, _, _, indexer = modules
    torch.manual_seed(23)
    q = torch.randn(2, 7, 32, 128).bfloat16().requires_grad_()
    k = torch.randn(2, 4, 128).bfloat16().requires_grad_()
    w = torch.randn(2, 7, 32).bfloat16().requires_grad_()
    calls = []

    def final_score_kernel(qi, ki, wi, **kwargs):
        assert qi.ndim == ki.ndim == 3 and ki.shape[1] == 1
        assert qi.is_contiguous() and ki.is_contiguous() and wi.is_contiguous()
        assert kwargs['ratio'] == ratio
        assert kwargs['cu_seqlens_q'].tolist() == [0, 7]
        assert kwargs['cu_seqlens_k'].tolist() == [0, 4]
        assert kwargs['max_seqlen_q'] == 7 and kwargs['max_seqlen_k'] == 4
        calls.append(True)
        scores = (
            torch.einsum('shd,td->sht', qi.float(), ki[:, 0].float()).relu()
            * wi.float()[:, :, None]
        ).sum(1)
        return {'scores': scores}

    monkeypatch.setattr(
        core, '_DSA', SimpleNamespace(indexer_forward_wrapper=final_score_kernel)
    )
    actual = indexer(q, k, w, ratio=ratio)
    expected = (
        torch.einsum('bshd,btd->bsht', q.float(), k.float()).relu()
        * w.float()[:, :, :, None]
    ).sum(2)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 7, 4) and not actual.requires_grad and len(calls) == 2


def test_empty_indexer_keys_do_not_launch_backend(modules, monkeypatch):
    core, _, _, indexer = modules
    monkeypatch.setattr(
        core, '_ensure_dsa_namespace', lambda: pytest.fail('empty launch')
    )
    actual = indexer(
        torch.empty(1, 3, 32, 128),
        torch.empty(1, 0, 128),
        torch.empty(1, 3, 32),
        ratio=2,
    )
    assert actual.shape == (1, 3, 0) and actual.dtype == torch.float32


def test_standard_sparse_rejects_unsupported_dimension(modules):
    _, _, attention, _ = modules
    with pytest.raises(ValueError, match='head dimension 512'):
        attention(
            torch.zeros(1, 3, 2, 4),
            torch.zeros(1, 3, 4),
            None,
            torch.zeros(2),
            None,
            ratio=0,
            window_size=4,
        )

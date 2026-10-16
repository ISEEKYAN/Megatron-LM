# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sparse attention reference VJP replayed over query-row chunks."""
import pytest
import torch


def _problem(seed):
    torch.manual_seed(seed)
    batch, rows, heads, dim, kv_rows, width = 2, 37, 4, 16, 23, 8
    query = torch.randn(batch, rows, heads, dim, dtype=torch.bfloat16)
    values = torch.randn(batch, kv_rows, dim, dtype=torch.bfloat16)
    sink = torch.randn(heads, dtype=torch.float32)
    indices = torch.randint(-1, kv_rows, (batch, rows, width))
    indices[..., 0] = torch.arange(rows) % kv_rows  # at least one valid slot
    offsets = torch.arange(batch)[:, None, None] * kv_rows
    flat = torch.where(indices >= 0, indices + offsets, -1)
    largest = []

    def reference_rows(q, v, b, start, stop):
        selected = flat[:, start:stop]
        gathered = v.flatten(0, 1)[selected.clamp_min(0).long()]
        largest.append(gathered.numel())
        logits = torch.einsum('bshd,bstd->bsht', q.float(), gathered.float()) * 0.25
        logits = logits.masked_fill((selected < 0).unsqueeze(2), -torch.inf)
        probabilities = torch.cat(
            [logits, b.expand(batch, stop - start, -1).unsqueeze(-1)], -1
        ).softmax(-1)[..., :-1]
        return torch.einsum('bsht,bstd->bshd', probabilities, gathered.float()).to(
            q.dtype
        )

    return query, values, sink, reference_rows, largest


def _chunked_grads(seed, chunk):
    from megatron.lite.primitive.modules.deployment_math.sparse import _ChunkedRowVJP

    query, values, sink, reference_rows, largest = _problem(seed)
    leaves = [t.clone().requires_grad_() for t in (query, values, sink)]
    rows = query.shape[1]

    def visible(q, v, b):
        with torch.no_grad():
            return reference_rows(q, v, b, 0, rows)

    output = _ChunkedRowVJP.apply(visible, reference_rows, chunk, *leaves)
    upstream = torch.randn(output.shape, generator=torch.Generator().manual_seed(7))
    output.backward(upstream.to(output.dtype))
    return [t.grad for t in leaves], upstream, largest


def _full_grads(seed, upstream, fp32_values=False):
    query, values, sink, reference_rows, _ = _problem(seed)
    if fp32_values:
        values = values.float()
    leaves = [t.clone().requires_grad_() for t in (query, values, sink)]
    output = reference_rows(*leaves, 0, query.shape[1])
    output.backward(upstream.to(output.dtype))
    return [t.grad for t in leaves]


@pytest.mark.parametrize("chunk", [1, 5, 16, 37, 1024])
def test_chunked_vjp_matches_full_reference(chunk):
    chunked, upstream, _ = _chunked_grads(3, chunk)
    # FP32-accumulated truth: same forward values, values gradient unrounded.
    truth = _full_grads(3, upstream, fp32_values=True)
    torch.testing.assert_close(chunked[0], truth[0], rtol=0, atol=0)
    torch.testing.assert_close(
        chunked[1].float(), truth[1].bfloat16().float(), rtol=0, atol=0
    )
    torch.testing.assert_close(chunked[2], truth[2], rtol=1e-6, atol=1e-6)
    # Against the original single-pass reference (BF16 index accumulation).
    original = _full_grads(3, upstream)
    for actual, expected in zip(chunked, original):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=5e-2, atol=5e-2
        )
    # The query gradient is row-local, so chunking cannot change it.
    assert torch.equal(chunked[0], original[0])


def test_chunked_vjp_is_deterministic():
    first, _, _ = _chunked_grads(5, 8)
    second, _, _ = _chunked_grads(5, 8)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_chunked_vjp_bounds_gathered_rows():
    _, _, largest = _chunked_grads(9, 8)
    _, _, unchunked = _chunked_grads(9, 1024)
    # Forward call plus per-chunk replays; the largest replay gather is 8/37 of
    # the full one.
    assert max(largest[1:]) * 4 < max(unchunked[1:])

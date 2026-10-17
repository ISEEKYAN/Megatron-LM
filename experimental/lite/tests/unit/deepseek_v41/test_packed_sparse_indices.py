# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core-built packed metadata preserves the deployed per-document slot order."""

import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_imports(v41_core_te):
    global PackedCPGeometry, packed_sparse_metadata, sparse_attention_from_indices
    from megatron.lite.primitive.modules.attention.packed_geometry import (
        PackedCPGeometry,
    )
    from megatron.lite.primitive.modules.attention.packed_indices import (
        packed_sparse_metadata,
    )
    from megatron.lite.primitive.modules.deployment_math.sparse import (
        sparse_attention_from_indices,
    )


def core_compressed_metadata(geometry, ratio):
    if not ratio:
        return None, 0
    begin, end = geometry.lookback(ratio - 1)
    _, _, _, cu, _ = geometry.compressor_input(
        torch.zeros(geometry.width, 1),
        torch.zeros(end - begin, 1),
        ratio,
        halo=ratio - 1,
        capacity=geometry.width // ratio + 1,
    )
    return cu, int(cu[-1])


def old_document_slots(geometry, ratio, selected, window, width, lookback_rows=None):
    """Old compact_sparse_slots order, translated to the shared physical KV."""
    bounds = [0]
    for a, b in zip(geometry.boundaries, geometry.boundaries[1:]):
        bounds.append(bounds[-1] + (b - a) // ratio if ratio else 0)
    total = bounds[-1]
    left = (
        max(0, geometry.global_start - window + 1)
        if lookback_rows is None
        else geometry.global_start - lookback_rows
    )
    expected = torch.full((geometry.width, width), -1, dtype=torch.int32)
    counts = torch.zeros(geometry.width, dtype=torch.int32)
    for doc, (a, b) in enumerate(zip(geometry.boundaries, geometry.boundaries[1:])):
        for row in range(
            max(a, geometry.global_start),
            min(b, geometry.global_start + geometry.width),
        ):
            q = row - geometry.global_start
            visible = (row - a + 1) // ratio if ratio else 0
            ids = list(range(visible)) if selected is None else selected[q].tolist()
            chosen = [bounds[doc] + k for k in ids if 0 <= k < visible]
            chosen += [
                total + k - left for k in range(max(a, row - window + 1), row + 1)
            ]
            expected[q, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
            counts[q] = len(chosen)
    return expected, counts


@pytest.mark.parametrize(
    'boundaries,cp', [((0, 3, 8, 9), 2), ((0, 1, 2), 4), ((0, 4, 11), 1)]
)
@pytest.mark.parametrize(
    'ratio,use_selector', [(0, False), (1, False), (1, True), (2, False), (2, True)]
)
def test_core_indices_match_document_slots_and_attention(
    boundaries, cp, ratio, use_selector
):
    """No cross-document keys, reordered valid slots, or visible tail padding."""
    torch.manual_seed(224)
    window, width = 3, 16
    for rank in range(cp):
        geometry = PackedCPGeometry(boundaries, rank, cp)
        real_rows = max(0, min(geometry.width, geometry.total - geometry.global_start))
        selected = None
        if use_selector:
            selected = torch.full((real_rows, 5), -1, dtype=torch.int32)
            for i, pos in enumerate(geometry.positions[:real_rows].tolist()):
                visible = (pos + 1) // ratio
                n = min(visible, 3)
                if n:
                    selected[i, :n] = torch.arange(n - 1, -1, -1)
        cu, total = core_compressed_metadata(geometry, ratio)
        actual, lengths = packed_sparse_metadata(
            geometry,
            selected,
            ratio=ratio,
            window_size=window,
            width=width,
            compressed_cu_seqlens=cu,
            compressed_rows=total,
        )
        expected, counts = old_document_slots(geometry, ratio, selected, window, width)
        expected, counts = expected[:real_rows], counts[:real_rows]
        assert torch.equal(actual, expected)
        assert torch.equal(lengths, counts)
        lookback = min(geometry.global_start, window - 1)
        values = torch.randn(1, total + lookback + real_rows, 4, requires_grad=True)
        query = torch.randn(1, real_rows, 2, 4, requires_grad=True)
        sink = torch.randn(2, requires_grad=True)
        output = sparse_attention_from_indices(
            query, values, sink, actual.unsqueeze(0), lengths.unsqueeze(0), scale=0.5
        )
        pieces = []
        for row in range(real_rows):
            indices = expected[row : row + 1].unsqueeze(0)
            gathered = values.flatten(0, 1)[indices.clamp_min(0).long()]
            logits = (
                torch.einsum(
                    'bshd,bstd->bsht', query[:, row : row + 1].float(), gathered.float()
                )
                * 0.5
            )
            logits = logits.masked_fill((indices < 0).unsqueeze(2), -torch.inf)
            probs = torch.cat(
                [logits, sink.expand(1, 1, -1).unsqueeze(-1)], -1
            ).softmax(-1)[..., :-1]
            pieces.append(torch.einsum('bsht,bstd->bshd', probs, gathered.float()))
        expected_output = (
            torch.cat(pieces, 1) if pieces else query.new_zeros(query.shape)
        )
        assert torch.equal(output, expected_output)
        assert output.shape[1] == real_rows
        output.sum().backward()
        assert all(
            t.grad is not None and torch.isfinite(t.grad).all()
            for t in (query, values, sink)
        )


def test_slot_translation_preserves_selector_duplicates_and_invalid_tail():
    geometry = PackedCPGeometry((0, 4, 7))
    selected = torch.tensor(
        [
            [-1, -1, -1],
            [0, 0, -1],
            [1, 0, -1],
            [2, 0, -1],
            [-1, -1, -1],
            [0, 0, -1],
            [1, 0, -1],
        ],
        dtype=torch.int32,
    )
    cu, total = core_compressed_metadata(geometry, 1)
    indices, lengths = packed_sparse_metadata(
        geometry,
        selected,
        ratio=1,
        window_size=2,
        width=8,
        compressed_cu_seqlens=cu,
        compressed_rows=total,
    )
    expected, counts = old_document_slots(geometry, 1, selected, 2, 8)
    assert torch.equal(indices, expected)
    assert torch.equal(lengths, counts)


def test_missing_left_window_is_rejected():
    geometry = PackedCPGeometry((0, 7), 1, 2)
    with pytest.raises(ValueError, match='missing left-neighbor'):
        packed_sparse_metadata(geometry, ratio=1, window_size=3, lookback_rows=0)


def test_missing_core_compressed_metadata_is_rejected():
    geometry = PackedCPGeometry((0, 7))
    with pytest.raises(ValueError, match='must come from Core'):
        packed_sparse_metadata(geometry, ratio=1, window_size=3)


def test_state_host_boundaries_exclude_padding_and_zero_rows():
    geometry = PackedCPGeometry((0, 1, 2), 3, 4)
    indices, lengths = packed_sparse_metadata(
        geometry,
        torch.empty(0, 2, dtype=torch.int32),
        ratio=1,
        window_size=2,
        kv_bounds=(0, 1, 2),
        query_bounds=(0, 0, 0),
    )
    assert indices.shape[0] == lengths.shape[0] == 0
    with pytest.raises(ValueError, match='Query boundaries'):
        packed_sparse_metadata(
            geometry,
            ratio=1,
            window_size=2,
            kv_bounds=(0, 1, 2),
            query_bounds=(0, 0, 1),
        )


@pytest.mark.parametrize('rank', [0, 1])
def test_fixed_core_halo_matches_full_document_gather(rank):
    geometry = PackedCPGeometry((0, 2, 7), rank, 2)
    window, halo, width, ratio = 4, 4, 16, 2
    real_rows = min(geometry.width, geometry.total - geometry.global_start)
    cu, total = core_compressed_metadata(geometry, ratio)
    actual, lengths = packed_sparse_metadata(
        geometry,
        ratio=ratio,
        window_size=window,
        width=width,
        lookback_rows=halo,
        compressed_cu_seqlens=cu,
        compressed_rows=total,
    )
    expected, expected_lengths = old_document_slots(
        geometry, ratio, None, window, width, lookback_rows=halo
    )
    expected, expected_lengths = expected[:real_rows], expected_lengths[:real_rows]
    assert torch.equal(actual, expected)
    assert torch.equal(lengths, expected_lengths)
    torch.manual_seed(928)
    full_window = torch.randn(1, geometry.total, 4)
    compressed = torch.randn(1, total, 4)
    query = torch.randn(1, real_rows, 2, 4)
    sink = torch.randn(2)
    start = geometry.global_start
    left = start - halo
    buffer = torch.zeros(1, halo + real_rows, 4)
    first = max(0, left)
    buffer[:, first - left :] = full_window[:, first : start + real_rows]
    physical = torch.cat((compressed, buffer), 1)
    output = sparse_attention_from_indices(
        query, physical, sink, actual[None], lengths[None], scale=0.5
    )
    global_indices = torch.where(expected >= total, expected + left, expected)
    wanted = sparse_attention_from_indices(
        query,
        torch.cat((compressed, full_window), 1),
        sink,
        global_indices[None],
        expected_lengths[None],
        scale=0.5,
    )
    assert torch.equal(output, wanted)

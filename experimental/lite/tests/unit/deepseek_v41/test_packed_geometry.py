# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CP-A A0: the whole-slice packed contract equals the per-document path.

ContiguousCPSequence.document (rows a rank owns of each document), the CSA
positions and the compressor cutoff (complete groups from the document start,
RoPE position = group start) are the reference. Positions and compressor
groups come from Core's packed CSA helpers; for every rank, document and row
they must give the same rows, positions and compressed groups.
"""
import pytest
import torch
from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence
from megatron.lite.primitive.modules.attention.packed_geometry import PackedCPGeometry

CASES = [
    ((5,), 1),
    ((1023, 2, 1023), 1),
    ((1023, 2, 1023), 2),
    ((2500, 1596), 2),
    ((7, 1, 3, 9, 2), 4),
    ((3, 3), 4),  # more ranks than some documents have rows
    ((1,), 3),  # every rank but the first is pad only
    ((17, 1, 1, 30), 3),
]


def _bounds(lengths):
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return tuple(out)


def _reference_rows(lengths, rank, size):
    """Per-document path: global rows and positions in packing order."""
    bounds = _bounds(lengths)
    seq = ContiguousCPSequence(bounds[-1], rank, size)
    rows, positions = [], []
    for b, e in zip(bounds, bounds[1:]):
        doc = seq.document(b, e)
        local = list(range(doc.start, doc.start + doc.local_length))
        rows += [b + p for p in local]
        positions += local  # CSA: positions = arange(start, start + length)
    return rows, positions


@pytest.mark.parametrize('lengths,size', CASES)
def test_rows_and_positions_match_per_document_path(lengths, size):
    bounds = _bounds(lengths)
    for rank in range(size):
        g = PackedCPGeometry(bounds, rank, size)
        rows, positions = _reference_rows(lengths, rank, size)
        valid = g.valid
        local = torch.arange(g.width) + g.global_start
        assert local[valid].tolist() == rows
        assert g.positions[valid].tolist() == positions
        assert g.width * size == g.padded_total >= g.total
        # Pad rows only after every real row.
        assert valid.tolist() == sorted(valid.tolist(), reverse=True)
        for d, (b, e) in enumerate(zip(bounds, bounds[1:])):
            doc = ContiguousCPSequence(bounds[-1], rank, size).document(b, e)
            start, stop = g.owned(d)
            assert (start - b, stop - start) == (doc.start, doc.local_length)


@pytest.mark.parametrize('lengths,size', CASES)
def test_slice_matches_contiguous_cp_sequence(lengths, size):
    bounds = _bounds(lengths)
    x = torch.arange(bounds[-1] * 3).reshape(1, bounds[-1], 3)
    for rank in range(size):
        g = PackedCPGeometry(bounds, rank, size)
        reference = ContiguousCPSequence(bounds[-1], rank, size).slice(x)
        local = g.slice(x)
        assert local.shape[1] == g.width
        assert torch.equal(local[:, : reference.shape[1]], reference)
        assert not local[:, reference.shape[1] :].any()


@pytest.mark.parametrize('lengths,size', CASES)
@pytest.mark.parametrize('ratio', [1, 2])
def test_compressor_groups_match_per_document_compression(lengths, size, ratio):
    """Each complete group is made once, by the rank holding its last token."""
    bounds = _bounds(lengths)
    halo = ratio - 1
    full = torch.arange(bounds[-1], dtype=torch.float32)
    made = {}
    for rank in range(size):
        g = PackedCPGeometry(bounds, rank, size)
        hidden = g.slice(full.view(1, -1)).view(-1, 1)
        start, _ = g.lookback(halo)
        boundary = full[start : g.global_start].view(-1, 1)
        capacity = g.width // ratio + 1
        gathered, group_ids, positions, comp_cu, row_map = g.compressor_input(
            hidden, boundary, ratio, halo=halo, capacity=capacity
        )
        # Compressor cutoff: floor(len / ratio) complete groups per document.
        assert comp_cu.tolist() == list(_bounds([n // ratio for n in lengths]))
        for slot in range(capacity):
            group = int(group_ids[slot])
            if group < 0:
                continue
            rows = gathered[slot * ratio : (slot + 1) * ratio, 0].long().tolist()
            doc = max(d for d in range(len(lengths)) if bounds[d] <= rows[0])
            assert rows == [bounds[doc] + group * ratio + i for i in range(ratio)]
            assert int(positions[slot]) == group * ratio  # RoPE at the group start
            last = rows[-1]
            assert g.global_start <= last < g.global_start + g.width
            logical = comp_cu[doc] + group
            assert int(row_map[logical]) == rank * capacity + slot
            assert int(logical) not in made
            made[int(logical)] = rank
    assert sorted(made) == list(range(_bounds([n // ratio for n in lengths])[-1]))


def test_lookback_from_cu_seqlens_and_invalid_inputs():
    g = PackedCPGeometry((0, 10, 30), 1, 2)
    assert (g.width, g.global_start, g.lookback(4)) == (15, 15, (11, 15))
    assert PackedCPGeometry((0, 10, 30), 0, 2).lookback(4) == (0, 0)
    cu = torch.tensor([0, 7, 9, 20], dtype=torch.int32)
    for rank in range(2):
        h = PackedCPGeometry.from_cu_seqlens(cu, rank, 2)
        assert h.device == torch.device('cpu')
        assert h == PackedCPGeometry((0, 7, 9, 20), rank, 2)
        assert torch.equal(h.cu_seqlens, cu)
        assert h.cu_seqlens_padded.tolist() == [0, 7, 9, 20]
    for bad in ((0,), (1, 4), (0, 3, 3)):
        with pytest.raises(ValueError):
            PackedCPGeometry(bad)
    with pytest.raises(ValueError):
        PackedCPGeometry((0, 4), 2, 2)

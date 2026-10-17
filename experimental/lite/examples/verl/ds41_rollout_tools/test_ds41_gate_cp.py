"""CPU self-test of ds41_gate.py's CP assignment: rows, loss share, merge.

The rows follow the CP-A layout contract (whole-pack contiguous split, tail
padding); they are checked against PackedCPGeometry (padded width) and the
protocol's ContiguousCPSequence (real rows only) where the tree has them.
"""

import pytest
import torch
from ds41_gate import cp_loss, cp_rows, fragment, merge_fragments

CASES = [(7, 1), (7, 2), (8, 2), (9, 4), (310, 2), (215, 4)]


def _real(total, rank, cp):
    width = -(-total // cp)
    return max(min(total - rank * width, width), 0)


@pytest.mark.parametrize('total,cp', CASES)
def test_ranks_cover_every_row_once_padded_or_not(total, cp):
    width = -(-total // cp)
    for padded in (False, True):
        seen = []
        for rank in range(cp):
            length = width if padded else _real(total, rank, cp)
            rows = cp_rows(total, rank, cp, length)
            assert rows.shape == (length,)
            valid = rows[rows >= 0].tolist()
            assert valid == list(range(rank * width, rank * width + len(valid)))
            assert (rows[len(valid) :] == -1).all()  # padding only at the tail
            seen += valid
        assert sorted(seen) == list(range(total))


def test_unexpected_local_length_is_refused():
    with pytest.raises(RuntimeError, match='expected'):
        cp_rows(9, 0, 4, 2)


@pytest.mark.parametrize('total,cp', CASES)
def test_rows_match_the_runtime_slicings(total, cp):
    ids = torch.arange(1, total + 1)[None]  # 0 marks padding
    slicers = []
    try:
        from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

        slicers.append(lambda r: ContiguousCPSequence(total, r, cp).slice(ids))
    except ImportError:
        pass
    try:
        from megatron.lite.primitive.modules.attention.packed_geometry import (
            PackedCPGeometry,
        )

        slicers.append(
            lambda r: PackedCPGeometry.from_cu_seqlens(
                torch.tensor([0, total]), r, cp
            ).slice(ids)
        )
    except ImportError:
        pass
    if not slicers:
        pytest.skip('no CP slicing in this tree')
    for slicer in slicers:
        for rank in range(cp):
            local = slicer(rank)[0] - 1
            assert torch.equal(cp_rows(total, rank, cp, local.numel()), local)


def test_cp_loss_shares_average_to_the_cp1_mean():
    total, cp = 9, 4
    full = torch.randn(total)
    cp1 = cp_loss(full, cp_rows(total, 0, 1, total), total, 1)
    assert torch.equal(cp1, -full.sum() / float(total))
    width = -(-total // cp)
    shares = []
    for rank in range(cp):
        rows = cp_rows(total, rank, cp, width)
        local = torch.where(rows >= 0, full[rows.clamp(min=0)], torch.tensor(123.0))
        shares.append(cp_loss(local, rows, total, cp))
    # The runtime averages over CP; padding rows (123.0) must not count.
    assert torch.allclose(torch.stack(shares).sum() / cp, cp1)


def test_merge_restores_row_order_and_rejects_gaps_or_disagreement():
    total, cp = 9, 4
    full = torch.arange(total, dtype=torch.float32)
    width = -(-total // cp)
    parts = []
    for rank in reversed(range(cp)):
        rows = cp_rows(total, rank, cp, width)
        parts.append({3: fragment(rows, full[rows.clamp(min=0)])})
    assert merge_fragments(parts) == {3: full.tolist()}
    without_rank0 = parts[:-1]  # parts are in reverse rank order
    with pytest.raises(RuntimeError, match='missing'):
        merge_fragments(without_rank0)
    bad = {3: dict(rows=[0], logprobs=[99.0])}
    with pytest.raises(RuntimeError, match='disagree'):
        merge_fragments(parts + [bad])

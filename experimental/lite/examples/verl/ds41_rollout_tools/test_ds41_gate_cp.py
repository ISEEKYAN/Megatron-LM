"""CPU self-test of ds41_gate.py's CP assignment: slicing rows, loss share, merge."""

import pytest
import torch
from ds41_gate import cp_loss, cp_rows, fragment, merge_fragments


@pytest.mark.parametrize(
    'total,cp', [(7, 1), (7, 2), (8, 2), (9, 4), (310, 2), (215, 4)]
)
def test_ranks_cover_every_row_once_with_tail_padding(total, cp):
    width = -(-total // cp)
    seen = []
    for rank in range(cp):
        rows = cp_rows(total, rank, cp)
        assert rows.shape == (width if cp > 1 else total,)
        valid = rows[rows >= 0]
        # Contiguous ownership: rank r holds [r * width, (r + 1) * width).
        assert valid.tolist() == list(
            range(rank * width, min((rank + 1) * width, total))
        )
        seen += valid.tolist()
    assert sorted(seen) == list(range(total))


def test_cp_loss_shares_sum_to_the_cp1_mean_times_cp():
    total, cp = 9, 4
    full = torch.randn(total)
    cp1 = cp_loss(full, cp_rows(total, 0, 1), total, 1)
    assert torch.equal(cp1, -full.sum() / float(total))
    shares = []
    for rank in range(cp):
        rows = cp_rows(total, rank, cp)
        local = torch.where(rows >= 0, full[rows.clamp(min=0)], torch.zeros(()))
        local[rows < 0] = 123.0  # padding rows must not count
        shares.append(cp_loss(local, rows, total, cp))
    # The runtime averages over CP: the mean of the shares is the CP1 loss.
    assert torch.allclose(torch.stack(shares).sum() / cp, cp1)


def test_merge_restores_row_order_and_rejects_gaps_or_disagreement():
    total, cp = 9, 4
    full = torch.arange(total, dtype=torch.float32)
    parts = []
    for rank in reversed(range(cp)):
        rows = cp_rows(total, rank, cp)
        local = full[rows.clamp(min=0)]
        parts.append({3: fragment(rows, local)})
    parts.append({3: fragment(cp_rows(total, 0, cp), full[cp_rows(total, 0, cp)])})
    assert merge_fragments(parts) == {3: full.tolist()}
    with pytest.raises(RuntimeError, match='missing'):
        merge_fragments(parts[1:-1])
    bad = {3: dict(rows=[0], logprobs=[99.0])}
    with pytest.raises(RuntimeError, match='disagree'):
        merge_fragments(parts + [bad])


def test_packed_geometry_and_protocol_slicing_agree():
    """CP-A's PackedCPGeometry and ContiguousCPSequence slice the same rows."""
    packed = pytest.importorskip(
        'megatron.lite.primitive.modules.attention.packed_geometry'
    )
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    for total, cp in ((7, 2), (9, 4), (310, 2)):
        ids = torch.arange(1, total + 1)[None]
        for rank in range(cp):
            a = packed.PackedCPGeometry.from_cu_seqlens(
                torch.tensor([0, total]), rank, cp
            ).slice(ids, seq_dim=1)
            b = ContiguousCPSequence(total, rank, cp).slice(ids, seq_dim=1)
            assert torch.equal(a, b)

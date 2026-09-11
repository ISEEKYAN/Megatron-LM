import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import candidates


def test_pool_membership_newest_pin_and_outside_winner():
    scores = torch.tensor([[[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -0.5]]])
    pool = candidates.candidate_blocks(scores, 9, topk_blocks=2, block_size=2)
    assert pool.tolist() == [
        [[True, True, False, False, False, False, False, False, True]]
    ]
    later = torch.tensor([[[1.0, 2.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]]])
    assert candidates.select_positions(
        later, 9, 2, offset=11, candidates=pool
    ).tolist() == [[[12, 19]]]
    assert candidates.select_positions(later, 9, 2, offset=11).tolist() == [[[13, 19]]]


def test_visibility_empty_prefix_and_unreachable_blocks():
    scores = torch.tensor([[[8.0, 7.0, 6.0, 5.0, 4.0], [8.0, 7.0, 6.0, 5.0, 4.0]]])
    lengths = torch.tensor([[0], [3]])
    pool = candidates.candidate_blocks(scores, lengths, topk_blocks=4, block_size=2)
    assert pool.tolist() == [[[False] * 5, [True, True, True, True, False]]]
    assert candidates.select_positions(
        scores, lengths, 4, candidates=pool
    ).tolist() == [[[-1] * 4, [0, 1, 2, -1]]]
    empty = scores[..., :0]
    assert candidates.candidate_blocks(empty, 0).shape == empty.shape
    assert candidates.select_positions(empty, 0, 2).shape == empty.shape


def test_reject_pretraining_and_invalid_counts():
    with pytest.raises(ValueError):
        candidates.candidate_blocks(torch.ones(1, 3), 3, phase='pretraining')
    with pytest.raises(ValueError):
        candidates.candidate_blocks(torch.ones(1, 3), 3, block_size=0)

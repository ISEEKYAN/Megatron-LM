import torch
from megatron.lite.model.deepseek_v41.lite.engram import Engram, NgramHash
from torch import nn


def hash_module():
    # Current/previous/second previous multipliers 3,5,7; separate prime buckets.
    return NgramHash(
        [0, 1, 2, 3], 0, torch.tensor([[3, 5, 7]]), torch.tensor([[[11], [13]]])
    )


def test_known_hashes_offsets_and_complete_resets():
    hasher = hash_module()
    tokens = torch.tensor([[1, 2, 3]])
    # 2grams: 3^0=3, 6^5=3, 9^10=3. 3grams: 3,3,3^7=4.
    assert hasher(tokens).tolist() == [[[[3, 14]], [[3, 14]], [[3, 15]]]]
    mask = torch.tensor([[True, False, True]])
    assert hasher(tokens, token_mask=mask).tolist() == [
        [[[3, 14]], [[0, 11]], [[9, 20]]]
    ]
    assert hasher(tokens, cu_seqlens=torch.tensor([0, 2, 3])).tolist() == [
        [[[3, 14]], [[3, 14]], [[9, 20]]]
    ]


def test_engram_per_copy_gate_mask_and_gradient():
    table = nn.Embedding(3, 2)
    projection = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        table.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        projection.weight.copy_(torch.tensor([[1.0, -1.0], [-2.0, 1.0], [3.0, 4.0]]))
    module = Engram(1, 2, table, projection)
    hidden = torch.tensor([[[[2.0], [-3.0]], [[4.0], [5.0]]]], requires_grad=True)
    ids = torch.tensor([[[0], [1]]])
    mask = torch.tensor([[True, False]])
    actual = module(hidden, ids, mask)
    eps = module.eps
    dots = torch.tensor(
        [
            2.0 / ((4.0 + eps) * (1.0 + eps)) ** 0.5,
            6.0 / ((9.0 + eps) * (4.0 + eps)) ** 0.5,
        ]
    )
    gates = torch.sigmoid(dots.sqrt())
    expected = hidden.detach().clone()
    expected[0, 0, :, 0] += 3 * gates
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(hidden.grad[0, 1], torch.ones(2, 1))
    assert torch.count_nonzero(table.weight.grad[1:]) == 0
    assert torch.count_nonzero(table.weight.grad[0]) > 0
    assert torch.count_nonzero(module.q_weight.grad) == 2
    assert torch.count_nonzero(module.k_weight.grad) == 2

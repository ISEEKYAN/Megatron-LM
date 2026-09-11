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


def test_fp8_table_freeze_switch_master_gradient_and_scale_publication():
    from megatron.lite.model.deepseek_v41.lite.engram import EngramTable
    from megatron.lite.primitive.quantization.ds41_fp8 import quantize_swa

    source = torch.linspace(-2, 2, 3 * 32).reshape(3, 32)
    quantized = quantize_swa(source)
    frozen = EngramTable(quantized.values, quantized.scale, trainable=False)
    live = EngramTable(quantized.values, quantized.scale, trainable=True)
    ids = torch.tensor([[0, 0, 2]])
    assert not list(frozen.parameters())
    assert 'master' not in frozen.state_dict()
    assert all(t.dtype != torch.float32 for t in frozen.state_dict().values())
    assert live.master.dtype == torch.float32 and live.master.requires_grad
    torch.testing.assert_close(frozen(ids), live(ids), atol=0, rtol=0)
    live(ids).float().sum().backward()
    assert live.master.grad.dtype == torch.float32
    torch.testing.assert_close(
        live.master.grad, torch.tensor([2.0, 0.0, 1.0])[:, None].expand(3, 32)
    )
    before = live.scale.float().clone()
    with torch.no_grad():
        live.master.mul_(16)
    live.refresh_storage()
    assert (live.scale.float() > before).all()
    torch.testing.assert_close(live(ids).float(), frozen(ids).float() * 16)


def test_table_module_dtype_conversion_preserves_storage_and_master():
    from megatron.lite.model.deepseek_v41.lite.engram import EngramTable
    from megatron.lite.primitive.quantization.ds41_fp8 import quantize_swa

    q = quantize_swa(torch.linspace(-1, 1, 64).reshape(2, 32))
    table = EngramTable(q.values, q.scale, trainable=True)
    with torch.no_grad():
        table.master.add_(0.000013)
    original = table.master.detach().clone()
    table.bfloat16()
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    assert table.master.dtype == torch.float32
    torch.testing.assert_close(table.master, original, atol=0, rtol=0)

import pytest
import torch

from megatron.lite.primitive.modules.engram_lookup import RowLookup
from megatron.lite.model.deepseek_v41.lite.engram import EngramTable, ShardedEngramTable
from megatron.lite.primitive.quantization.ds41_fp8 import quantize_swa


def test_raw_bytes_and_order_match_published_rows():
    q = quantize_swa(torch.arange(7 * 256).reshape(7, 256).float() / 128)
    lookup = RowLookup((0, 7))
    ids = torch.tensor([[6, 0, 6, 2] * 6])
    values, scales = lookup.raw_rows(q.values, q.scale, ids)
    assert torch.equal(values.view(torch.uint8), q.values.view(torch.uint8)[ids])
    assert torch.equal(scales.view(torch.uint8), q.scale.view(torch.uint8)[ids])
    assert values.flatten(-2).shape == (1, 6144)


@pytest.mark.parametrize("trainable", [False, True])
def test_provider_matches_local_and_coalesces_gradients(trainable):
    q = quantize_swa(torch.linspace(-1, 1, 7 * 256).reshape(7, 256))
    reference = EngramTable(q.values, q.scale, trainable=trainable)
    actual = ShardedEngramTable(q.values, q.scale, RowLookup((0, 7)), trainable=trainable)
    ids = torch.tensor([[6, 0, 6, 2]])
    torch.testing.assert_close(actual(ids), reference(ids), atol=0, rtol=0)
    if trainable:
        weights = torch.arange(4).reshape(1, 4, 1).float()
        (actual(ids) * weights).sum().backward()
        assert actual.master.grad.dtype == torch.float32
        expected = torch.zeros_like(actual.master)
        expected[6] = 2
        expected[0] = 1
        expected[2] = 3
        torch.testing.assert_close(actual.master.grad, expected, atol=0, rtol=0)
    else:
        assert not list(actual.parameters())


def test_empty_requests_and_invalid_ids():
    lookup = RowLookup((0, 2))
    values = torch.zeros(2, 32, dtype=torch.uint8)
    scales = torch.ones(2, 1, dtype=torch.uint8)
    assert lookup.raw_rows(values, scales, torch.empty(0, 24, dtype=torch.int64))[0].shape == (0, 24, 32)
    with pytest.raises(ValueError):
        lookup.raw_rows(values, scales, torch.tensor([2]))
    with pytest.raises(ValueError):
        RowLookup((0, 2, 1))

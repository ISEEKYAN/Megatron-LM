import torch

from megatron.lite.primitive.quantization.ds41_index import fake_quant_index, quantize_index
from megatron.lite.primitive.quantization.ds41_kv import fake_quant_main_kv, quantize_main_kv


def test_index_group_scale_and_zero():
    x = torch.full((2,64), 6.25)
    index, main = quantize_index(x), quantize_main_kv(x)
    assert index.scale.shape == (2,2)
    assert main.scale.shape == (2,4)
    assert index.scale.dtype == torch.float8_e8m0fnu
    assert index.scale.view(torch.uint8).tolist() == [[128,128],[128,128]]
    assert main.scale.view(torch.uint8).tolist() == [[56]*4]*2
    assert quantize_index(torch.zeros(32)).scale.view(torch.uint8).tolist() == [1]
    assert torch.equal(index.decoded, torch.full_like(x,6.))


def test_switches_independent_and_input_gradient():
    for main_enabled in (False, True):
        for index_enabled in (False, True):
            x = torch.full((32,), .3)
            x[-1] = 6.
            x.requires_grad_()
            main = fake_quant_main_kv(x, enabled=main_enabled)
            index = fake_quant_index(x, enabled=index_enabled)
            assert torch.equal(main, quantize_main_kv(x).decoded if main_enabled else x)
            assert torch.equal(index, quantize_index(x).decoded if index_enabled else x)
            grad = torch.arange(32).float()
            (index * grad).sum().backward()
            assert torch.equal(x.grad, grad)


def test_index_midpoints_and_sign_bits():
    row = [.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.]
    x = torch.tensor([row + [-v for v in row]] * 2).reshape(1,32)
    before = x.clone()
    actual = quantize_index(x)
    expected_bytes = [0x20, 0x42, 0x64, 0x76, 0xa8, 0xca, 0xec, 0xfe] * 2
    assert actual.packed.view(torch.uint8).tolist() == [expected_bytes]
    assert actual.scale.view(torch.uint8).tolist() == [[127]]
    assert torch.equal(x, before)
    assert not actual.scale.requires_grad

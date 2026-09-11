import pytest
import torch

from megatron.lite.primitive.quantization.ds41_fp8 import (
    dynamic_fp8_linear,
    fake_quant_swa,
    quantize_linear_activation,
    quantize_swa,
)


@pytest.mark.parametrize("codec", [quantize_swa, quantize_linear_activation])
def test_fp8_rounding_zero_and_scale(codec):
    x = torch.zeros(2,32)
    x[0,:3] = torch.tensor([.265625, .296875, 448.])
    actual = codec(x)
    assert actual.scale.view(torch.uint8).tolist() == [[127], [105]]
    assert actual.decoded[0,:3].tolist() == [.25, .3125, 448.]
    assert actual.values.dtype == torch.float8_e4m3fn
    assert not actual.scale.requires_grad


def test_swa_ste_full_vector():
    x = torch.ones(2,64, requires_grad=True)
    grad = torch.arange(128).reshape_as(x).float()
    result = fake_quant_swa(x)
    assert torch.equal(result, quantize_swa(x).decoded)
    (result * grad).sum().backward()
    assert torch.equal(x.grad, grad)


def test_linear_cpu_cannot_masquerade_as_fp8_gemm():
    with pytest.raises(RuntimeError, match="CUDA"):
        dynamic_fp8_linear(torch.ones(2,32), torch.ones(32,32))

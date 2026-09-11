import pytest
import torch
from megatron.lite.primitive.quantization.ds41_kv import (
    fake_quant_main_kv,
    quantize_main_kv,
)


def test_main_midpoints_codes_scales_and_sign():
    row = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
    x = torch.tensor([row + [-v for v in row]])
    result = quantize_main_kv(x)
    codes = [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]
    expected = torch.tensor(
        [[codes[i] | (codes[i + 1] << 4) for i in range(0, 16, 2)]], dtype=torch.uint8
    )
    assert torch.equal(result.packed.view(torch.uint8), expected)
    assert result.scale.dtype == torch.float8_e4m3fn
    assert result.scale.view(torch.uint8).tolist() == [[56]]
    assert result.decoded.tolist() == [
        [
            0.0,
            1.0,
            1.0,
            2.0,
            2.0,
            4.0,
            4.0,
            6.0,
            -0.0,
            -1.0,
            -1.0,
            -2.0,
            -2.0,
            -4.0,
            -4.0,
            -6.0,
        ]
    ]


def test_zero_saturation_and_gradient():
    zero = quantize_main_kv(torch.zeros(2, 32))
    assert zero.scale.shape == (2, 2)
    assert zero.scale.view(torch.uint8).tolist() == [[1, 1], [1, 1]]
    assert torch.count_nonzero(zero.decoded) == 0
    x = torch.full((2, 32), 6.25, requires_grad=True)
    result = quantize_main_kv(x)
    assert torch.equal(result.decoded, torch.full_like(x, 6.0))
    assert not result.scale.requires_grad
    grad = torch.arange(64).reshape(2, 32).float()
    output = fake_quant_main_kv(x)
    assert torch.equal(output, result.decoded)
    (output * grad).sum().backward()
    assert torch.equal(x.grad, grad)


def test_main_phase_shape_and_nonfinite_rejection():
    with pytest.raises(ValueError):
        fake_quant_main_kv(torch.ones(16), phase="pretraining")
    with pytest.raises(ValueError):
        quantize_main_kv(torch.ones(17))
    with pytest.raises(ValueError):
        quantize_main_kv(torch.full((16,), float("nan")))
    with pytest.raises(TypeError):
        quantize_main_kv(torch.ones(16, dtype=torch.int32))

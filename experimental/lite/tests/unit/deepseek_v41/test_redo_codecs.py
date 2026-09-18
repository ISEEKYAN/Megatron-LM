# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.lite.model.deepseek_v41.codecs import CODECS
from megatron.lite.primitive.ckpt.hf_weights import _dequantize_block_scaled_tensor
from megatron.lite.primitive.quantization import mxfp4, nvfp4, qat
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4


@pytest.mark.parametrize('shape', [(3, 64), (2, 3, 64), (2, 2, 3, 64)])
@pytest.mark.parametrize('group', [0, -1, 16, 32])
def test_grouping_preserves_rows_and_backward(shape, group):
    values = torch.linspace(-3, 4, torch.tensor(shape).prod().item()).reshape(shape)
    flat = values.reshape(-1, shape[-1])
    expected = qat.compute_amax(flat, group)
    actual = qat.compute_amax(values, group)
    assert torch.equal(actual.reshape(-1), expected.reshape(-1))
    assert torch.equal(qat._grouped_view(values, group).reshape(-1), values.reshape(-1))


@pytest.mark.parametrize('codec,block,dtype', [
    (nvfp4.quantize_main_kv, 16, torch.float8_e4m3fn),
    (mxfp4.quantize_index, 32, torch.float8_e8m0fnu),
])
def test_fp4_codec_rounding_and_surface(codec, block, dtype):
    values = torch.tensor([0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.]).repeat(8)[:64]
    result = codec(values.reshape(1, 64))
    assert result.packed.shape == (1, 32)
    assert result.scale.shape == (1, 64//block) and result.scale.dtype == dtype
    assert torch.equal(result.decoded[0, :9], torch.tensor([0.,0.,1.,1.,2.,2.,4.,4.,6.]))
    assert torch.equal(codec(result.decoded).packed, result.packed)
    if block == 32:
        assert torch.equal(dequantize_mxfp4(result.packed, result.scale), result.decoded)


def test_codec_registry_rejects_mixed_scale_contracts():
    for key in [('main_kv', 32, 'e8m0', 'e2m1'), ('index', 16, 'e4m3', 'e2m1')]:
        with pytest.raises(KeyError):
            CODECS[key]
    x = torch.randn(2, 64, requires_grad=True)
    CODECS['main_kv', 16, 'e4m3', 'e2m1'](x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_tail_block_uses_declared_32_instead_of_65_div_3():
    x = torch.ones(65, 65)
    scales = torch.arange(1, 10).reshape(3, 3).float()
    actual = _dequantize_block_scaled_tensor(x, scales, x.shape, block_shape=(32,32))
    assert actual[31,31] == 1 and actual[32,32] == 5 and actual[64,64] == 9
    assert torch.equal(actual, scales.repeat_interleave(32,0).repeat_interleave(32,1)[:65,:65])
    with pytest.raises(ValueError, match='scale shape'):
        _dequantize_block_scaled_tensor(x, scales, x.shape, block_shape=(16,16))

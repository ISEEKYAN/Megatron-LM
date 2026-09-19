# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.model.deepseek_v41.codecs import CODECS
from megatron.lite.primitive.ckpt.hf_weights import _dequantize_block_scaled_tensor
from megatron.lite.primitive.quantization import mxfp4, nvfp4
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4


@pytest.mark.parametrize(
    'codec,block,dtype',
    [
        (nvfp4.quantize_main_kv, 16, torch.float8_e4m3fn),
        (mxfp4.quantize_index, 32, torch.float8_e8m0fnu),
    ],
)
def test_fp4_codec_rounding_and_surface(codec, block, dtype):
    values = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]).repeat(8)[
        :64
    ]
    result = codec(values.reshape(1, 64))
    assert result.packed.shape == (1, 32)
    assert result.scale.shape == (1, 64 // block) and result.scale.dtype == dtype
    assert torch.equal(
        result.decoded[0, :9],
        torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]),
    )
    assert torch.equal(codec(result.decoded).packed, result.packed)
    if block == 32:
        assert torch.equal(
            dequantize_mxfp4(result.packed, result.scale), result.decoded
        )


def test_main_kv_codec_preserves_straight_through_gradient():
    x = torch.randn(2, 64, requires_grad=True)
    CODECS['main_kv', 16, 'e4m3', 'e2m1'](x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.gpus(1)
@pytest.mark.parametrize('owns_k', [False, True])
def test_cross_layer_indexer_fp8_projection(v41_core_te, monkeypatch, owns_k):
    from megatron.lite.primitive.modules.attention.csa import (
        CrossLayerAttentionConfig,
        CrossLayerIndexer,
    )
    assert torch.cuda.is_available(), "FP8 numerical evidence requires CUDA"
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    config = CrossLayerAttentionConfig(dim=32, q_rank=32, index_heads=1, index_dim=32)
    indexer = CrossLayerIndexer(config, owns_k).cuda().float()
    # Every row/block has max in (224, 448], hence its E8M0 scale is exactly 1.
    # All decoded products are integers; sums stay below 2**24, so FP32
    # accumulation is exact regardless of GEMM reduction order.
    x = torch.linspace(225, 300, 32, device='cuda').repeat(16, 1)
    weight = torch.linspace(225, 350, 1024, device='cuda').reshape(32, 32)
    with torch.no_grad():
        indexer.wq_b.weight.copy_(weight)
    expected = (
        x.to(torch.float8_e4m3fn).float() @ weight.to(torch.float8_e4m3fn).float().T
    )
    baseline = torch.nn.functional.linear(x, weight)
    actual = indexer.wq_b(x)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, baseline), "FP8 was bypassed"
    relative_error = (actual - baseline).norm() / baseline.norm()
    # E4M3 normal rounding <= 1/16 per operand: product error <= 33/256.
    bound = (33 / 256) * (x.abs() @ weight.abs().T).norm() / baseline.norm()
    assert 0 < relative_error <= bound


def test_tail_block_uses_declared_32_instead_of_65_div_3():
    x = torch.ones(65, 65)
    scales = torch.arange(1, 10).reshape(3, 3).float()
    actual = _dequantize_block_scaled_tensor(x, scales, x.shape, block_shape=(32, 32))
    assert actual[31, 31] == 1 and actual[32, 32] == 5 and actual[64, 64] == 9
    assert torch.equal(
        actual, scales.repeat_interleave(32, 0).repeat_interleave(32, 1)[:65, :65]
    )
    with pytest.raises(ValueError, match='scale shape'):
        _dequantize_block_scaled_tensor(x, scales, x.shape, block_shape=(16, 16))

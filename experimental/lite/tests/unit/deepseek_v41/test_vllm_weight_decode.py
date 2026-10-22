# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""All nibble/finite-scale operands, including signed zeros, share one codec."""
import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.experts import _decode_weight
from megatron.lite.primitive.quantization.mxfp4 import dequantize_mxfp4


def operands(device):
    # Every byte contains both E2M1 nibble values; every finite E8M0 scale.
    codes = torch.arange(256, dtype=torch.int16, device=device).to(torch.uint8)
    codes = codes.repeat(255, 1).contiguous().view(torch.int8)
    scale = torch.arange(255, dtype=torch.uint8, device=device)
    scale = scale[:, None].expand(255, 16).contiguous().view(torch.float8_e8m0fnu)
    return codes, scale


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_shared_decoder_composition(dtype):
    codes, scale = operands('cpu')
    expected = dequantize_mxfp4(codes, scale).to(dtype)
    actual = _decode_weight._torchdynamo_orig_callable(codes, scale, dtype)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert torch.signbit(actual[127, 16::32]).all()  # negative-zero low nibble


@pytest.mark.gpus
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_fused_cuda_decoder_preserves_all_operand_bytes(dtype):
    codes, scale = operands('cuda')
    expected = dequantize_mxfp4(codes, scale).to(dtype)
    actual = _decode_weight(codes, scale, dtype)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert torch.signbit(actual[127, 16::32]).all()

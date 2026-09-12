# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Indexer Q/K E2M1/group32/E8M0 QAT, independently switched from main KV."""

import torch

from .ds41_kv import QuantizedValues, _fake_quant, _quantize_nibbles, _validate_input
from .mxfp4 import dequantize_mxfp4


@torch.no_grad()
def quantize_index(post_rope):
    _validate_input(post_rope, 32)
    blocks = post_rope.float().reshape(*post_rope.shape[:-1], -1, 32)
    # Official activation kernel clamps amax before selecting the E8M0 scale.
    amax = blocks.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    scale = (exponent + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    codes = _quantize_nibbles(blocks / torch.exp2(exponent).unsqueeze(-1))
    codes = codes.reshape(post_rope.shape)
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous().view(torch.int8)
    decoded = dequantize_mxfp4(packed, scale).to(post_rope.dtype)
    if not torch.isfinite(decoded).all():
        raise ValueError("index codec produced nonfinite values")
    return QuantizedValues(packed, scale, decoded)


def fake_quant_index(post_rope, *, enabled=True):
    return _fake_quant(post_rope, enabled, quantize_index)

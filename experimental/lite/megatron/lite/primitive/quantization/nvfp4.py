# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVFP4 E2M1/group16/E4M3 activation encoding with detached scales."""
import torch

from .mxfp4 import QuantizedValues, _fake_quant, _quantize_nibbles, _validate_input


@torch.no_grad()
def quantize_main_kv(post_rope):
    _validate_input(post_rope, 16)
    blocks = post_rope.float().reshape(*post_rope.shape[:-1], -1, 16)
    amax = blocks.abs().amax(-1).clamp_min(6 * 2.0**-9)
    scale = (amax / 6).to(torch.float8_e4m3fn)
    if not torch.isfinite(scale.float()).all():
        raise ValueError("main-KV scale exceeds finite E4M3 range")
    codes = _quantize_nibbles(
        (blocks / scale.float().unsqueeze(-1)).clamp(-6, 6), ties="even"
    )
    codes = codes.reshape(post_rope.shape)
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous().view(torch.int8)
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=post_rope.device)
    decoded = table[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    decoded = decoded.reshape(blocks.shape) * scale.float().unsqueeze(-1)
    return QuantizedValues(
        packed, scale, decoded.reshape(post_rope.shape).to(post_rope.dtype)
    )


def fake_quant_main_kv(post_rope, *, enabled=True, phase="post-training"):
    return _fake_quant(post_rope, enabled, quantize_main_kv, phase)

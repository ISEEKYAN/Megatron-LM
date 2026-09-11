# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Post-RoPE main-KV E2M1/group16/E4M3 codec and approved detached-scale STE."""

from dataclasses import dataclass

import torch


def _quantize_nibbles(values):
    """Official PTX cvt.rn E2M1; ModelOpt's weight codec uses different ties."""
    magnitude = values.abs()
    index = torch.zeros_like(magnitude, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        index += (magnitude > boundary).to(torch.uint8)
    for boundary in (0.75, 1.75, 3.5):
        index += (magnitude == boundary).to(torch.uint8)
    return index | (values.signbit().to(torch.uint8) << 3)


@dataclass(frozen=True)
class QuantizedValues:
    packed: torch.Tensor
    scale: torch.Tensor
    decoded: torch.Tensor


class _IdentityGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, decoded):
        return decoded

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _validate_input(x, block):
    if x.ndim == 0 or x.shape[-1] == 0 or x.shape[-1] % block:
        raise ValueError(f"last dimension must be nonzero and divisible by {block}")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("codec input must be F32, BF16 or F16")
    if not torch.isfinite(x).all():
        raise ValueError("codec input must be finite")


@torch.no_grad()
def quantize_main_kv(post_rope):
    _validate_input(post_rope, 16)
    blocks = post_rope.float().reshape(*post_rope.shape[:-1], -1, 16)
    amax = blocks.abs().amax(-1).clamp_min(6 * 2.0**-9)
    scale = (amax / 6).to(torch.float8_e4m3fn)
    if not torch.isfinite(scale.float()).all():
        raise ValueError("main-KV scale exceeds finite E4M3 range")
    codes = _quantize_nibbles((blocks / scale.float().unsqueeze(-1)).clamp(-6, 6))
    codes = codes.reshape(post_rope.shape)
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous().view(torch.int8)
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=post_rope.device)
    decoded = table[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    decoded = decoded.reshape(blocks.shape) * scale.float().unsqueeze(-1)
    return QuantizedValues(
        packed, scale, decoded.reshape(post_rope.shape).to(post_rope.dtype)
    )


def fake_quant_main_kv(post_rope, *, enabled=True, phase="post-training"):
    if type(enabled) is not bool:
        raise TypeError("enabled must be bool")
    if phase != "post-training":
        raise ValueError("main-KV QAT is supported only for post-training")
    if not enabled:
        return post_rope
    return _IdentityGradient.apply(post_rope, quantize_main_kv(post_rope).decoded)

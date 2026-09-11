# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Indexer Q/K E2M1/group32/E8M0 QAT, independently switched from main KV."""

import torch

from .ds41_kv import QuantizedValues, _IdentityGradient, _validate_input
from .mxfp4 import dequantize_mxfp4, quantize_mxfp4


@torch.no_grad()
def quantize_index(post_rope):
    _validate_input(post_rope, 32)
    packed, scale = quantize_mxfp4(post_rope)
    decoded = dequantize_mxfp4(packed, scale).to(post_rope.dtype)
    if not torch.isfinite(decoded).all():
        raise ValueError("index codec produced nonfinite values")
    return QuantizedValues(packed, scale, decoded)


def fake_quant_index(post_rope, *, enabled=True):
    if type(enabled) is not bool:
        raise TypeError("enabled must be bool")
    if not enabled:
        return post_rope
    return _IdentityGradient.apply(post_rope, quantize_index(post_rope).decoded)

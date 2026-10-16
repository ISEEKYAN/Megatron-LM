# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-visible engram arithmetic and owned gradients."""

import torch

from .autograd import visible_forward
from .linear import decoded_bf16_master


def engram_reference(hidden, kv, query, key, *, eps, token_mask=None):
    """Owned FP32/decoded-BF16 STE reference; retain every live master leaf."""
    copies, dim = hidden.shape[-2:]
    keys, values = kv.float().split((copies * dim, dim), -1)
    keys = keys.unflatten(-1, (copies, dim))
    h = hidden.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        keys.square().mean(-1) + eps
    )
    dot = (
        (h * keys * decoded_bf16_master(query) * decoded_bf16_master(key)).sum(-1)
        * rstd
        * dim**-0.5
    )
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (h + gate.unsqueeze(-1) * values.unsqueeze(-2)).to(hidden.dtype)


def engram_post(hidden, kv, query, key, *, eps, token_mask=None):
    """Native visible BF16 gate/residual, with the declared owned reference VJP.

    Query/key masters remain FP32 leaves. Visible arithmetic consumes their
    BF16 deployment values. No gradient is taken through the mask or hashing.
    Higher-order derivatives follow the same unsupported contract as _Visible.
    """

    def reference(h, v, q, k):
        return engram_reference(h, v, q, k, eps=eps, token_mask=token_mask)

    def visible(h, v, q, k):
        from megatron.lite.primitive.kernels.deployment_engram import post_wkv

        shape = h.shape
        result = post_wkv(
            h.reshape(-1, shape[-2], shape[-1]),
            v.reshape(-1, v.shape[-1]),
            q.bfloat16(),
            k.bfloat16(),
            eps=eps,
            token_mask=None if token_mask is None else token_mask.reshape(-1),
        )
        return result.reshape_as(h)

    return visible_forward(visible, reference, hidden, kv, query, key)

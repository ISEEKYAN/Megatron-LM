# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Thin adapters for Megatron Core's native mHC operations.

Requires Core's native_sinkhorn API (nv/dev 0cd11658f); no local fallback.
The model owns its RMS formula, shifted pre-mix and residual orientation.
"""

import torch


def sinkhorn(logits, iterations, eps):
    from megatron.core.transformer.hyper_connection import native_sinkhorn

    return native_sinkhorn(logits, iterations, eps)


def aggregate(hidden, pre):
    from megatron.core.transformer.hyper_connection import native_h_aggregate

    # CED requires the release's bitwise VJP; compilation reassociates its sum.
    return torch.compiler.disable(native_h_aggregate)(hidden.float(), pre.float()).to(hidden.dtype)


def post_mix(output, residual, post, comb):
    from megatron.core.transformer.hyper_connection import native_h_post_bda

    return native_h_post_bda(
        comb.float(), residual.float(), post.float(), output.float(), None
    ).to(output.dtype)

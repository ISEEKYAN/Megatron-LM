# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unpadded single-rank THD execution with sequence-local state lifetimes."""

import torch
from megatron.lite.primitive.utils.packed_seq import packed_sequence_ranges


def packed_forward(
    sequence_forward, hidden, pre_mix, cu_seqlens, *, input_ids=None, image_mask=None
):
    """Run a pure sequence callable over each logical sample, preserving its graph.

    The callable returns (hidden, next_pre_mix), creates a fresh AttentionState
    per invocation, and keeps bias/statistic publication outside forward. RNG is
    consumed in sequence order, exactly as for independent calls. This is a
    correctness path, not fused packed attention or distributed CP transport.
    """
    if hidden.ndim != 4 or hidden.shape[0] != 1 or pre_mix.shape != hidden.shape[:-1]:
        raise ValueError("Expected packed hidden [1,T,HC,D] and pre_mix [1,T,HC]")
    for tensor in (input_ids, image_mask):
        if tensor is not None and tensor.shape != hidden.shape[:2]:
            raise ValueError("Token inputs must match packed [1,T] dimensions")
    outputs, mixes = [], []
    for begin, end in packed_sequence_ranges(cu_seqlens, hidden.shape[1]):
        kwargs = {}
        if input_ids is not None:
            kwargs['input_ids'] = input_ids[:, begin:end]
        if image_mask is not None:
            kwargs['image_mask'] = image_mask[:, begin:end]
        h, p = sequence_forward(hidden[:, begin:end], pre_mix[:, begin:end], **kwargs)
        outputs.append(h)
        mixes.append(p)
    return torch.cat(outputs, dim=1), torch.cat(mixes, dim=1)

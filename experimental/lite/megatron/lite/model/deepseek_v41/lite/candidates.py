# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sequence-local two-level CSA2 candidate selection for post-training."""

import torch
from torch.nn import functional as F


def _visible(scores, lengths):
    lengths = torch.as_tensor(lengths, device=scores.device)
    return torch.arange(scores.shape[-1], device=scores.device) < lengths


def candidate_blocks(
    scores, lengths, *, topk_blocks=2048, block_size=8, phase="post-training"
):
    if phase != "post-training":
        raise ValueError("Two-level candidates are a post-training policy")
    if block_size <= 0 or topk_blocks <= 0:
        raise ValueError("Candidate block size and count must be positive")
    width = scores.shape[-1]
    if width == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    masked = scores.masked_fill(~_visible(scores, lengths), -torch.inf)
    maxima = F.pad(masked, (0, -width % block_size), value=-torch.inf)
    maxima = maxima.unflatten(-1, (-1, block_size)).amax(-1)
    last = (torch.as_tensor(lengths, device=scores.device) - 1) // block_size
    blocks = torch.arange(maxima.shape[-1], device=scores.device)
    maxima = maxima.masked_fill(blocks == last, torch.inf)
    top = maxima.topk(min(topk_blocks, maxima.shape[-1]), dim=-1)
    keep = torch.zeros_like(maxima, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    # Preserve complete block membership; consumers independently apply causal
    # visibility, including the unfinished portion of the newest block.
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_positions(scores, lengths, topk, *, offset=0, candidates=None):
    if topk < 0:
        raise ValueError("Top-K must be nonnegative")
    visible = _visible(scores, lengths)
    masked = scores.masked_fill(~visible, -torch.inf)
    if candidates is not None:
        if candidates.shape != scores.shape or candidates.dtype != torch.bool:
            raise ValueError("Candidate mask must be boolean and match scores")
        masked = masked.masked_fill(~candidates, -torch.inf)
    indices = masked.topk(min(topk, scores.shape[-1]), dim=-1, sorted=False).indices
    indices = indices.sort(-1).values
    # An empty/small pool must never reintroduce an excluded position through
    # the arbitrary tail of topk(-inf). Invalid slots are represented by -1.
    valid = masked.gather(-1, indices) > -torch.inf
    return torch.where(valid, indices + offset, -1).to(torch.int32)

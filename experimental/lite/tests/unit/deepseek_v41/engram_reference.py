# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unchanged Torch Engram reference from MLite 5cf22a8, engram_lookup.py.

Unit operands supply KV directly through Identity owners; this tests only the
post-WKV gate/residual and its VJP, not row lookup or the projection GEMM.
"""
import torch
from torch import nn


class Engram(nn.Module):
    def __init__(self, hidden_size, copies, embedding, projection, *, eps=1e-6):
        super().__init__()
        self.dim = hidden_size
        self.copies = copies
        self.eps = eps
        self.embed = embedding
        self.wkv = projection
        self.q_weight = nn.Parameter(torch.ones(copies, hidden_size))
        self.k_weight = nn.Parameter(torch.ones(copies, hidden_size))

    def forward(self, hidden, hash_ids, token_mask=None):
        if hidden.ndim != 4 or hidden.shape[-2:] != (self.copies, self.dim):
            raise ValueError("Expected residual stream [B,S,HC,D]")
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.copies * self.dim, self.dim], -1)
        key = key.float().unflatten(-1, (self.copies, self.dim))
        h = hidden.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (
            (h * key * self.q_weight.float() * self.k_weight.float()).sum(-1)
            * rstd
            * self.dim**-0.5
        )
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)

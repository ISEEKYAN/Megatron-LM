# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shifted hyper-connections using Core kernels, layout [B,S,HC,D]."""
import torch
from megatron.lite.primitive.kernels.mhc import aggregate, post_mix, sinkhorn
from torch import nn
from torch.nn import functional as F


def contract_hc(hidden: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 4 or pre_mix.shape != hidden.shape[:-1]:
        raise ValueError("Expected paired hidden [B,S,HC,D] and pre_mix [B,S,HC]")
    return aggregate(hidden, pre_mix)


def expand_hc(tokens: torch.Tensor, copies: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 3 or copies < 1:
        raise ValueError("Expected tokens [B,S,D] and positive HC count")
    hidden = tokens.unsqueeze(2).expand(*tokens.shape[:2], copies, tokens.shape[-1])
    pre = tokens.new_zeros(*tokens.shape[:2], copies, dtype=torch.float32)
    pre[..., 0] = 1
    return hidden, pre


def mix_residual(output, residual, post, comb):
    # comb axes are SOURCE, DESTINATION; matmul(comb, residual) transposes the
    # official information flow, even when Sinkhorn makes both sums near one.
    return post_mix(output, residual, post, comb)


class HCMixes(nn.Module):
    def __init__(self, hidden_size, copies, norm_eps=1e-6, hc_eps=1e-6, iterations=20):
        super().__init__()
        if copies < 1 or iterations < 1:
            raise ValueError("HC copies and Sinkhorn iterations must be positive")
        self.copies = copies
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.iterations = iterations
        size = copies * (copies + 2)
        self.fn = nn.Parameter(
            torch.empty(size, copies * hidden_size, dtype=torch.float32)
        )
        self.base = nn.Parameter(torch.zeros(size, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        nn.init.xavier_uniform_(self.fn)

    def forward(self, hidden):
        flat = hidden.flatten(2).float()
        mixes = F.linear(flat, self.fn.float()) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.norm_eps
        )
        sizes = [self.copies, self.copies, self.copies**2]
        pre, post, comb = mixes.split(sizes, dim=-1)
        bp, bpost, bc = self.base.float().split(sizes)
        pre = torch.sigmoid(pre * self.scale[0] + bp) + self.hc_eps
        post = 2 * torch.sigmoid(post * self.scale[1] + bpost)
        comb = (comb * self.scale[2] + bc).reshape(
            *flat.shape[:-1], self.copies, self.copies
        )
        comb = sinkhorn(comb, self.iterations, self.hc_eps)
        return pre, post, comb


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)

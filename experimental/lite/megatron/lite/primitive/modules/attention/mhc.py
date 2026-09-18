# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadHyperConnectionHead(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = eps
        self.hc_fn = nn.Parameter(
            torch.empty(hc_mult, hc_mult * hidden_size, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.hc_fn)
        nn.init.zeros_(self.hc_base)
        nn.init.ones_(self.hc_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, self.hc_fn.float()) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
            + self.eps
        )
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype)


from inspect import unwrap

from megatron.core.fusions.fused_mhc_kernels import fused_h_aggregate, fused_h_post_bda
from megatron.core.transformer.hyper_connection import (
    _sinkhorn_iterations,
    native_h_aggregate,
    native_h_post_bda,
)

RMSNorm = nn.RMSNorm
_sinkhorn_iterations = unwrap(_sinkhorn_iterations)


def contract_hc(hidden, pre_mix):
    op = fused_h_aggregate if hidden.is_cuda else unwrap(native_h_aggregate)
    return op(hidden.float(), pre_mix.float()).to(hidden.dtype)


def mix_residual(output, residual, post, comb):
    op = fused_h_post_bda if output.is_cuda else unwrap(native_h_post_bda)
    return op(comb, residual, post, output, None).to(output.dtype)


def expand_hc(tokens, copies):
    from megatron.lite.primitive.parallel.mhc import expand_mhc_hidden_for_pipeline

    hidden = expand_mhc_hidden_for_pipeline(tokens, hc_mult=copies)
    pre = tokens.new_zeros(*tokens.shape[:2], copies, dtype=torch.float32)
    pre[..., 0] = 1
    return hidden, pre


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
        comb = _sinkhorn_iterations(comb, self.iterations, self.hc_eps)
        return pre, post, comb

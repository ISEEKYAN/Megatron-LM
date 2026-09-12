# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen mathematical components; sparse attention uses explicit gathered rows."""
import torch
from torch import nn
from torch.nn import functional as F


class Qwen3_8_FlashNextHyperConnection(nn.Module):
    def __init__(self, hidden_size, count=4, lowrank=320, eps=1e-6, write=True):
        super().__init__()
        if hidden_size <= 0 or count < 2 or lowrank <= 0:
            raise ValueError('HC_DIMENSIONS')
        self.hidden_size, self.count, self.eps = hidden_size, count, eps
        width = hidden_size * count
        self.hc_norm = nn.Module()
        self.hc_norm.register_parameter("weight", nn.Parameter(torch.zeros(width)))
        self.input_mix_weight_down = nn.Linear(width, lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(lowrank, width, bias=False)
        self.block_inject_weight = (
            nn.Linear(width, count, bias=False) if write else None
        )

    def mix(self, x):
        if x.shape[-1] != self.hidden_size * self.count:
            raise ValueError('HC_STREAM_WIDTH')
        streams = x.unflatten(-1, (self.count, self.hidden_size))
        compute = streams.double() if x.dtype == torch.float64 else streams.float()
        normalized = (
            compute * torch.rsqrt(compute.square().mean(-1, keepdim=True) + self.eps)
        ).flatten(-2)
        normalized = normalized * (1 + self.hc_norm.weight.to(normalized.dtype))
        gate = torch.sigmoid(
            self.input_mix_weight_up(
                F.silu(
                    self.input_mix_weight_down(
                        normalized.to(self.input_mix_weight_down.weight.dtype)
                    )
                    / self.count
                )
            )
        )
        return (normalized * gate).unflatten(-1, (self.count, self.hidden_size)).mean(
            -2
        ).to(x.dtype), (x, normalized)

    def combine(self, y, residual):
        if self.block_inject_weight is None:
            raise RuntimeError('HC_READ_ONLY')
        x, normalized = residual
        gate = 2 * torch.sigmoid(
            self.block_inject_weight(
                normalized.to(self.block_inject_weight.weight.dtype)
            )
            / self.count
        )
        return x + (gate.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2)


def qsa_routes(q, k, lengths, *, token_budget=2048, compress_ratio=4, offset=0):
    if (
        q.ndim != 4
        or k.ndim != 4
        or k.shape[2] != 1
        or q.shape[0] != k.shape[0]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError('QSA_QK_SHAPE')
    if token_budget <= 0 or compress_ratio <= 1 or token_budget % compress_ratio:
        raise ValueError('QSA_BUDGET')
    if (
        offset < 0
        or lengths.shape != (q.shape[0],)
        or bool((lengths < 0).any())
        or bool((lengths > offset + q.shape[1]).any())
    ):
        raise ValueError('QSA_LENGTHS')
    if bool((lengths // compress_ratio > k.shape[1]).any()):
        raise ValueError('QSA_MISSING_BLOCKS')
    result = torch.full(
        (*q.shape[:2], token_budget + compress_ratio - 1),
        -1,
        dtype=torch.int64,
        device=q.device,
    )
    budget = token_budget // compress_ratio
    for b in range(q.shape[0]):
        for t in range(min(q.shape[1], max(0, int(lengths[b]) - offset))):
            end = offset + t + 1
            visible = end // compress_ratio
            blocks = torch.arange(visible, device=q.device)
            if visible > budget:
                scores = (
                    torch.einsum(
                        'hd,pd->hp', q[b, t].float(), k[b, :visible, 0].float()
                    )
                    .relu()
                    .sum(0)
                    / q.shape[-1] ** 0.5
                )
                blocks = scores.topk(budget).indices
            chosen = (
                blocks[:, None] * compress_ratio
                + torch.arange(compress_ratio, device=q.device)
            ).flatten()
            chosen = torch.cat(
                [chosen, torch.arange(visible * compress_ratio, end, device=q.device)]
            )
            result[b, t, : chosen.numel()] = chosen
    return result


def sparse_attention(q, k, v, routes):
    if (
        q.ndim != 4
        or k.shape != v.shape
        or k.ndim != 4
        or q.shape[0] != k.shape[0]
        or q.shape[-1] != k.shape[-1]
        or q.shape[2] % k.shape[2]
    ):
        raise ValueError('QSA_ATTENTION_SHAPE')
    if (
        routes.shape[:2] != q.shape[:2]
        or routes.dtype != torch.int64
        or bool(((routes < -1) | (routes >= k.shape[1])).any())
    ):
        raise ValueError('QSA_ROUTE_RANGE')
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    selected_k = k[batch, routes.clamp_min(0)].repeat_interleave(
        q.shape[2] // k.shape[2], dim=3
    )
    selected_v = v[batch, routes.clamp_min(0)].repeat_interleave(
        q.shape[2] // k.shape[2], dim=3
    )
    scores = (
        torch.einsum('bshd,bskhd->bshk', q.float(), selected_k.float())
        / q.shape[-1] ** 0.5
    )
    valid = routes >= 0
    scores = scores.masked_fill(~valid[:, :, None, :], -torch.inf)
    scores = torch.where(valid.any(-1)[:, :, None, None], scores, 0)
    probability = scores.softmax(-1) * valid[:, :, None, :]
    return torch.einsum('bshk,bskhd->bshd', probability, selected_v.float()).to(q.dtype)

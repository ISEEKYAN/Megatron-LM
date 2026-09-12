# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Differentiable V4.1 vision tower and spatial aligner.

State names follow the pinned release; no inference-mode boundary is installed.
Trainability and distributed synchronization belong to the training protocol.
"""

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Keep one FP32 cast so backward accumulates before returning to BF16."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * value).to(dtype)


def get_vision_cos_sin(n_h, n_w, dim, theta, device=None):
    frequency = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # Height frequencies precede width frequencies in each half of the head.
    height, width = torch.meshgrid(
        torch.arange(n_h, device=device), torch.arange(n_w, device=device), indexing='ij'
    )
    phase = (
        (torch.stack((height, width), -1).reshape(-1, 2, 1).float() * frequency)
        .flatten(1)
        .unsqueeze(1)
    )
    return phase.cos(), phase.sin()


def apply_rotary(x, cos, sin):
    first, second = x.float().chunk(2, -1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(x.dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, patches):
        return self.proj(patches.flatten(1))


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // self.n_heads
        if args.vision_dim % self.n_heads or self.head_dim % 4:
            raise ValueError('Vision heads require a dimension divisible by four')
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim)

    def forward(self, x, cos, sin):
        q, k, v = [
            part.reshape(len(x), self.n_heads, self.head_dim) for part in self.wqkv(x).chunk(3, -1)
        ]
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        result = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(result.transpose(0, 1).reshape(len(x), -1))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, -1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(args.vision_dim), RMSNorm(args.vision_dim)
        self.attn, self.mlp = Attention(args), MLP(args)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)

    def forward(self, patches, n_h, n_w):
        if min(n_h, n_w) < 1 or len(patches) != n_h * n_w:
            raise ValueError('Patch count must equal the positive image grid area')
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        if self.downsample_ratio < 1:
            raise ValueError('Downsample ratio must be positive')
        self.w1 = nn.Linear(args.vision_dim * self.downsample_ratio**2, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x, n_h, n_w):
        if min(n_h, n_w) < 1 or x.ndim != 2 or len(x) != n_h * n_w:
            raise ValueError('Features must match the positive image grid')
        ratio = self.downsample_ratio
        spatial = x.reshape(n_h, n_w, -1).permute(2, 0, 1)
        spatial = F.pad(spatial, (0, -n_w % ratio, 0, -n_h % ratio))
        cells = F.unfold(spatial.unsqueeze(0), ratio, stride=ratio)[0].transpose(0, 1)
        return self.w2(F.gelu(self.w1(cells)))

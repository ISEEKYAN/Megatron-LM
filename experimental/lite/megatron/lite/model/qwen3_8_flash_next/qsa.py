# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen sparse attention for complete padded or packed rows (no CP transport)."""
import torch
from megatron.lite.primitive.utils.rope import _apply_rotary_pos_emb_bshd
from torch import nn

from .math import qsa_routes, sparse_attention


class Qwen3_8_FlashNextQSAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.q_proj = nn.Linear(
            c.hidden_size, 2 * c.num_attention_heads * c.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            c.num_attention_heads * c.head_dim, c.hidden_size, bias=False
        )
        self.q_norm = self._scale(c.head_dim)
        self.k_norm = self._scale(c.head_dim)
        self.indexer = nn.Module()
        self.indexer.index_qk_proj = nn.Linear(
            c.hidden_size, (c.indexer_n_heads + 1) * c.indexer_head_dim, bias=False
        )
        self.indexer.q_layernorm = self._scale(c.indexer_head_dim)
        self.indexer.k_layernorm = self._scale(c.indexer_head_dim)
        self.indexer.requires_grad_(False)

    @staticmethod
    def _scale(width):
        module = nn.Module()
        module.register_parameter("weight", nn.Parameter(torch.zeros(width)))
        return module

    def _norm(self, x, weight):
        return (
            x.float()
            * torch.rsqrt(
                x.float().square().mean(-1, keepdim=True) + self.config.rms_norm_eps
            )
            * (1 + weight.weight.float())
        ).to(x.dtype)

    def forward(self, x, angles, *, lengths=None, cu_seqlens=None):
        """Angles are expanded rotary angles [B,S,1,R]; documents reset RoPE externally."""
        if (
            x.ndim != 3
            or angles.shape[:2] != x.shape[:2]
            or angles.ndim != 4
            or angles.shape[2] != 1
            or angles.shape[-1] % 2
            or not 0
            < angles.shape[-1]
            <= min(self.config.head_dim, self.config.indexer_head_dim)
        ):
            raise ValueError('QSA_INPUT_ANGLES')
        if cu_seqlens is not None:
            if (
                lengths is not None
                or x.shape[0] != 1
                or cu_seqlens.ndim != 1
                or cu_seqlens.numel() < 2
                or int(cu_seqlens[0]) != 0
                or int(cu_seqlens[-1]) != x.shape[1]
                or not bool((cu_seqlens.diff() > 0).all())
            ):
                raise ValueError('QSA_PACKED_BOUNDARIES')
            pieces = [
                self.forward(x[:, a:b], angles[:, a:b])
                for a, b in zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:])
            ]
            return torch.cat(pieces, 1)
        c = self.config
        b, s, _ = x.shape
        if lengths is None:
            lengths = torch.full((b,), s, device=x.device, dtype=torch.long)
        with torch.no_grad():
            projected = self.indexer.index_qk_proj(x).reshape(
                b, s, c.indexer_n_heads + 1, c.indexer_head_dim
            )
            iq = _apply_rotary_pos_emb_bshd(
                self._norm(projected[:, :, :-1], self.indexer.q_layernorm), angles
            )
            blocks = s // c.indexer_compress_ratio
            raw = projected[:, : blocks * c.indexer_compress_ratio, -1:]
            pooled = (
                raw.reshape(b, blocks, c.indexer_compress_ratio, 1, c.indexer_head_dim)
                .float()
                .mean(2)
                .to(x.dtype)
            )
            ik = _apply_rotary_pos_emb_bshd(
                self._norm(pooled, self.indexer.k_layernorm),
                angles[
                    :, : blocks * c.indexer_compress_ratio : c.indexer_compress_ratio
                ],
            )
            routes = qsa_routes(
                iq,
                ik,
                lengths,
                token_budget=c.indexer_budget,
                compress_ratio=c.indexer_compress_ratio,
            )
        q, gate = (
            self.q_proj(x)
            .reshape(b, s, c.num_attention_heads, 2 * c.head_dim)
            .chunk(2, -1)
        )
        k = self.k_proj(x).reshape(b, s, c.num_key_value_heads, c.head_dim)
        v = self.v_proj(x).reshape_as(k)
        q = _apply_rotary_pos_emb_bshd(self._norm(q, self.q_norm), angles)
        k = _apply_rotary_pos_emb_bshd(self._norm(k, self.k_norm), angles)
        output = (
            sparse_attention(q, k, v, routes[..., : min(s, routes.shape[-1])])
            * gate.sigmoid()
        )
        return self.o_proj(output.flatten(-2))

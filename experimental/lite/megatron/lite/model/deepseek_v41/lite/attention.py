# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Differentiable CSA2 full-sequence semantics with explicit per-call ownership.

This is the single-rank correctness implementation. It materializes attention
scores; it does not claim the fused sparse kernel's performance or CP support.
"""

import math
from dataclasses import dataclass, replace

import torch
from megatron.lite.primitive.quantization import ds41_fp8
from megatron.lite.primitive.quantization.ds41_index import fake_quant_index
from megatron.lite.primitive.quantization.ds41_kv import fake_quant_main_kv
from torch import nn
from torch.nn import functional as F

from .block import RMSNorm
from .candidates import candidate_blocks, select_positions


@dataclass(frozen=True)
class CSA2Config:
    dim: int = 5120
    heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_rank: int = 1280
    o_rank: int = 1024
    groups: int = 8
    index_heads: int = 32
    index_dim: int = 128
    topk: int = 512
    window: int = 128
    candidate_blocks: int = 2048
    block_size: int = 8
    eps: float = 1e-20
    rope_theta: float = 10000
    compress_rope_theta: float = 160000
    original_length: int = 65536
    factor: float = 16
    beta_fast: float = 32
    beta_slow: float = 1
    linear_fp8: bool = True
    main_qat: bool = True
    index_qat: bool = True
    swa_fp8: bool = True


@dataclass(frozen=True)
class AttentionState:
    kv_owner: int | None = None
    index_owner: int | None = None
    latent: torch.Tensor | None = None
    main_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def rotate(x, positions, config, ratio, *, inverse=False):
    rd = config.rope_dim
    theta = config.compress_rope_theta if ratio != 0 else config.rope_theta
    dims = torch.arange(0, rd, 2, device=x.device, dtype=torch.float32)
    freqs = 1 / theta ** (dims / rd)
    if ratio != 0 and config.original_length > 0:

        def correction(rotations):
            return (
                rd
                * math.log(config.original_length / (rotations * 2 * math.pi))
                / (2 * math.log(theta))
            )

        low = max(math.floor(correction(config.beta_fast)), 0)
        high = min(math.ceil(correction(config.beta_slow)), rd - 1)
        ramp = (
            (torch.arange(rd // 2, device=x.device) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        freqs = freqs / config.factor * ramp + freqs * (1 - ramp)
    angles = positions.float().unsqueeze(-1) * freqs
    phase = torch.polar(torch.ones_like(angles), -angles if inverse else angles)
    phase = phase.reshape(1, positions.numel(), *([1] * (x.ndim - 3)), rd // 2)
    tail = torch.view_as_complex(
        x[..., -rd:].float().contiguous().unflatten(-1, (-1, 2))
    )
    rotated = torch.view_as_real(tail * phase).flatten(-2).to(x.dtype)
    return torch.cat([x[..., :-rd], rotated], -1)


class Linear(nn.Linear):
    def __init__(self, input_size, output_size, *, fp8=False, dtype=torch.bfloat16):
        super().__init__(input_size, output_size, bias=False, dtype=dtype)
        self.fp8 = fp8

    def forward(self, x):
        if getattr(self, 'native_fp32', False) and not self.fp8:
            from megatron.lite.primitive.modules.native_fp32_linear import (
                native_fp32_linear,
            )

            return native_fp32_linear(x, self.weight)
        return (
            ds41_fp8.dynamic_fp8_linear(x, self.weight)
            if self.fp8
            else F.linear(x, self.weight)
        )


class Compressor(nn.Module):
    def __init__(self, config, ratio):
        super().__init__()
        self.ratio = ratio
        dtype = torch.float32 if ratio > 1 else torch.bfloat16
        self.wkv = Linear(config.dim, config.head_dim, dtype=dtype)
        self.norm = RMSNorm(config.head_dim, config.eps)
        if ratio > 1:
            self.wgate = Linear(config.dim, config.head_dim, dtype=torch.float32)

    def forward(self, x):
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        cutoff = x.shape[1] // self.ratio * self.ratio
        values = self.wkv(x[:, :cutoff].float()).unflatten(1, (-1, self.ratio))
        gates = self.wgate(x[:, :cutoff].float()).unflatten(1, (-1, self.ratio))
        return self.norm((values * gates.softmax(2)).sum(2).to(x.dtype))


class Indexer(nn.Module):
    def __init__(self, config, owns_k):
        super().__init__()
        self.wq_b = Linear(
            config.q_rank, config.index_heads * config.index_dim, fp8=config.linear_fp8
        )
        self.weights_proj = Linear(config.dim, config.index_heads)
        if owns_k:
            self.wk = Linear(config.head_dim, config.index_dim)
            self.k_norm = RMSNorm(config.index_dim, config.eps)
        # Post-training port policy: selection is frozen, with no indexer loss.
        # Shared compressor/Q projections outside this module remain trainable.
        self.requires_grad_(False)


class CSA2Attention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if not 0 <= layer_id < 40:
            raise ValueError(
                "Only the 40 backbone layers are supported; DSpark is excluded"
            )
        if (
            config.heads % config.groups
            or config.rope_dim % 2
            or not 0 < config.rope_dim <= min(config.head_dim, config.index_dim)
        ):
            raise ValueError("Invalid grouped-head or rotary dimensions")
        self.config, self.layer_id = config, layer_id
        self.ratio = 0 if layer_id < 2 else 2 if layer_id < 20 else 1
        self.owns_kv = layer_id in (2, 8, 14, 20)
        self.owns_index = layer_id in (2, 8, 14, 20, 24, 28, 32, 36)
        self.wq_a = Linear(config.dim, config.q_rank, fp8=config.linear_fp8)
        self.q_norm = RMSNorm(config.q_rank, config.eps)
        self.wq_b = Linear(
            config.q_rank, config.heads * config.head_dim, fp8=config.linear_fp8
        )
        self.wkv = Linear(config.dim, config.head_dim, fp8=config.linear_fp8)
        self.kv_norm = RMSNorm(config.head_dim, config.eps)
        self.attn_sink = nn.Parameter(torch.zeros(config.heads, dtype=torch.float32))
        self.wo_a = Linear(
            config.heads * config.head_dim // config.groups,
            config.groups * config.o_rank,
        )
        self.wo_b = Linear(
            config.groups * config.o_rank, config.dim, fp8=config.linear_fp8
        )
        self.compressor = Compressor(config, self.ratio) if self.owns_kv else None
        self.indexer = Indexer(config, self.owns_kv) if self.owns_index else None

    def forward(self, x, state, *, candidate_mask=None):
        c, layer, ratio = self.config, self.layer_id, self.ratio
        b, length, _ = x.shape
        positions = torch.arange(length, device=x.device)
        qr = self.q_norm(self.wq_a(x))
        # V4.1 deliberately has no per-query-head RMS after wq_b.
        q = rotate(
            self.wq_b(qr).unflatten(-1, (c.heads, c.head_dim)), positions, c, ratio
        )
        window = rotate(self.kv_norm(self.wkv(x)), positions, c, ratio)
        if c.swa_fp8:
            window = ds41_fp8.fake_quant_swa(window)
        kv = window
        visible = (positions[None, :] <= positions[:, None]) & (
            positions[None, :] > positions[:, None] - c.window
        )
        mask = visible.expand(b, -1, -1)
        if ratio:
            if self.owns_kv:
                latent = self.compressor(x)
                cp = torch.arange(latent.shape[1], device=x.device) * ratio
                # Index keys branch BEFORE main RoPE/QAT. No in-place aliasing.
                index_k = self.indexer.k_norm(self.indexer.wk(latent))
                index_k = fake_quant_index(
                    rotate(index_k, cp, c, ratio), enabled=c.index_qat
                )
                main = fake_quant_main_kv(
                    rotate(latent, cp, c, ratio), enabled=c.main_qat
                )
                state = AttentionState(
                    kv_owner=layer, latent=latent, main_kv=main, index_k=index_k
                )
            expected_owner = (
                20 if layer >= 20 else 14 if layer >= 14 else 8 if layer >= 8 else 2
            )
            if (
                state.kv_owner != expected_owner
                or state.main_kv is None
                or state.index_k is None
            ):
                raise ValueError("Missing or incorrect CSA2 KV source state")
            if state.main_kv.shape[:2] != (b, length // ratio):
                raise ValueError("Shared KV belongs to a different sequence shape")
            lengths = ((positions + 1) // ratio).unsqueeze(-1)
            if self.owns_index:
                iq = self.indexer.wq_b(qr).unflatten(-1, (c.index_heads, c.index_dim))
                iq = fake_quant_index(
                    rotate(iq, positions, c, ratio), enabled=c.index_qat
                )
                weights = self.indexer.weights_proj(x) * (
                    c.index_dim**-0.5 * c.index_heads**-0.5
                )
                scores = (
                    torch.einsum('bshd,btd->bsht', iq, state.index_k).relu()
                    * weights.unsqueeze(-1)
                ).sum(2)
                pool = state.candidates
                if layer == 20:
                    pool = candidate_blocks(
                        scores,
                        lengths,
                        topk_blocks=c.candidate_blocks,
                        block_size=c.block_size,
                    )
                injected_pool = pool if candidate_mask is None else candidate_mask
                if layer > 20 and injected_pool is None:
                    raise ValueError("Reindex requires the layer-20 candidate pool")
                selected = select_positions(
                    scores,
                    lengths,
                    c.topk,
                    candidates=injected_pool if layer > 20 else None,
                )
                state = replace(
                    state, index_owner=layer, indices=selected, candidates=pool
                )
            else:
                expected_index = (
                    (20 + ((layer - 20) // 4) * 4) if layer >= 20 else expected_owner
                )
                if state.index_owner != expected_index or state.indices is None:
                    raise ValueError("Missing or incorrect CSA2 index source state")
            kv = torch.cat([window, state.main_kv], 1)
            global_mask = torch.zeros(
                b, length, state.main_kv.shape[1], dtype=torch.bool, device=x.device
            )
            if state.indices.numel():
                # Clamp padding only for scatter addressing; false -1 entries
                # cannot overwrite a valid selection of position zero.
                counts = torch.zeros_like(global_mask, dtype=torch.int32)
                counts.scatter_add_(
                    -1, state.indices.clamp_min(0).long(), (state.indices >= 0).int()
                )
                global_mask = counts > 0
            mask = torch.cat([mask, global_mask], -1)
        logits = (
            torch.einsum('bshd,btd->bsht', q.float(), kv.float()) * c.head_dim**-0.5
        )
        logits = logits.masked_fill(~mask.unsqueeze(2), -torch.inf)
        sink = self.attn_sink.expand(b, length, -1).unsqueeze(-1)
        probabilities = torch.cat([logits, sink], -1).softmax(-1)[..., :-1]
        output = torch.einsum('bsht,btd->bshd', probabilities, kv.float()).to(x.dtype)
        output = rotate(output, positions, c, ratio, inverse=True)
        grouped = output.reshape(b, length, c.groups, -1)
        weight = self.wo_a.weight.reshape(c.groups, c.o_rank, -1)
        if getattr(self.wo_a, 'native_fp32', False):
            from megatron.lite.primitive.modules.native_fp32_linear import (
                native_fp32_linear,
            )

            output = torch.stack(
                [
                    native_fp32_linear(grouped[:, :, i], weight[i])
                    for i in range(c.groups)
                ],
                dim=2,
            ).flatten(2)
        else:
            output = torch.einsum('bsgd,grd->bsgr', grouped, weight).flatten(2)
        return self.wo_b(output), state

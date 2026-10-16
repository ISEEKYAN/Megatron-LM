# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Caller-owned DS41 caches using the rollout writers and gather kernels."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IndexCache:
    packed: torch.Tensor
    scale: torch.Tensor

    @property
    def shape(self):
        return (*self.packed.shape[:-1], self.packed.shape[-1] * 2)

    @property
    def ndim(self):
        return self.packed.ndim

    @property
    def device(self):
        return self.packed.device


def cache_geometry(batch, rows, block_size, *, device):
    """Give each request disjoint pages, including empty and partial pages."""
    if batch < 1 or rows < 0 or block_size < 1:
        raise ValueError("Invalid private cache geometry")
    pages = max(1, (rows + block_size - 1) // block_size)
    table = torch.arange(batch * pages, device=device, dtype=torch.int32)
    table = table.reshape(batch, pages)
    slots = torch.arange(rows, device=device, dtype=torch.int64)[None, :]
    slots = slots + table[:, :1].to(torch.int64) * block_size
    lengths = torch.full((batch,), rows, dtype=torch.int32, device=device)
    return table, slots.contiguous(), lengths


def rotary_cache(config, ratio, device):
    """Construct the exact deployment rotary cache from the release config."""
    if config.deployment_hf_config is None:
        raise ValueError("Deployment cache requires the release HF config")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    hf = DeepseekV41Config(**config.deployment_hf_config)
    with set_current_vllm_config(VllmConfig()), torch.device(device):
        rope = build_deepseek_v4_rope(
            hf,
            head_dim=config.head_dim,
            rope_head_dim=config.rope_dim,
            max_position_embeddings=hf.max_position_embeddings,
            compress_ratio=ratio,
        )
    return rope.cos_sin_cache.to(device=device)


def gather_main(cache, table, lengths, rows, block_size):
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        dequantize_and_gather_k_cache,
    )

    output = torch.empty(
        (table.shape[0], rows, 512), dtype=torch.bfloat16, device=cache.device
    )
    if rows:
        dequantize_and_gather_k_cache(
            output, cache, lengths, None, table, block_size, 0
        )
    return output


def pool_norm(raw, gamma, ratio, eps):
    """Lower full request rows to the rollout compressor's actual ring API."""
    if ratio not in (1, 2) or raw.ndim != 3 or raw.shape[-1] != 512 * ratio:
        raise ValueError('Deployment compressor requires dense CR1/2 rows')
    if not raw.is_cuda or raw.dtype != torch.float32:
        raise ValueError('Deployment compressor requires CUDA FP32 CR1/2 rows')
    if gamma.shape != (512,) or gamma.device != raw.device:
        raise ValueError('Deployment compressor requires colocated gamma[512]')
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        fused_save_compress_norm,
    )

    batch, rows, _ = raw.shape
    output = torch.empty((batch * rows, 512), device=raw.device, dtype=torch.bfloat16)
    if rows:
        positions = torch.arange(rows, device=raw.device, dtype=torch.int64).repeat(
            batch
        )
        requests = torch.arange(batch, device=raw.device, dtype=torch.int32)
        requests = requests.repeat_interleave(rows)
        starts = torch.arange(batch + 1, device=raw.device, dtype=torch.int32) * rows
        state = (
            torch.zeros((batch, 2, 1024), device=raw.device, dtype=torch.float32)
            if ratio == 2
            else None
        )
        slots = requests.to(torch.int64) * 2 + positions.remainder(2)
        fused_save_compress_norm(
            raw.flatten(0, 1).contiguous(),
            positions,
            state,
            slots,
            starts if ratio == 2 else None,
            requests if ratio == 2 else None,
            gamma.bfloat16(),
            eps,
            ratio,
            output,
        )
    cutoff = rows // ratio * ratio
    return output.reshape(batch, rows, 512)[:, ratio - 1 : cutoff : ratio].contiguous()


@torch.no_grad()
def index_query(query, raw_weights, positions, cos_sin):
    if not query.is_cuda or query.dtype != torch.bfloat16:
        raise ValueError('Deployment index query requires CUDA BF16')
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )

    batch, rows, heads, width = query.shape
    if not rows:
        return (
            IndexCache(
                torch.empty(
                    (*query.shape[:-1], width // 2),
                    device=query.device,
                    dtype=torch.uint8,
                ),
                torch.empty(
                    (*query.shape[:-1], width // 32),
                    device=query.device,
                    dtype=torch.uint8,
                ),
            ),
            raw_weights.bfloat16(),
        )
    (packed, scales), weights = fused_indexer_q_rope_quant(
        positions.repeat(batch),
        query.flatten(0, 1).contiguous(),
        cos_sin,
        raw_weights.flatten(0, 1).contiguous(),
        width**-0.5,
        heads**-0.5,
        use_fp4=True,
        weights_out_dtype=torch.bfloat16,
    )
    return (
        IndexCache(
            packed.reshape(batch, rows, heads, width // 2),
            scales.contiguous()
            .view(torch.uint8)
            .reshape(batch, rows, heads, width // 32),
        ),
        weights.reshape(batch, rows, heads),
    )


@torch.no_grad()
def index_cache(k_pre, gamma, eps, ratio, cos_sin, *, block_size=64):
    """Keep the packet read by rollout's gather and DeepGEMM, without QAT."""
    if not k_pre.is_cuda or k_pre.dtype != torch.bfloat16:
        raise ValueError("Deployment index cache requires CUDA BF16 projections")
    from vllm import _custom_ops as ops
    from vllm.models.deepseek_v41.common.ops.indexer_k_store import (
        indexer_k_norm_rope_store,
    )

    batch, rows, width = k_pre.shape
    if width != 128 or ratio not in (1, 2):
        raise ValueError("Deployment index cache requires D128 and CR1/2")
    table, slots, _ = cache_geometry(batch, rows, block_size, device=k_pre.device)
    cache = torch.zeros(
        (table.numel(), block_size, 68), dtype=torch.uint8, device=k_pre.device
    )
    positions = torch.arange(rows, device=k_pre.device, dtype=torch.int64)
    positions = (positions * ratio + ratio - 1).repeat(batch)
    if rows:
        indexer_k_norm_rope_store(
            k_pre.flatten(0, 1).contiguous(),
            positions,
            cos_sin,
            gamma.bfloat16(),
            eps,
            cache,
            slots.flatten(),
            ratio,
            True,
        )
    packed = torch.empty((batch * rows, 64), dtype=torch.uint8, device=k_pre.device)
    scales = torch.empty((batch * rows, 4), dtype=torch.uint8, device=k_pre.device)
    if rows:
        cumulative = torch.arange(batch + 1, device=k_pre.device, dtype=torch.int32)
        cumulative = cumulative * rows
        ops.cp_gather_indexer_k_quant_cache(cache, packed, scales, table, cumulative)
    return IndexCache(packed.reshape(batch, rows, 64), scales.reshape(batch, rows, 4))


def compressed_cache(latent, ratio, cos_sin, reference, *, block_size=64):
    """Use NVFP4 writer and BI native gather; only the VJP uses the reference."""
    from .autograd import visible_forward

    if not latent.is_cuda or latent.dtype != torch.bfloat16:
        raise ValueError("Deployment main cache requires CUDA BF16 latents")

    def visible(values):
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            rope_quant_insert,
        )

        batch, rows, _ = values.shape
        table, slots, lengths = cache_geometry(
            batch, rows, block_size, device=values.device
        )
        cache = torch.zeros(
            (table.numel(), block_size, 288), dtype=torch.uint8, device=values.device
        )
        positions = torch.arange(rows, device=values.device, dtype=torch.int64)
        positions = (positions * ratio + ratio - 1).repeat(batch)
        if rows:
            rope_quant_insert(
                values.flatten(0, 1).contiguous(),
                positions,
                cos_sin,
                cache,
                slots.flatten(),
                ratio,
            )
        return gather_main(cache, table, lengths, rows, block_size)

    return visible_forward(visible, reference, latent)


def _swa_insert(query, values, positions, cos_sin, eps, block_size):
    """One joint native Q-norm/RoPE + SWA KV RoPE/quant/insert call (row-paired)."""
    if (
        not query.is_cuda
        or query.dtype != torch.bfloat16
        or values.dtype != torch.bfloat16
    ):
        raise ValueError("Deployment SWA cache requires CUDA BF16 Q/KV")
    batch, rows, heads, width = query.shape
    table, slots, lengths = cache_geometry(
        batch, rows, block_size, device=values.device
    )
    cache = torch.zeros(
        (table.numel(), block_size, 528), dtype=torch.uint8, device=values.device
    )
    padded_heads = 64 if heads <= 64 else 128
    output = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        query.flatten(0, 1).contiguous(),
        values.flatten(0, 1).contiguous(),
        cache.view(cache.shape[0], -1),
        slots.flatten(),
        positions.repeat(batch),
        cos_sin,
        padded_heads,
        eps,
        block_size,
        False,
        True,
        True,
        False,
    )
    return (
        output[:, :heads].reshape(batch, rows, heads, width),
        gather_main(cache, table, lengths, rows, block_size),
    )


def swa_cache(q, kv, positions, cos_sin, eps, reference, *, block_size=64, start=None):
    """Call the joint native Q/SWA writer and the deployment MXFP8 gather.

    With contiguous CP, ``q`` holds this rank's rows (at ``positions``) and
    ``kv`` the whole gathered sequence (rows ``0..L``) starting at ``start``.
    The kernel pairs Q and KV rows under one position array, so it runs twice:
    once over the full KV with a zero placeholder query (the SWA cache), and
    once over the local rows (only its query output is used). Both outputs are
    row-wise, so the bytes equal the CP1 call's corresponding rows.
    """
    from .autograd import visible_forward

    def visible(query, values):
        if start is None:
            return _swa_insert(query, values, positions, cos_sin, eps, block_size)
        batch, rows = query.shape[:2]
        key_positions = torch.arange(values.shape[1], device=values.device)
        # Same head count as the real query: the joint writer is only exercised
        # with real (padded 64/128) head layouts.
        placeholder = query.new_zeros((batch, values.shape[1], *query.shape[2:]))
        _, window = _swa_insert(
            placeholder, values, key_positions, cos_sin, eps, block_size
        )
        if rows == 0:
            # A CP rank can own no rows of a short document; the native joint
            # Q+KV entry has no zero-token early exit (its grid would be empty).
            return query.new_empty(query.shape), window
        local_values = values[:, start : start + rows]
        rotated, _ = _swa_insert(
            query, local_values, positions, cos_sin, eps, block_size
        )
        return rotated, window

    return visible_forward(visible, reference, q, kv)

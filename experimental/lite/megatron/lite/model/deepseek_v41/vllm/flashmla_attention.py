# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sparse FlashMLA attention core on the deployment arithmetic (FLASHMLA_SPARSE_DSV41).

Forward replays vLLM's batch-invariant path of ``DeepseekV4FlashMLAAttention`` op for
op: Q RoPE fused with the MXFP8 SWA cache insert, the NVFP4 compressed cache insert,
then the attention the installed vLLM serves (``inference_path``): ``gather``
(d667e410e1: dequantize-and-gather, ``combine_topk_swa_indices``,
``flash_mla_sparse_fwd``) or ``paged`` (``flashmla-bi-direct``:
``sparse_paged_indices`` and the unsplit ``flash_mla_with_kvcache`` on the quantized
caches), then ``fused_inv_rope_fp8_quant`` (the head of ``dsv41_o_proj``). Only cache
paging and workspace layout differ from serving, which the kernels do not observe,
so the output bits are the serving bits.

Backward is cuDNN's DSA sparse-attention backward (what Core's
``CSASparseAttnFunc`` uses) on the forward's Q, output and LSE and the KV the
caches dequantize to (re-gathered from the saved caches), plus closed-form RoPE
transposes. Gradient contract:

* the output FP8 cast, the KV caches' MXFP8/NVFP4 casts and the BF16 roundings
  after RoPE are identity STEs (gradient w.r.t. the dequantized value passes to
  the pre-cast value; no clipping branch, the NVFP4 saturation from the e4m3
  scale rounding included);
* RoPE / inverse RoPE contribute their exact transposes with the FP32 cache;
* the sparse indices are fixed; ``attn_sink`` gets the sink-logit gradient.

Context parallel: each rank owns a contiguous row range of the packed sequence.
SWA KV and compressed latents are all-gathered (forward) and their gradients
reduce-scattered (backward); queries stay local. The attention kernel has no
split-KV, so the forward is bitwise equal to CP=1. ``attn_sink``'s gradient is
the rank-local partial; the data/context-parallel gradient reduction sums it.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist

SWA_RECORD_BYTES = 528
COMPRESSED_RECORD_BYTES = 288
SWA_BLOCK = 64
COMPRESSED_BLOCK = 64


@dataclass(frozen=True)
class FlashMLALayerConfig:
    compress_ratio: int
    window: int = 128
    top_k: int = 512
    num_heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    o_groups: int = 8
    o_quant_group: int = 32

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5


class AttentionCoreOutput(NamedTuple):
    """``o_proj`` input as serving feeds ``fp8_einsum`` plus its gradient carrier.

    ``data``/``scale`` are the FP8 activation and packed UE8M0 scales of
    ``fused_inv_rope_fp8_quant`` (``[T, G, 8 * 512]``). ``grad`` is a zero-strided
    BF16 tensor of the same shape: its value is unused, its gradient is the
    gradient w.r.t. the dequantized activation (the consumer's FP8 STE).
    """

    data: torch.Tensor
    scale: torch.Tensor
    grad: torch.Tensor


# ---------------------------------------------------------------------------
# RoPE transposes (GPT-J pairs on the last ``rope_dim`` dims, FP32 cos||sin)
# ---------------------------------------------------------------------------


def _cos_sin(cos_sin_cache, positions, rope_dim):
    half = rope_dim // 2
    rows = cos_sin_cache.index_select(0, positions.long())
    return rows[:, :half], rows[:, half:rope_dim]


def _rope_transpose(
    grad, positions, cos_sin_cache, rope_dim, inverse, out_dtype=None, in_place=False
):
    """VJP of the forward (``inverse=False``) or inverse rotation.

    The rotated dims are computed in FP32 and rounded once to ``out_dtype``
    (default: ``grad``'s dtype); the other dims are copied, or kept when
    ``in_place`` (``grad`` is then overwritten and returned).
    """
    out = grad if in_place else grad.to(out_dtype or grad.dtype, copy=True)
    lead = grad.shape[1:-1]
    cos, sin = _cos_sin(cos_sin_cache, positions, rope_dim)
    cos = cos.view(cos.shape[0], *([1] * len(lead)), -1)
    sin = sin.view(sin.shape[0], *([1] * len(lead)), -1)
    pairs = grad[..., -rope_dim:].float().unflatten(-1, (rope_dim // 2, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    if inverse:  # y_e = x_e c + x_o s, y_o = x_o c - x_e s
        d_even, d_odd = even * cos - odd * sin, even * sin + odd * cos
    else:  # y_e = x_e c - x_o s, y_o = x_o c + x_e s
        d_even, d_odd = even * cos + odd * sin, odd * cos - even * sin
    out[..., -rope_dim:] = torch.stack((d_even, d_odd), -1).flatten(-2)
    return out


# ---------------------------------------------------------------------------
# Layout planning (host side)
# ---------------------------------------------------------------------------


@dataclass
class _Plan:
    swa_slots: torch.Tensor  # [S_total] int64
    comp_slots: Optional[torch.Tensor]  # [N_total] int64
    comp_positions: Optional[torch.Tensor]  # [N_total] int64 boundary positions
    num_swa_blocks: int
    num_comp_blocks: int
    query_start_loc: torch.Tensor  # [R + 1] int32, local rows
    seq_lens: torch.Tensor  # [R] int32
    gather_lens: torch.Tensor  # [R] int32
    swa_table: torch.Tensor  # [R, B] int32
    comp_table: Optional[torch.Tensor]  # [R, B'] int32
    comp_lens: Optional[torch.Tensor]  # [R] int32
    n: int  # compressed rows per request in the workspace
    m: int  # workspace rows per request
    token_to_req: torch.Tensor  # [S_local] int32
    is_valid: torch.Tensor  # [S_local] bool
    replay_start: torch.Tensor  # [R] int32 zeros
    swa_dest: torch.Tensor  # [R * M] int64 global token row or -1
    comp_dest: Optional[torch.Tensor]  # [R * M] int64 global compressed row or -1
    swa_src: torch.Tensor  # [S_total] int64 workspace row of each token, R * M if none
    comp_src: Optional[torch.Tensor]  # [N_total] int64 workspace row of each entry


def _plan(cu_seqlens, row_begin, row_end, cfg, device):
    cr, window = cfg.compress_ratio, cfg.window
    cu = [int(x) for x in cu_seqlens]
    lengths = [b - a for a, b in zip(cu, cu[1:])]
    swa_slots, comp_slots, comp_positions = [], [], []
    swa_block_start, comp_block_start, comp_offset = [], [], []
    swa_blocks = comp_blocks = comp_rows = 0
    for length in lengths:
        swa_block_start.append(swa_blocks)
        swa_slots.append(swa_blocks * SWA_BLOCK + torch.arange(length))
        swa_blocks += -(-length // SWA_BLOCK)
        comp_block_start.append(comp_blocks)
        comp_offset.append(comp_rows)
        if cr:
            entries = length // cr
            comp_slots.append(comp_blocks * COMPRESSED_BLOCK + torch.arange(entries))
            comp_positions.append(torch.arange(entries) * cr + cr - 1)
            comp_blocks += -(-entries // COMPRESSED_BLOCK)
            comp_rows += entries

    requests = []
    for doc, (a, b) in enumerate(zip(cu, cu[1:])):
        qa, qb = max(a, row_begin), min(b, row_end)
        if qa < qb:
            seq_len = qb - a
            requests.append((doc, qa - row_begin, qb - row_begin, seq_len))
    gather = [min(seq, qb - qa + window - 1) for _, qa, qb, seq in requests]
    n = max((seq // cr for *_, seq in requests), default=0) if cr else 0
    m = n + max(gather, default=0)
    swa_width = max(-(-seq // SWA_BLOCK) for *_, seq in requests)
    swa_table = torch.zeros(len(requests), swa_width, dtype=torch.int32)
    comp_table = comp_lens = None
    if cr:
        comp_width = max(max(-(-(seq // cr) // COMPRESSED_BLOCK) for *_, seq in requests), 1)
        comp_table = torch.zeros(len(requests), comp_width, dtype=torch.int32)
        comp_lens = torch.tensor([seq // cr for *_, seq in requests], dtype=torch.int32)
    swa_dest = torch.full((len(requests), m), -1, dtype=torch.int64)
    swa_src = torch.full((cu[-1],), len(requests) * m, dtype=torch.int64)
    comp_src = torch.full((comp_rows,), len(requests) * m, dtype=torch.int64) if cr else None
    comp_dest = torch.full((len(requests), m), -1, dtype=torch.int64) if cr else None
    for r, ((doc, _, _, seq), g) in enumerate(zip(requests, gather)):
        blocks = -(-seq // SWA_BLOCK)
        swa_table[r, :blocks] = torch.arange(blocks) + swa_block_start[doc]
        swa_dest[r, n : n + g] = cu[doc] + seq - g + torch.arange(g)
        swa_src[cu[doc] + seq - g : cu[doc] + seq] = r * m + n + torch.arange(g)
        if cr:
            entries = seq // cr
            cblocks = -(-entries // COMPRESSED_BLOCK)
            comp_table[r, :cblocks] = torch.arange(cblocks) + comp_block_start[doc]
            comp_dest[r, :entries] = comp_offset[doc] + torch.arange(entries)
            comp_src[comp_offset[doc] : comp_offset[doc] + entries] = r * m + torch.arange(entries)

    def dev(t):
        return None if t is None else t.to(device, non_blocking=True)

    starts = [0] + [qb for _, _, qb, _ in requests]
    token_to_req = torch.cat(
        [torch.full((qb - qa,), r, dtype=torch.int32) for r, (_, qa, qb, _) in enumerate(requests)]
    )
    return _Plan(
        swa_slots=dev(torch.cat(swa_slots)),
        comp_slots=dev(torch.cat(comp_slots)) if cr else None,
        comp_positions=dev(torch.cat(comp_positions)) if cr else None,
        num_swa_blocks=swa_blocks,
        num_comp_blocks=max(comp_blocks, 1),
        query_start_loc=dev(torch.tensor(starts, dtype=torch.int32)),
        seq_lens=dev(torch.tensor([seq for *_, seq in requests], dtype=torch.int32)),
        gather_lens=dev(torch.tensor(gather, dtype=torch.int32)),
        swa_table=dev(swa_table),
        comp_table=dev(comp_table),
        comp_lens=dev(comp_lens),
        n=n,
        m=m,
        token_to_req=dev(token_to_req),
        is_valid=dev(torch.ones(token_to_req.shape[0], dtype=torch.bool)),
        replay_start=dev(torch.zeros(len(requests), dtype=torch.int32)),
        swa_dest=dev(swa_dest.flatten()),
        comp_dest=dev(comp_dest.flatten()) if cr else None,
        swa_src=dev(swa_src),
        comp_src=dev(comp_src),
    )


# ---------------------------------------------------------------------------
# Forward: the serving ops
# ---------------------------------------------------------------------------


def _require_batch_invariant():
    from vllm import envs

    if not envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError("the FlashMLA training core replays the VLLM_BATCH_INVARIANT=1 path")


def inference_path() -> str:
    """Which BI attention path the installed vLLM serves FLASHMLA_SPARSE_DSV41 with.

    ``paged``: every causal row on FlashMLA's unsplit sparse decode kernel over
    the quantized caches (``sparse_paged_indices``, vllm ``dsv41-bi/flashmla-bi-direct``).
    ``gather``: dequantize-and-gather to BF16, then the sparse prefill kernel
    (vLLM d667e410e1).
    """
    from vllm.model_executor.kernels import batch_invariant

    return "paged" if hasattr(batch_invariant, "sparse_paged_indices") else "gather"


@dataclass
class _Caches:
    swa: torch.Tensor  # [blocks, SWA_BLOCK, 528] uint8
    compressed: Optional[torch.Tensor]  # [blocks, COMPRESSED_BLOCK, 288] uint8


def _insert(q, kv_all, kv_local, latent_all, positions, positions_all, cfg, cos_sin, plan, cp):
    """Serving's cache writes; returns (q_rope, caches)."""
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert

    device, cr = q.device, cfg.compress_ratio
    swa = torch.zeros(
        plan.num_swa_blocks, SWA_BLOCK, SWA_RECORD_BYTES, dtype=torch.uint8, device=device
    )
    if cp:
        # Serving's PCP order: KV of every rank first, then the rank-local Q.
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(
            kv_all, swa, plan.swa_slots, positions_all, cos_sin, SWA_BLOCK, None, True
        )
        slots = plan.swa_slots[:0]
    else:
        slots = plan.swa_slots
    q_rope = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv_local,
        swa.view(plan.num_swa_blocks, -1),
        slots,
        positions,
        cos_sin,
        cfg.num_heads,
        0.0,
        SWA_BLOCK,
        False,  # apply_q_norm: Q is normed before wq_b
        True,  # kv_mxfp8: the V4.1 528 B record
        True,  # apply_q_rope
        False,  # is_q_interleaved
    )
    compressed = None
    if cr:
        compressed = torch.zeros(
            plan.num_comp_blocks,
            COMPRESSED_BLOCK,
            COMPRESSED_RECORD_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        rope_quant_insert(latent_all, plan.comp_positions, cos_sin, compressed, plan.comp_slots, cr)
    return q_rope, _Caches(swa, compressed)


def _gather(caches, topk, rows, cfg, plan):
    """BF16 KV workspace and combined indices of the gather path (d667e410e1).

    Also the backward's view of either path: the same KV values per row.
    """
    from vllm.models.deepseek_v41.common.ops import (
        combine_topk_swa_indices,
        dequantize_and_gather_k_cache,
    )

    cr = cfg.compress_ratio
    device = caches.swa.device
    workspace = torch.empty(
        plan.seq_lens.shape[0], plan.m, cfg.head_dim, dtype=torch.bfloat16, device=device
    )
    if cr:
        dequantize_and_gather_k_cache(
            workspace,
            caches.compressed,
            seq_lens=plan.comp_lens,
            gather_lens=None,
            block_table=plan.comp_table,
            block_size=COMPRESSED_BLOCK,
            offset=0,
        )
    dequantize_and_gather_k_cache(
        workspace,
        caches.swa,
        seq_lens=plan.seq_lens,
        gather_lens=plan.gather_lens,
        block_table=plan.swa_table,
        block_size=SWA_BLOCK,
        offset=plan.n,
    )
    if topk is None:
        topk = torch.full((rows, 1), -1, dtype=torch.int32, device=device)
    indices, lens = combine_topk_swa_indices(
        topk,
        plan.query_start_loc,
        plan.seq_lens,
        plan.gather_lens,
        cfg.window,
        cr,
        cfg.top_k if cr else 0,
        plan.m,
        plan.n,
    )
    return workspace, indices, lens


def _attend_gather(q_rope, caches, topk, sink, cfg, plan):
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    workspace, indices, lens = _gather(caches, topk, q_rope.shape[0], cfg, plan)
    out = torch.empty_like(q_rope)
    out, _, lse = flash_mla_sparse_fwd(
        q=q_rope,
        kv=workspace.view(-1, 1, cfg.head_dim),
        indices=indices.unsqueeze(1),
        sm_scale=cfg.softmax_scale,
        attn_sink=sink,
        topk_length=lens,
        out=out,
    )
    return out, lse


# One CTA per row along grid.y in the unsplit decode kernel.
_PAGED_MAX_ROWS = 65535


def _attend_paged(q_rope, caches, topk, sink, cfg, plan):
    """``DeepseekV4FlashMLAAttention._forward_bi`` of ``flashmla-bi-direct``."""
    from vllm.model_executor.kernels.batch_invariant import sparse_paged_indices
    from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta, flash_mla_with_kvcache

    rows, cr, window = q_rope.shape[0], cfg.compress_ratio, cfg.window
    device = q_rope.device
    top_k = cfg.top_k if cr else 0
    indices = torch.empty(rows, window + top_k, dtype=torch.int32, device=device)
    swa_lens = torch.empty(rows, dtype=torch.int32, device=device)
    extra_lens = torch.empty(rows, dtype=torch.int32, device=device)
    swa_indices = indices[:, :window]
    extra_indices = indices[:, window:] if cr else None
    sparse_paged_indices(
        topk if cr else None,
        plan.token_to_req,
        plan.is_valid,
        plan.query_start_loc,
        plan.seq_lens,
        plan.replay_start,
        plan.comp_table,
        COMPRESSED_BLOCK,
        plan.swa_table,
        SWA_BLOCK,
        0,
        rows,
        window,
        cr,
        top_k,
        swa_indices,
        swa_lens,
        extra_indices,
        extra_lens if cr else None,
    )
    out = torch.empty_like(q_rope)
    lse = torch.empty(rows, cfg.num_heads, dtype=torch.float32, device=device)
    swa_cache = caches.swa.unsqueeze(-2)
    extra_cache = caches.compressed.unsqueeze(-2) if cr else None
    for start in range(0, rows, _PAGED_MAX_ROWS):
        end = min(start + _PAGED_MAX_ROWS, rows)
        _, block_lse = flash_mla_with_kvcache(
            q=q_rope[start:end].unsqueeze(1),
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=cfg.head_dim,
            tile_scheduler_metadata=FlashMLASchedMeta(),
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices[start:end].unsqueeze(1),
            topk_length=swa_lens[start:end],
            softmax_scale=cfg.softmax_scale,
            attn_sink=sink,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=None if not cr else extra_indices[start:end].unsqueeze(1),
            extra_topk_length=None if not cr else extra_lens[start:end],
            out=out[start:end].unsqueeze(1),
            enable_batch_invariant=True,
        )
        lse[start:end] = block_lse.reshape(end - start, cfg.num_heads)
    return out, lse


def _output_quant(out, positions, cos_sin, cfg):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    return fused_inv_rope_fp8_quant(
        out,
        positions,
        cos_sin,
        n_groups=cfg.o_groups,
        heads_per_group=cfg.num_heads // cfg.o_groups,
        nope_dim=cfg.nope_dim,
        rope_dim=cfg.rope_dim,
        quant_group_size=cfg.o_quant_group,
        tma_aligned_scales=True,
        quantize=True,
    )


# ---------------------------------------------------------------------------
# Context-parallel collectives (variable row counts)
# ---------------------------------------------------------------------------


def _gather_rows(local, group, counts):
    """All-gather dim-0 chunks of possibly different sizes, in rank order."""
    width = max(counts)
    padded = local.new_zeros((width, *local.shape[1:]))
    padded[: local.shape[0]] = local
    parts = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(parts, padded.contiguous(), group=group)
    return torch.cat([p[:c] for p, c in zip(parts, counts)])


def _reduce_scatter_rows(full, group, counts, rank):
    """Sum ``full`` over ranks and keep this rank's chunk (FP32 accumulation)."""
    width = max(counts)
    full = full.float()
    chunks, start = [], 0
    for c in counts:
        chunk = full.new_zeros((width, *full.shape[1:]))
        chunk[:c] = full[start : start + c]
        chunks.append(chunk)
        start += c
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, chunks, group=group)
    return out[: counts[rank]]


def _counts(n, group):
    t = torch.tensor([n], dtype=torch.int64, device=torch.cuda.current_device())
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, t, group=group)
    return [int(p) for p in parts]


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------


class _FlashMLAAttnCore(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        swa_kv,
        comp_latent,
        attn_sink,
        topk,
        positions,
        cu_seqlens,
        cfg,
        cos_sin,
        cp_group,
        deterministic,
    ):
        _require_batch_invariant()
        cp = cp_group is not None and dist.get_world_size(cp_group) > 1
        if cp:
            rank = dist.get_rank(cp_group)
            row_counts = _counts(q.shape[0], cp_group)
            kv_all = _gather_rows(swa_kv, cp_group, row_counts)
            positions_all = _gather_rows(positions, cp_group, row_counts)
            comp_counts = (
                _counts(comp_latent.shape[0], cp_group) if comp_latent is not None else None
            )
            latent_all = (
                _gather_rows(comp_latent, cp_group, comp_counts)
                if comp_latent is not None
                else None
            )
            row_begin = sum(row_counts[:rank])
        else:
            rank, row_counts, comp_counts = 0, None, None
            kv_all, positions_all, latent_all, row_begin = swa_kv, positions, comp_latent, 0
        total = kv_all.shape[0]
        if cu_seqlens is None:
            cu_seqlens = [0, total]
        plan = _plan(cu_seqlens, row_begin, row_begin + q.shape[0], cfg, q.device)
        if cfg.compress_ratio:
            expected = int(plan.comp_slots.shape[0])
            assert latent_all is not None and latent_all.shape[0] == expected, (
                f"comp_latent rows {None if latent_all is None else latent_all.shape[0]} "
                f"!= {expected} compressed entries"
            )
            latent_all = latent_all.contiguous()
        q_rope, caches = _insert(
            q.contiguous(),
            kv_all.contiguous(),
            swa_kv.contiguous(),
            latent_all,
            positions,
            positions_all,
            cfg,
            cos_sin,
            plan,
            cp,
        )
        attend = _attend_paged if inference_path() == "paged" else _attend_gather
        out, lse = attend(q_rope, caches, topk, attn_sink.contiguous(), cfg, plan)
        data, scale = _output_quant(out, positions, cos_sin, cfg)
        ctx.plan, ctx.caches = plan, caches
        ctx.save_for_backward(q_rope, topk, out, lse, attn_sink, positions, positions_all, cos_sin)
        ctx.meta = (
            cfg,
            cp_group if cp else None,
            rank,
            row_counts,
            comp_counts,
            total,
            None if latent_all is None else latent_all.shape[0],
            deterministic,
        )
        ctx.mark_non_differentiable(data, scale)
        grad = q.new_zeros(()).expand(data.shape)
        return data, scale, grad

    @staticmethod
    def backward(ctx, _d_data, _d_scale, d_o):
        q_rope, topk, out, lse, sink, positions, positions_all, cos_sin = ctx.saved_tensors
        plan, caches = ctx.plan, ctx.caches
        cfg, group, rank, row_counts, comp_counts, total, comp_total, deterministic = ctx.meta
        rows, heads, dim = out.shape
        workspace, indices, lens = _gather(caches, topk, rows, cfg, plan)
        d_o = d_o.reshape(rows, heads, dim)
        d_out = _rope_transpose(
            d_o, positions, cos_sin, cfg.rope_dim, inverse=True, out_dtype=out.dtype
        )
        d_q_rope, d_kv, d_sink = _sparse_backward(
            q_rope,
            workspace.view(-1, dim),
            out,
            d_out,
            lse,
            sink,
            indices,
            lens,
            cfg.softmax_scale,
            deterministic,
        )
        del workspace
        del d_out
        d_q = _rope_transpose(
            d_q_rope, positions, cos_sin, cfg.rope_dim, inverse=False, in_place=True
        )
        # Each token / compressed entry takes its workspace row's gradient;
        # rows this rank did not gather read the appended zero row.
        d_kv = torch.cat([d_kv.float(), d_kv.new_zeros(1, dim, dtype=torch.float32)])
        d_swa = _rope_transpose(
            d_kv[plan.swa_src], positions_all, cos_sin, cfg.rope_dim, inverse=False, in_place=True
        )
        d_latent = None
        if cfg.compress_ratio:
            rope_positions = plan.comp_positions // cfg.compress_ratio * cfg.compress_ratio
            d_latent = _rope_transpose(
                d_kv[plan.comp_src],
                rope_positions,
                cos_sin,
                cfg.rope_dim,
                inverse=False,
                in_place=True,
            )
        del d_kv
        if group is not None:
            d_swa = _reduce_scatter_rows(d_swa, group, row_counts, rank)
            if d_latent is not None:
                d_latent = _reduce_scatter_rows(d_latent, group, comp_counts, rank)
        return (
            d_q.to(q_rope.dtype),
            d_swa.to(q_rope.dtype),
            None if d_latent is None else d_latent.to(q_rope.dtype),
            d_sink.to(sink.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _sparse_backward(q, kv, out, d_out, lse, sink, indices, lens, scale, deterministic):
    """cuDNN DSA sparse-attention backward, as Core's ``CSASparseAttnFunc``."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        fused_sparse_attention as core,
    )

    core._ensure_dsa_namespace()
    lse = lse.contiguous()
    result = core._DSA.sparse_attention_backward_wrapper(
        q,
        kv,
        out,
        d_out.contiguous(),
        lse,
        sink,
        indices,
        softmax_scale=scale,
        topk_length=lens,
        **core._deterministic_sparse_bwd_kwargs(
            q, kv, out, lse, sink, indices, lens, scale, requested=deterministic
        ),
    )
    return result["dq"], result["dkv"], result["d_sink"]


def flashmla_attention_core(
    q: torch.Tensor,
    swa_kv: torch.Tensor,
    comp_latent: Optional[torch.Tensor],
    topk_local: Optional[torch.Tensor],
    positions: torch.Tensor,
    attn_sink: torch.Tensor,
    layer_cfg: FlashMLALayerConfig,
    rope_cache: torch.Tensor,
    *,
    cu_seqlens=None,
    cp_group=None,
    deterministic: bool = False,
) -> AttentionCoreOutput:
    """Attention core of one FLASHMLA_SPARSE_DSV41 layer (see module docstring).

    Args:
        q: ``wq_b`` output, BF16 ``[S_local, 64, 512]``, no RoPE.
        swa_kv: ``kv_norm`` output, BF16 ``[S_local, 512]``, no RoPE.
        comp_latent: compressor latents of the rank's group-boundary tokens in
            order, BF16 ``[N_local, 512]``, no RoPE; ``None`` when ``cr == 0``.
        topk_local: indexer local compressed indices ``[S_local, top_k]`` int32,
            ascending with ``-1`` last; ``None`` when ``cr == 0``.
        positions: int64 ``[S_local]`` positions within each document.
        attn_sink: FP32 ``[64]``.
        layer_cfg: the layer's geometry.
        rope_cache: the layer's FP32 ``rotary_emb.cos_sin_cache`` ``[max_pos, 64]``.
        cu_seqlens: global document boundaries of the packed sequence (host ints
            or a tensor); ``None`` is one document.
        cp_group: context-parallel group; ranks hold contiguous row ranges.
        deterministic: run cuDNN's deterministic backward.
    """
    if not q.is_cuda:
        raise RuntimeError("the FlashMLA training core needs CUDA tensors")
    from . import load_vllm_flashmla

    load_vllm_flashmla()
    assert rope_cache.dtype == torch.float32 and rope_cache.shape[-1] == layer_cfg.rope_dim
    assert positions.dtype == torch.int64
    if layer_cfg.compress_ratio == 0:
        assert comp_latent is None and topk_local is None
    if cu_seqlens is not None and torch.is_tensor(cu_seqlens):
        cu_seqlens = cu_seqlens.tolist()
    data, scale, grad = _FlashMLAAttnCore.apply(
        q,
        swa_kv,
        comp_latent,
        attn_sink,
        topk_local,
        positions,
        cu_seqlens,
        layer_cfg,
        rope_cache,
        cp_group,
        deterministic,
    )
    return AttentionCoreOutput(data, scale, grad)

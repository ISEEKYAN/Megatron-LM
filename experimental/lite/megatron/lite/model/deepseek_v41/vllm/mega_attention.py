# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 mega attention core: the vLLM inference forward and an owned backward.

Forward is the BI=1 prefill of vLLM's ``FLASHMLA_MEGA_ATTN_DSV41`` layer
(vllm-project/vllm#62365 @ d667e410e1, ``_forward_prefill_mega``), op for op:

1. SWA KV: ``fused_deepseek_v4_kv_rope_insert`` (GPT-J RoPE on dims [448, 512),
   bf16 rounding, MXFP8 record) into a temporary paged cache;
2. compressed KV: ``rope_quant_insert`` (RoPE at ``pos // cr * cr``, NVFP4 record)
   from the per-token compressor latent, group-boundary rows only;
3. ``dequantize_and_gather_k_cache`` of both caches into one bf16 workspace,
   compressed rows at offset 0 and SWA rows at offset ``N``;
4. ``combine_swa_first_indices`` (``combine_topk_swa_indices`` for ratio 0);
5. ``fused_norm_rope_attn_rope_cast_fwd`` on the fused-layout query.

Each output row only depends on its own query, its index row and the KV rows
those index, so packing, chunking and context parallelism do not change bits.

Gradient contract (backward is ``csrc/mega_attention_bwd.cu`` plus torch glue):

* Output FP8 cast: identity STE. The upstream gradient is taken with respect to
  the dequantized output ``fp8 * 2**(sf - 127)`` in the kernel's fused layout
  ``[S, 8, 8 * 512]``; it arrives on the ``grad`` carrier of the output.
* Output inverse RoPE and query RoPE: exact rotation VJPs at the row position.
  The bf16 rounding of the rotated query is an identity STE.
* KV cache quantization (SWA MXFP8, compressed NVFP4) and the bf16 rounding of
  the rotated KV: identity STE (no clipping mask: neither record clips, its
  scale covers the block maximum). The gradient reaches the pre-RoPE ``swa_kv``
  and compressor latent through the exact KV RoPE VJP at the cache position.
* Attention: the derivative of ``softmax([s, sink]) @ K`` with ``K = V`` the
  dequantized KV the forward read, ``s = scale * rope(q) . K``, P recomputed in
  FP32 from the saved LSE (``logaddexp(lse, sink)``), and
  ``D = sum_j P_j dP_j`` recomputed from P, not from the visible FP8 output, so
  each row of dS sums to zero. The forward's bf16 rounding of P before the PV
  product is an identity STE. ``dsink = -sum_rows P_sink * D``.
* bf16 operands with FP32 accumulation; dKV accumulates with FP32 atomics, so
  dKV (and the KV input gradients) are not bitwise reproducible run to run.
"""
import math
from pathlib import Path
from typing import NamedTuple

import torch

HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
WV_GROUP = 8
QUANT_GROUP = 32
WINDOW = 128
TOP_K = 512
PAGE = 64
SWA_RECORD = 528
COMPRESSED_RECORD = 288


class MegaAttentionOutput(NamedTuple):
    """``data``/``scale`` feed ``wo_a``'s ``fp8_einsum`` as vLLM's ``_o_proj`` does.

    ``grad`` is a zero-strided bf16 placeholder of ``data``'s shape. It carries
    no values; the consumer's backward returns the gradient with respect to the
    dequantized output into it (FP8 output STE).
    """

    data: torch.Tensor
    scale: torch.Tensor
    grad: torch.Tensor


# ---- layouts ---------------------------------------------------------------


def q_from_fused(q):
    """Fused query layout ``[S, H, 512]`` (16-dim chunks across heads) to natural."""
    s, h, d = q.shape
    return q.reshape(s, d // 16, h, 16).transpose(1, 2).reshape(s, h, d)


def q_to_fused(q):
    s, h, d = q.shape
    return q.reshape(s, h, d // 16, 16).transpose(1, 2).reshape(s, h, d)


def o_from_fused(o):
    """Output layout ``[S, groups, 8 * 512]`` (32-dim chunks across 8 heads) to natural."""
    s, g, width = o.shape
    d = width // WV_GROUP
    return (
        o.reshape(s, g, d // 32, WV_GROUP, 32).permute(0, 1, 3, 2, 4).reshape(s, g * WV_GROUP, d)
    )


def dequantize_output(data, scale):
    """Dequantized ``[S, groups, 8 * 512]`` FP32 output of the mega kernel."""
    s, g, width = data.shape
    exponents = scale.contiguous().view(torch.uint8).reshape(s, g, width // 32).float() - 127
    return (data.float().reshape(s, g, -1, 32) * torch.exp2(exponents).unsqueeze(-1)).reshape(
        s, g, width
    )


def rope(x, cos_sin_cache, positions, inverse=False):
    """GPT-J RoPE of the last 64 dims at ``positions`` in FP32; ``inverse`` is the transpose."""
    cs = cos_sin_cache.index_select(0, positions.long()).float()
    cos, sin = cs[:, : ROPE_DIM // 2], cs[:, ROPE_DIM // 2 :]
    shape = (-1,) + (1,) * (x.dim() - 2) + (ROPE_DIM // 2,)
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    if inverse:
        sin = -sin
    x = x.float()
    even, odd = x[..., -ROPE_DIM::2], x[..., -ROPE_DIM + 1 :: 2]
    out = x.clone()
    out[..., -ROPE_DIM::2] = even * cos - odd * sin
    out[..., -ROPE_DIM + 1 :: 2] = odd * cos + even * sin
    return out


# ---- request plan ----------------------------------------------------------


class _Plan(NamedTuple):
    starts: list  # global first token of each request (document)
    seq_lens: list  # KV tokens of each request: up to its last local query row
    query_start_loc: list  # local query row offsets, len R + 1
    n: int  # compressed rows per request in the workspace
    m: int  # workspace rows per request


def _plan(cu_seqlens, row_start, row_end, ratio):
    starts, seq_lens, qsl = [], [], [0]
    for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        lo, hi = max(a, row_start), min(b, row_end)
        if lo < hi:
            starts.append(a)
            seq_lens.append(hi - a)
            qsl.append(qsl[-1] + hi - lo)
    if qsl[-1] != row_end - row_start:
        raise ValueError("cu_seqlens must cover every local query row")
    n = max(s // ratio for s in seq_lens) if ratio else 0
    return _Plan(starts, seq_lens, qsl, n, n + max(seq_lens))


def _forward(q, swa_kv, latent, topk, positions, sink, cos_sin_cache, ratio, window, top_k, plan):
    """Steps 1-5 of the module docstring; returns (out data, out scale, lse, kv, indices, lens)."""
    from vllm.models.deepseek_v41.common.ops import (
        combine_topk_swa_indices,
        dequantize_and_gather_k_cache,
    )
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert
    from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import alloc_mega_attn_output

    device = q.device
    i32 = dict(dtype=torch.int32, device=device)
    i64 = dict(dtype=torch.int64, device=device)
    requests = len(plan.starts)
    seq_lens = torch.tensor(plan.seq_lens, **i32)

    def paged(lengths, record):
        pages = [max(-(-length // PAGE), 1) for length in lengths]
        base = [sum(pages[:r]) for r in range(requests)]
        width = max(pages)
        table = torch.tensor(
            [[b + min(p, n - 1) for p in range(width)] for b, n in zip(base, pages)], **i32
        )
        cache = torch.zeros(sum(pages), PAGE, record, dtype=torch.uint8, device=device)
        return cache, table, base

    tokens = torch.cat(
        [torch.arange(a, a + s, **i64) for a, s in zip(plan.starts, plan.seq_lens)]
    )
    token_pos = torch.cat([torch.arange(s, **i64) for s in plan.seq_lens])
    swa_cache, swa_table, swa_base = paged(plan.seq_lens, SWA_RECORD)
    swa_slots = torch.cat(
        [torch.arange(s, **i64) + b * PAGE for b, s in zip(swa_base, plan.seq_lens)]
    )
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(
        swa_kv.index_select(0, tokens).contiguous(),
        swa_cache,
        swa_slots,
        token_pos,
        cos_sin_cache,
        PAGE,
        None,
        True,
    )
    kv = torch.empty(requests, plan.m, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if ratio:
        counts = [s // ratio for s in plan.seq_lens]
        comp_cache, comp_table, comp_base = paged(counts, COMPRESSED_RECORD)
        boundary = torch.cat(
            [torch.arange(c, **i64) * ratio + ratio - 1 for c in counts]
        )  # within-document boundary positions
        owners = torch.cat([torch.full((c,), a, **i64) for a, c in zip(plan.starts, counts)])
        comp_slots = torch.cat([torch.arange(c, **i64) + b * PAGE for b, c in zip(comp_base, counts)])
        rope_quant_insert(
            latent.index_select(0, owners + boundary).contiguous(),
            boundary,
            cos_sin_cache,
            comp_cache,
            comp_slots,
            ratio,
        )
        dequantize_and_gather_k_cache(
            kv[:, : plan.n],
            comp_cache,
            seq_lens=torch.tensor(counts, **i32),
            gather_lens=None,
            block_table=comp_table,
            block_size=PAGE,
            offset=0,
        )
    dequantize_and_gather_k_cache(
        kv,
        swa_cache,
        seq_lens=seq_lens,
        gather_lens=seq_lens,
        block_table=swa_table,
        block_size=PAGE,
        offset=plan.n,
    )
    qsl = torch.tensor(plan.query_start_loc, **i32)
    if ratio:
        from vllm.model_executor.kernels.batch_invariant import combine_swa_first_indices

        indices, lens = combine_swa_first_indices(
            topk, qsl, seq_lens, seq_lens, window, ratio, top_k, plan.m, plan.n
        )
    else:
        dummy = torch.full((q.shape[0], top_k), -1, **i32)
        indices, lens = combine_topk_swa_indices(
            dummy, qsl, seq_lens, seq_lens, window, 0, 0, plan.m, 0
        )
    out = alloc_mega_attn_output(q.shape[0], HEADS // WV_GROUP, device)
    _, _, _, lse = torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_fwd(
        q,
        kv.view(-1, 1, HEAD_DIM),
        indices.unsqueeze(1),
        HEAD_DIM**-0.5,
        HEAD_DIM,
        sink,
        lens,
        False,  # enable_q_norm
        0.0,
        positions.to(torch.int32),
        False,  # GPT-J RoPE
        ROPE_DIM,
        cos_sin_cache,
        HEADS // WV_GROUP,
        QUANT_GROUP,
        True,
        True,
        True,
        out.data,
        out.scale,
    )
    return out.data, out.scale, lse, kv, indices, lens


def _kv_rows(plan, ratio, device):
    """Per workspace row: (global token, RoPE position) for compressed and SWA rows."""
    i64 = dict(dtype=torch.int64, device=device)
    comp, swa = [], []
    for r, (a, s) in enumerate(zip(plan.starts, plan.seq_lens)):
        row = r * plan.m
        if ratio:
            j = torch.arange(s // ratio, **i64)
            comp.append(torch.stack([row + j, a + j * ratio + ratio - 1, j * ratio]))
        t = torch.arange(s, **i64)
        swa.append(torch.stack([row + plan.n + t, a + t, t]))
    cat = lambda parts: torch.cat(parts, 1) if parts else torch.empty(3, 0, **i64)
    return cat(comp), cat(swa)


_EXTENSION = None


def _extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="ds41_mega_attention_bwd",
            sources=[str(Path(__file__).with_name("csrc") / "mega_attention_bwd.cu")],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a"],
            verbose=False,
        )
    return _EXTENSION


def attention_backward_kernel(qr, grad, kv, indices, lens, lse_full, scale):
    """Owned sparse-attention VJP on rotated operands (no RoPE, no layouts).

    ``qr``/``grad`` bf16 ``[S, 64, 512]`` natural layout; ``kv`` bf16 ``[M, 512]``;
    ``lse_full`` FP32 ``[S, 64]`` natural-log LSE including the sink. Returns
    FP32 ``dq [S, 64, 512]``, ``dkv [M, 512]`` and ``D [S, 64]``.
    """
    dq = torch.empty(qr.shape, dtype=torch.float32, device=qr.device)
    dkv = torch.zeros(kv.shape, dtype=torch.float32, device=qr.device)
    delta = torch.empty(qr.shape[:2], dtype=torch.float32, device=qr.device)
    _extension().backward(
        qr.contiguous(),
        grad.contiguous(),
        kv.contiguous(),
        indices.contiguous(),
        lens.contiguous(),
        (lse_full * math.log2(math.e)).contiguous(),
        float(scale),
        dq,
        dkv,
        delta,
    )
    return dq, dkv, delta


class _MegaAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, swa_kv, latent, topk, positions, sink, cos_sin_cache, ratio, window, top_k,
        cu_seqlens, group,
    ):
        rank, size = (group.rank(), group.size()) if group is not None else (0, 1)
        local = q.shape[0]
        swa_full, latent_full = swa_kv.detach(), None if latent is None else latent.detach()
        if size > 1:
            swa_full = _all_gather(swa_full, group)
            latent_full = None if latent_full is None else _all_gather(latent_full, group)
        total = local * size
        cu = [0, total] if cu_seqlens is None else [int(x) for x in cu_seqlens]
        plan = _plan(cu, rank * local, (rank + 1) * local, ratio)
        sink32 = sink.detach().float().contiguous()
        data, scale, lse, kv, indices, lens = _forward(
            q.detach().contiguous(), swa_full.bfloat16().contiguous(),
            None if latent_full is None else latent_full.bfloat16().contiguous(),
            None if topk is None else topk.to(torch.int32).contiguous(),
            positions, sink32, cos_sin_cache, ratio, window, top_k, plan,
        )
        ctx.save_for_backward(q, kv, indices, lens, lse, positions, sink32, cos_sin_cache)
        ctx.meta = (plan, ratio, group, rank, size, swa_kv.dtype, swa_kv.shape,
                    None if latent is None else (latent.dtype, latent.shape), sink.dtype)
        ctx.mark_non_differentiable(data, scale)
        carrier = torch.zeros((), dtype=torch.bfloat16, device=q.device).expand(data.shape)
        return data, scale, carrier

    @staticmethod
    def backward(ctx, _data, _scale, grad):
        q, kv, indices, lens, lse, positions, sink, cos_sin_cache = ctx.saved_tensors
        plan, ratio, group, rank, size, swa_dtype, swa_shape, latent_meta, sink_dtype = ctx.meta
        if grad is None:
            grad = torch.zeros(q.shape[0], HEADS // WV_GROUP, WV_GROUP * HEAD_DIM, device=q.device)
        # o = R(-theta) O, so dL/dO = R(theta) dL/do.
        grad_o = rope(o_from_fused(grad.float()), cos_sin_cache, positions).bfloat16()
        qr = rope(q_from_fused(q), cos_sin_cache, positions).bfloat16()
        lse_full = torch.logaddexp(lse, sink.unsqueeze(0))
        scale = HEAD_DIM**-0.5
        dqr, dkv, delta = attention_backward_kernel(
            qr, grad_o, kv.view(-1, HEAD_DIM), indices, lens, lse_full, scale
        )
        dq = q_to_fused(rope(dqr, cos_sin_cache, positions, inverse=True)).to(q.dtype)
        dsink = -(torch.exp(sink.unsqueeze(0) - lse_full) * delta).sum(0).to(sink_dtype)
        comp_rows, swa_rows = _kv_rows(plan, ratio, q.device)
        total = q.shape[0] * size
        dswa = torch.zeros(total, HEAD_DIM, dtype=torch.float32, device=q.device)
        dswa[swa_rows[1]] = rope(dkv[swa_rows[0]], cos_sin_cache, swa_rows[2], inverse=True)
        dlatent = None
        if latent_meta is not None:
            dlatent = torch.zeros(total, HEAD_DIM, dtype=torch.float32, device=q.device)
            dlatent[comp_rows[1]] = rope(
                dkv[comp_rows[0]], cos_sin_cache, comp_rows[2], inverse=True
            )
        if size > 1:
            dswa = _reduce_scatter(dswa, group)
            dlatent = None if dlatent is None else _reduce_scatter(dlatent, group)
        dswa = dswa.to(swa_dtype)
        if dlatent is not None:
            dlatent = dlatent.to(latent_meta[0])
        return dq, dswa, dlatent, None, None, dsink, None, None, None, None, None, None


def _all_gather(x, group):
    out = x.new_empty((x.shape[0] * group.size(),) + tuple(x.shape[1:]))
    torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter(x, group):
    out = x.new_empty((x.shape[0] // group.size(),) + tuple(x.shape[1:]))
    torch.distributed.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def mega_attention(
    q,
    swa_kv,
    comp_latent,
    topk_local,
    positions,
    attn_sink,
    *,
    compress_ratio,
    cos_sin_cache,
    window=WINDOW,
    top_k=TOP_K,
    cu_seqlens=None,
    cp_group=None,
):
    """Mega attention core of one DS4.1 layer (forward bitwise = vLLM BI=1 prefill).

    Args:
        q: bf16 ``[S_local, 64, 512]`` ``wq_b`` output in the fused query layout.
        swa_kv: bf16 ``[S_local, 512]`` ``kv_norm`` output, before RoPE.
        comp_latent: bf16 ``[S_local, 512]`` per-token compressor latent (as vLLM's
            ``fused_save_compress_norm``; only group-boundary rows are read and
            receive gradient), or ``None`` for ratio 0.
        topk_local: int32 ``[S_local, top_k]`` document-local compressed indices,
            ascending, ``-1`` at the end; ``None`` for ratio 0.
        positions: ``[S_local]`` within-document positions of the local rows.
        attn_sink: ``[64]`` sink logits.
        compress_ratio: 0, 1 or 2.
        cos_sin_cache: FP32 ``[max_pos, 64]``, vLLM's ``rotary_emb.cos_sin_cache``.
        cu_seqlens: global document offsets over all CP ranks' rows (THD);
            ``None`` is one document.
        cp_group: context-parallel group; ranks hold contiguous equal row slices.

    Returns:
        :class:`MegaAttentionOutput`.
    """
    if q.shape[1:] != (HEADS, HEAD_DIM) or swa_kv.shape != (q.shape[0], HEAD_DIM):
        raise ValueError("mega attention expects q [S, 64, 512] and swa_kv [S, 512]")
    if (compress_ratio == 0) != (comp_latent is None) or (compress_ratio == 0) != (
        topk_local is None
    ):
        raise ValueError("ratio-0 layers take no compressed latent or top-k; others need both")
    if cp_group is not None and cp_group.size() == 1:
        cp_group = None
    return MegaAttentionOutput(
        *_MegaAttention.apply(
            q, swa_kv, comp_latent, topk_local, positions, attn_sink, cos_sin_cache,
            compress_ratio, window, top_k,
            None if cu_seqlens is None else torch.as_tensor(cu_seqlens).tolist(),
            cp_group,
        )
    )

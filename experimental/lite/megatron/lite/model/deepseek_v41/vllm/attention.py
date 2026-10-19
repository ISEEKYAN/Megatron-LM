# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 attention outside the core: vLLM BI=1 mega-attention prefill, op for op.

Forward follows ``DeepseekV4MegaAttnAttention`` (vllm-project/vllm#62365 @ d667e410e1,
flags of ``B_bi1_mega``) for a packed THD batch:

1. ``qr_kv``: BI MXFP8 GEMM of ``cat(wq_a, wkv)`` (FlashInfer activation quant).
2. ``fused_q_kv_rmsnorm_quant``: q_norm + MXFP8 Q (shared by both wq_b), kv_norm.
3. ``wq_b``: BI MXFP8 GEMM on the prequantized Q, output rows in the mega
   kernel's fused layout (``q_fused_permutation``, applied after quantization
   exactly as vLLM's load-time row gather).
4. compressor (KV-source layers): BF16 BI GEMM ``cat(wkv, wgate)`` to FP32, then
   ``fused_save_compress_norm`` (softmax-gated pooling for ratio 2, RMSNorm).
5. indexer (index-source layers, frozen): see :func:`indexer_topk`.
6. ``wo_a``: DeepGEMM ``fp8_einsum`` over the core's ``out_fp8/out_sf`` with
   ``wo_a`` columns in ``o_fused_permutation``; ``wo_b``: BI MXFP8 GEMM.

Gradients: GEMMs use the decoded-operand VJPs of :mod:`.linear`; every
quantization is an identity STE; norms and pooling replay their FP32 reference.
"""

from types import SimpleNamespace

import torch

from .linear import _row_scales, linear

HEADS, HEAD_DIM, ROPE_DIM, GROUPS = 64, 512, 64, 8
INDEX_HEADS, INDEX_DIM, TOP_K, WINDOW = 32, 128, 512, 128
STATE_ROWS = 8


# ---- q / kv norms -------------------------------------------------------------


def _rms_vjp(x, w, grad, eps):
    with torch.autocast(device_type=x.device.type, enabled=False):
        v = x.float()
        inv = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
        g = grad.float() * w.bfloat16().float()
        dx = inv * (g - v * inv.square() * (g * v).mean(-1, keepdim=True))
        dw = (grad.float() * v * inv).sum(0)
    return dx.to(x.dtype), dw.to(w.dtype)


class _QKVNorm(torch.autograd.Function):
    """vLLM ``fused_q_kv_rmsnorm_quant``; Q leaves as MXFP8 codes + swizzled scales.

    ``q`` (differentiable) is the decoded MXFP8 value; the MXFP8 cast is an
    identity STE onto the BF16 RMSNorm output.
    """

    @staticmethod
    def forward(ctx, qr, kv, q_weight, kv_weight, eps):
        from vllm.models.deepseek_v41.common.ops.query_quant import fused_q_kv_rmsnorm_quant

        qa, kv_out = fused_q_kv_rmsnorm_quant(
            qr, kv, q_weight.bfloat16().contiguous(), kv_weight.bfloat16().contiguous(), eps
        )
        raw = _row_scales(qa.scale, *qr.shape)
        q = (
            (
                qa.data.float().reshape(qr.shape[0], -1, 32)
                * torch.exp2(raw.float() - 127).unsqueeze(-1)
            )
            .reshape(qr.shape)
            .bfloat16()
        )
        ctx.save_for_backward(qr, kv, q_weight, kv_weight)
        ctx.eps = eps
        ctx.mark_non_differentiable(qa.data, qa.scale)
        return q, kv_out, qa.data, qa.scale

    @staticmethod
    def backward(ctx, dq, dkv, _codes, _scales):
        qr, kv, q_weight, kv_weight = ctx.saved_tensors
        dqr = dqw = dkv_in = dkvw = None
        if dq is not None:
            dqr, dqw = _rms_vjp(qr, q_weight, dq, ctx.eps)
        if dkv is not None:
            dkv_in, dkvw = _rms_vjp(kv, kv_weight, dkv, ctx.eps)
        return dqr, dkv_in, dqw, dkvw, None


def q_fused_permutation(device):
    from vllm.models.deepseek_v41.common.ops.fused_layout import q_fused_permutation as perm

    return perm(HEADS, HEAD_DIM).to(device)


def o_fused_permutation(device):
    from vllm.models.deepseek_v41.common.ops.fused_layout import o_fused_permutation as perm

    return perm(GROUPS, HEAD_DIM).to(device)


def projections(x, p, eps, fused=True):
    """Steps 1-3. ``p``: wq_a, wkv, q_norm, kv_norm, wq_b masters.

    Returns q (fused layout, bf16 [T,64,512]), kv (bf16 [T,512], normed, no
    RoPE) and the shared Q quantization ``(q_decoded, codes, scales)``.
    """
    qr_kv = linear(x, torch.cat((p.wq_a, p.wkv)), fp8=True)
    qr, kv = qr_kv.split((p.wq_a.shape[0], p.wkv.shape[0]), -1)
    q_dec, kv, codes, scales = _QKVNorm.apply(qr, kv, p.q_norm, p.kv_norm, eps)
    shared = SimpleNamespace(data=codes, scale=scales)
    perm = q_fused_permutation(x.device) if fused else None
    q = linear(q_dec, p.wq_b, fp8=True, activation=shared, row_perm=perm)
    return q.view(-1, HEADS, HEAD_DIM), kv, (q_dec, shared)


# ---- compressor ---------------------------------------------------------------


def _segments(cu_seqlens, total, device):
    cu = [0, total] if cu_seqlens is None else [int(v) for v in cu_seqlens]
    lens = [b - a for a, b in zip(cu[:-1], cu[1:])]
    owner = torch.cat(
        [torch.full((n,), r, dtype=torch.int32, device=device) for r, n in enumerate(lens)]
    )
    return cu, lens, owner


def _compress_reference(kv_score, norm_weight, eps, ratio, rows):
    """FP32 replay of fused_save_compress_norm at group-boundary ``rows``."""
    kv = kv_score[:, :HEAD_DIM]
    if ratio == 2:
        pair = torch.stack((kv[rows - 1], kv[rows]))
        score = torch.stack((kv_score[rows - 1, HEAD_DIM:], kv_score[rows, HEAD_DIM:]))
        pooled = (pair * torch.softmax(score, 0)).sum(0)
    else:
        pooled = kv[rows]
    inv = torch.rsqrt(pooled.square().sum(-1, keepdim=True) / HEAD_DIM + eps)
    return pooled * inv * norm_weight.bfloat16().float()


class _Compress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv_score, positions, norm_weight, eps, ratio, cu_seqlens):
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
            fused_save_compress_norm,
        )

        tokens = kv_score.shape[0]
        cu, lens, owner = _segments(cu_seqlens, tokens, kv_score.device)
        latent = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16, device=kv_score.device)
        if ratio == 2:
            state = torch.zeros(
                len(lens), STATE_ROWS, 2 * HEAD_DIM, dtype=torch.float32, device=kv_score.device
            )
            slots = owner.long() * STATE_ROWS + positions.long() % STATE_ROWS
            qsl = torch.tensor(cu, dtype=torch.int32, device=kv_score.device)
            fused_save_compress_norm(
                kv_score,
                positions,
                state,
                slots,
                qsl,
                owner,
                norm_weight.bfloat16(),
                eps,
                ratio,
                latent,
            )
        else:
            fused_save_compress_norm(
                kv_score,
                positions,
                None,
                positions.clone(),
                None,
                None,
                norm_weight.bfloat16(),
                eps,
                ratio,
                latent,
            )
        rows = ((positions + 1) % ratio == 0).nonzero().flatten()
        ctx.save_for_backward(kv_score, norm_weight, rows)
        ctx.options = (eps, ratio)
        return latent

    @staticmethod
    def backward(ctx, grad):
        kv_score, norm_weight, rows = ctx.saved_tensors
        eps, ratio = ctx.options
        with torch.enable_grad(), torch.autocast(device_type=grad.device.type, enabled=False):
            score = kv_score.detach().requires_grad_(ctx.needs_input_grad[0])
            weight = norm_weight.detach().requires_grad_(ctx.needs_input_grad[2])
            out = _compress_reference(score, weight, eps, ratio, rows)
            wanted = [t for t in (score, weight) if t.requires_grad]
            grads = iter(torch.autograd.grad(out, wanted, grad[rows].float()))
        dscore = next(grads) if score.requires_grad else None
        dweight = next(grads).to(norm_weight.dtype) if weight.requires_grad else None
        return dscore, None, dweight, None, None, None


class ContextParallel(SimpleNamespace):
    """Contiguous equal CP row slices: ``group``, ``offset`` (first local row) and the
    global ``positions`` of all ranks' rows."""


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = x.new_empty((x.shape[0] * group.size(), *x.shape[1:]))
        torch.distributed.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty((grad.shape[0] // ctx.group.size(), *grad.shape[1:]))
        torch.distributed.reduce_scatter_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


def compressor(x, positions, wkv, wgate, norm_weight, eps, ratio, cu_seqlens=None, cp=None):
    """Per-token latent bf16 [T,512] (valid at ``(pos+1) % ratio == 0`` rows only).

    With ``cp`` the scores are gathered and the latent covers all ranks' rows."""
    weight = wkv if wgate is None else torch.cat((wkv, wgate))
    kv_score = linear(x, weight, fp8=False)
    if cp is not None:
        kv_score = _GatherRows.apply(kv_score, cp.group)
        positions = cp.positions
    return _Compress.apply(
        kv_score.contiguous(), positions.contiguous(), norm_weight, eps, ratio, cu_seqlens
    )


# ---- indexer (frozen) -----------------------------------------------------------


class IndexerState:
    """K bytes of an owning indexer and the candidate blocks, for later layers."""

    def __init__(self):
        self.keys = {}  # kv-source layer -> (k values fp8 [Nc,128], k scales int32 [Nc])
        self.candidates = None
        self.topk = None
        self.q = None


@torch.no_grad()
def indexer_topk(
    x,
    positions,
    qa,
    latent,
    p,
    *,
    ratio,
    cos_sin_cache,
    eps,
    state,
    source,
    owns_k,
    candidate,
    cu_seqlens=None,
    cp=None,
):
    """Top-k compressed indices of every row, vLLM's BI prefill indexer.

    ``qa``: the shared MXFP8 Q (codes, swizzled scales); ``latent``: the KV
    source's per-token compressor latent; ``p``: wq_b, weights_proj (+ wk, k_norm
    when ``owns_k``). ``candidate``: None, "write" or "read". Returns int32
    [T, 512] document-local indices, ascending, -1 at the tail. With ``cp``,
    ``latent`` covers all ranks' rows and only the local query rows are scored.
    """
    from vllm.model_executor.kernels import batch_invariant
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _apply_candidate_mask,
        _select_candidate_blocks,
        as_fp32_logits,
    )
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import fused_indexer_q_rope_quant
    from vllm.models.deepseek_v41.common.ops.indexer_k_store import indexer_k_norm_rope_store
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    device, tokens = x.device, x.shape[0]
    offset, k_positions = (0, positions) if cp is None else (cp.offset, cp.positions)
    cu, lens, k_owner = _segments(cu_seqlens, k_positions.shape[0], device)
    owner = k_owner[offset : offset + tokens]
    counts = [n // ratio for n in lens]
    k_start = torch.tensor([sum(counts[:r]) for r in range(len(lens))], device=device)
    if owns_k:
        k_pre = linear(latent, p.wk, fp8=False).to(torch.bfloat16)
        page = 64
        pages = -(-max(sum(counts), 1) // page)
        cache = torch.zeros(pages, page * (INDEX_DIM + 4), dtype=torch.uint8, device=device)
        cache = cache.view(pages, page, INDEX_DIM + 4)
        slots = k_start[k_owner.long()] + k_positions.long() // ratio
        indexer_k_norm_rope_store(
            k_pre, k_positions, cos_sin_cache, p.k_norm.bfloat16(), eps, cache, slots, ratio, False
        )
        flat = cache.view(pages, -1)
        values = flat[:, : page * INDEX_DIM].reshape(-1, INDEX_DIM)[: sum(counts)]
        scales = flat[:, page * INDEX_DIM :].reshape(-1, 4)[: sum(counts)]
        state.keys[source] = (
            values.view(torch.float8_e4m3fn).contiguous(),
            scales.contiguous().view(torch.int32).squeeze(-1),
        )
    k_values, k_scales = state.keys[source]

    weights = linear(x, p.weights_proj, fp8=False).to(torch.bfloat16)
    q = linear(qa[0], p.wq_b, fp8=True, activation=qa[1]).view(-1, INDEX_HEADS, INDEX_DIM)
    q_fp8, w = fused_indexer_q_rope_quant(
        positions,
        q,
        cos_sin_cache,
        weights,
        INDEX_DIM**-0.5,
        INDEX_HEADS**-0.5,
        use_fp4=False,
        weights_out_dtype=torch.float32,
    )
    state.q = (q_fp8, w)
    q_scale, w = mqa_logits_mx_inputs(q_fp8, None, w)
    ks = k_start[owner.long()].to(torch.int32)
    ke = ks + ((positions + 1) // ratio).to(torch.int32)
    out = torch.full((tokens, TOP_K), -1, dtype=torch.int32, device=device)
    candidates = state.candidates if candidate == "read" else None
    if candidate == "write":
        state.candidates = torch.full((tokens, 2048), -1, dtype=torch.int32, device=device)
        candidates = state.candidates
    for r, (a, b) in enumerate(zip(cu[:-1], cu[1:])):
        a, b = max(a, offset) - offset, min(b, offset + tokens) - offset
        if a >= b:
            continue
        logits = fp8_fp4_mqa_logits(
            (q_fp8[a:b], q_scale[a:b]), (k_values, k_scales), w[a:b], ks[a:b], ke[a:b], counts[r]
        )
        row_ends = ke[a:b] - ks[a:b]
        logits = as_fp32_logits(logits, row_ends)
        if candidate == "write":
            _select_candidate_blocks(logits, None, row_ends, 2048, 8, candidates[a:b])
        elif candidate == "read":
            _apply_candidate_mask(logits, None, row_ends, candidates[a:b], 8)
        batch_invariant.top_k_per_row_prefill_index_order(
            logits,
            torch.zeros_like(row_ends),
            row_ends,
            out[a:b],
            b - a,
            logits.stride(0),
            logits.stride(1),
            TOP_K,
        )
    state.topk = out
    return out


# ---- output projections ---------------------------------------------------------


class _WoA(torch.autograd.Function):
    """``fp8_einsum('bhr,hdr->bhd')`` on the core's FP8 output; STE into its carrier."""

    @staticmethod
    def forward(ctx, data, scale, carrier, weight, fused):
        from megatron.lite.primitive.quantization.block_fp8 import (
            dequantize_block_fp8,
            quantize_block_fp8,
        )
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            deepgemm_post_process_fp8_weight_block,
        )
        from vllm.utils.deep_gemm import fp8_einsum

        width = GROUPS * HEAD_DIM
        perm = o_fused_permutation(weight.device) if fused else None
        fused = (
            weight.reshape(-1, width) if perm is None else weight.reshape(-1, width)[:, perm]
        )  # [8*1024, 4096], 32-wide column blocks move whole
        codes, block_scales = quantize_block_fp8(fused, (32, 32), scale_format="e8m0")
        rows = (block_scales.view(torch.uint8).repeat_interleave(32, 0).to(torch.int32) << 23).view(
            torch.float32
        )
        wq, ws = deepgemm_post_process_fp8_weight_block(
            wq=codes,
            ws=rows,
            quant_block_shape=(1, 32),
            use_e8m0=False,
            is_bmm=True,
            bmm_batch_size=GROUPS,
        )
        z = torch.empty(
            (
                data.shape[0],
                GROUPS,
                weight.shape[-2] if weight.dim() == 3 else weight.shape[0] // GROUPS,
            ),
            dtype=torch.bfloat16,
            device=data.device,
        )
        if data.shape[0]:
            fp8_einsum(
                "bhr,hdr->bhd",
                (data[:, :GROUPS], scale[:, :GROUPS]),
                (wq, ws),
                z,
                recipe=(1, 1, 32),
            )
        decoded_w = dequantize_block_fp8(codes, block_scales, (32, 32)).to(torch.bfloat16)
        ctx.save_for_backward(data, scale, decoded_w, perm)
        ctx.weight_shape, ctx.weight_dtype = weight.shape, weight.dtype
        return z

    @staticmethod
    def backward(ctx, grad):
        data, scale, decoded_w, perm = ctx.saved_tensors
        with torch.autocast(device_type=grad.device.type, enabled=False):
            o = dequantize_output(data, scale).float()  # [T, 8, 4096] fused layout
            w = decoded_w.float().view(GROUPS, -1, GROUPS * HEAD_DIM)  # [g, d, r]
            g = grad.float()  # [T, g, d]
            do = torch.einsum("bhd,hdr->bhr", g, w)
            dw_fused = torch.einsum("bhd,bhr->hdr", g, o).reshape(-1, GROUPS * HEAD_DIM)
            if perm is None:
                dw = dw_fused
            else:
                dw = torch.empty_like(dw_fused)
                dw[:, perm] = dw_fused
        return (
            None,
            None,
            do.to(torch.bfloat16),
            dw.reshape(ctx.weight_shape).to(ctx.weight_dtype),
            None,
        )


def dequantize_output(data, scale):
    """FP8 ``[T, 8, 4096]`` + MN-major packed UE8M0 -> FP32 in the fused layout."""
    tokens = data.shape[0]
    raw = scale[:tokens].contiguous().view(torch.uint8).reshape(tokens, GROUPS, -1)
    blocks = raw[..., : GROUPS * HEAD_DIM // 32]
    sf = torch.exp2(blocks.float() - 127)
    return (data.float().reshape(tokens, GROUPS, -1, 32) * sf.unsqueeze(-1)).reshape(data.shape)


def o_projection(out, wo_a, wo_b, fused=True):
    """``out``: (data, scale, carrier) from the attention core. Returns bf16 [T, 5120]."""
    z = _WoA.apply(out[0], out[1], out[2], wo_a, fused)
    return linear(z.flatten(1), wo_b, fp8=True)

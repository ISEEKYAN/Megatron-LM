# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-visible sparse arithmetic and owned gradients."""

import torch
from torch.autograd.function import once_differentiable


@torch.no_grad()
def indexer_scores(query, keys, weights, lengths):
    """Native SM100 BF16 MXFP4 index scores for the deployment selector.

    Inputs may be rollout Q/K packets or owned QAT decoded values. Index ranking has no VJP in the
    CSA recipe; attention keeps its ordinary VJP with the selected slots fixed.
    """
    from .cache import IndexCache

    key_tensor = keys.packed if isinstance(keys, IndexCache) else keys
    query_tensor = query.packed if isinstance(query, IndexCache) else query
    if not all(t.is_cuda for t in (query_tensor, key_tensor, weights, lengths)):
        raise RuntimeError('Deployment index scoring requires CUDA')
    if query.ndim != 4 or key_tensor.ndim != 3:
        raise ValueError('Index Q/K must be BSHD/BTD')
    batch, rows, heads, width = query.shape
    if heads != 32 or width != 128 or keys.shape[0] != batch or keys.shape[-1] != width:
        raise ValueError('Deployment index scoring requires paired H32/D128 Q/K')
    if weights.shape != query.shape[:-1]:
        raise ValueError('Index head weights must be BSH')
    columns = keys.shape[1]
    if rows == 0 or columns == 0:
        return query_tensor.new_empty((batch, rows, columns), dtype=torch.float32)
    from megatron.lite.primitive.quantization.mxfp4 import quantize_index
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    q = query if isinstance(query, IndexCache) else quantize_index(query)
    k = keys if isinstance(keys, IndexCache) else quantize_index(keys)
    q_scale = q.scale.contiguous().view(torch.int32).squeeze(-1)
    k_scale = k.scale.contiguous().view(torch.int32).squeeze(-1)
    ends = torch.broadcast_to(lengths, (batch, rows, 1)).to(torch.int32)
    # Keep each request's KV base aligned to the native 128-column tile.
    # One native launch covers all query rows. The native output is compact:
    # every row starts at its request-local column zero.
    pitch = (columns + 127) // 128 * 128
    packed_k = torch.nn.functional.pad(k.packed, (0, 0, 0, pitch - columns))
    packed_scale = torch.nn.functional.pad(k_scale, (0, pitch - columns))
    starts = torch.arange(batch, device=query.device, dtype=torch.int32) * pitch
    starts = starts[:, None].expand(batch, rows).contiguous().flatten()
    scores = fp8_fp4_mqa_logits(
        (q.packed.flatten(0, 1).view(torch.int8), q_scale.flatten(0, 1)),
        (packed_k.flatten(0, 1).view(torch.int8), packed_scale.flatten(0, 1)),
        weights.flatten(0, 1).bfloat16(),
        starts,
        (starts + ends.flatten()).contiguous(),
        max_seqlen_k=columns,
    )
    # FP32 storage serves the Top-K API; the native scores already rounded to
    # BF16. DeepGEMM leaves the tail undefined, so never expose it to selection.
    scores = scores.reshape(batch, rows, columns).float()
    valid = torch.arange(columns, device=query.device) < ends
    return scores.masked_fill(~valid, -torch.inf)


@torch.no_grad()
def indexer_select_positions(scores, lengths, topk, *, candidates=None):
    """Use the BI receiver's prefill selector and ascending valid KV positions."""
    if not scores.is_cuda or scores.dtype != torch.float32 or scores.ndim != 3:
        raise RuntimeError('Deployment index selection requires CUDA BSN FP32 scores')
    if topk not in (512, 1024, 2048):
        raise ValueError('Deployment index Top-K must be 512, 1024, or 2048')
    batch, rows, columns = scores.shape
    if rows == 0 or columns == 0:
        return torch.empty((batch, rows, 0), dtype=torch.int32, device=scores.device)
    from vllm.model_executor.kernels.batch_invariant import top_k_per_row_prefill

    flat = scores.flatten(0, 1)
    if candidates is not None:
        if candidates.shape != scores.shape or candidates.dtype != torch.bool:
            raise ValueError('Candidate mask must be boolean and match scores')
        masked = torch.empty_strided(
            flat.shape, flat.stride(), device=flat.device, dtype=flat.dtype
        )
        masked.copy_(flat).masked_fill_(~candidates.flatten(0, 1), -torch.inf)
        flat = masked
    ends = (
        torch.broadcast_to(lengths, (batch, rows, 1))
        .to(torch.int32)
        .flatten()
        .contiguous()
    )
    starts = torch.zeros_like(ends)
    selected = torch.empty(
        (batch * rows, topk), device=scores.device, dtype=torch.int32
    )
    top_k_per_row_prefill(
        flat,
        starts,
        ends,
        selected,
        selected.shape[0],
        flat.stride(0),
        flat.stride(1),
        topk,
    )
    valid = selected >= 0
    safe = selected.clamp(min=0, max=columns - 1).long()
    valid &= flat.gather(1, safe) > -torch.inf
    selected = (
        torch.where(valid, selected, columns).sort(-1).values[:, : min(topk, columns)]
    )
    return torch.where(selected < columns, selected, -1).reshape(
        batch, rows, selected.shape[-1]
    )


# Query rows per reference VJP chunk. The fp32 reference gathers
# [rows, width, dim] KV, which is 15.8 GiB for 12928 rows at width 640.
SPARSE_VJP_ROW_CHUNK = 1024


class _ChunkedRowVJP(torch.autograd.Function):
    """Visible forward; the reference VJP is replayed over query-row chunks.

    Each output row depends only on its own query row and the shared values and
    sink, so per-chunk VJPs sum to the full one. Shared gradients accumulate in
    FP32 in a fixed chunk order and round to the input dtype once.
    """

    @staticmethod
    def forward(ctx, visible, reference_rows, chunk, query, values, bias):
        if chunk <= 0:
            raise ValueError("Sparse VJP chunk must be positive")
        ctx.reference_rows, ctx.chunk = reference_rows, chunk
        ctx.save_for_backward(query, values, bias)
        return visible(query, values, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        query, values, bias = ctx.saved_tensors
        need_q, need_v, need_b = ctx.needs_input_grad[3:]
        grad_q = torch.empty_like(query) if need_q else None
        grad_v = torch.zeros_like(values, dtype=torch.float32) if need_v else None
        grad_b = torch.zeros_like(bias, dtype=torch.float32) if need_b else None
        if need_q or need_v or need_b:
            with torch.enable_grad(), torch.autocast(
                device_type=query.device.type, enabled=False
            ):
                # FP32 leaf: the reference reads values.float(), so the forward
                # is unchanged and each chunk's gradient stays unrounded.
                shared_v = values.detach().float().requires_grad_(need_v)
                shared_b = bias.detach().requires_grad_(need_b)
                rows = query.shape[1]
                for start in range(0, rows, ctx.chunk):
                    stop = min(start + ctx.chunk, rows)
                    rows_q = query[:, start:stop].detach().requires_grad_(need_q)
                    output = ctx.reference_rows(rows_q, shared_v, shared_b, start, stop)
                    wanted = [
                        t for t in (rows_q, shared_v, shared_b) if t.requires_grad
                    ]
                    grads = iter(
                        torch.autograd.grad(output, wanted, grad[:, start:stop])
                    )
                    if need_q:
                        grad_q[:, start:stop] = next(grads)
                    if need_v:
                        grad_v += next(grads).float()
                    if need_b:
                        grad_b += next(grads).float()
        return (
            None,
            None,
            None,
            grad_q,
            None if grad_v is None else grad_v.to(values.dtype),
            None if grad_b is None else grad_b.to(bias.dtype),
        )


def sparse_attention(
    q, kv, sink, mask, main_indices, *, main_size, window_size, width, scale
):
    if q.shape[1] == 0:
        # A CP rank can own no query rows of a document; FlashMLA rejects an
        # empty launch. Keep the graph to kv and sink (zero gradients) so every
        # CP rank runs the same backward collectives.
        return q.new_zeros(q.shape) + (kv.sum() + sink.sum()).to(q.dtype) * 0
    # Native prefill gathers compressed KV before SWA and preserves selected
    # index order. Construct the same physical row order from the owned state.
    reordered = torch.cat([kv[:, window_size:], kv[:, :window_size]], 1)
    from megatron.lite.primitive.kernels.sparse_metadata import sparse_metadata

    indices, lengths = sparse_metadata(
        mask, main_indices, main_size=main_size, window_size=window_size, width=width
    )
    batch, rows, heads, dim = q.shape
    offsets = torch.arange(batch, device=q.device, dtype=torch.int32)[:, None, None]
    flat_indices = torch.where(indices >= 0, indices + offsets * kv.shape[1], -1)

    return sparse_attention_from_indices(
        q, reordered, sink, flat_indices, lengths, scale=scale
    )


def sparse_attention_from_indices(q, kv, sink, flat_indices, lengths, *, scale):
    """Use caller-owned physical row addresses without rebuilding sparse slots."""
    batch, rows, heads, dim = q.shape
    if flat_indices.shape[:2] != (batch, rows) or lengths.shape != (batch, rows):
        raise ValueError('Sparse addresses must cover every query row')
    if flat_indices.dtype != torch.int32 or lengths.dtype != torch.int32:
        raise ValueError('Sparse addresses and lengths must be int32')

    def reference_rows(query, values, bias, start, stop):
        selected = flat_indices[:, start:stop]
        gathered = values.flatten(0, 1)[selected.clamp_min(0).long()]
        logits = (
            torch.einsum('bshd,bstd->bsht', query.float(), gathered.float()) * scale
        )
        logits = logits.masked_fill((selected < 0).unsqueeze(2), -torch.inf)
        probabilities = torch.cat(
            [logits, bias.expand(batch, stop - start, -1).unsqueeze(-1)], -1
        ).softmax(-1)[..., :-1]
        return torch.einsum('bsht,bstd->bshd', probabilities, gathered.float()).to(
            query.dtype
        )

    def visible(query, values, bias):
        if not query.is_cuda:
            return reference_rows(query, values, bias, 0, query.shape[1])
        from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

        padded_heads = 64 if heads <= 64 else 128
        padded_sink = torch.nn.functional.pad(
            bias, (0, padded_heads - heads)
        ).contiguous()
        padded_q = torch.nn.functional.pad(
            query.flatten(0, 1), (0, 0, 0, padded_heads - heads)
        ).contiguous()
        output = torch.empty_like(padded_q)
        flash_mla_sparse_fwd(
            q=padded_q,
            kv=values.flatten(0, 1).unsqueeze(1).contiguous(),
            indices=flat_indices.flatten(0, 1).unsqueeze(1),
            sm_scale=scale,
            attn_sink=padded_sink,
            topk_length=lengths.flatten(),
            out=output,
        )
        return output[:, :heads].reshape(batch, rows, heads, dim)

    if rows == 0:
        from .autograd import visible_forward

        return visible_forward(
            visible,
            lambda query, values, bias: reference_rows(query, values, bias, 0, 0),
            q,
            kv,
            sink,
        )
    return _ChunkedRowVJP.apply(
        visible, reference_rows, SPARSE_VJP_ROW_CHUNK, q, kv, sink
    )

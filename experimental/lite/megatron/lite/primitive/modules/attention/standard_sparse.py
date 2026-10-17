# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core sparse kernels for the non-BI DS41 standard CUDA path."""

import torch

from .packed_geometry import PackedCPGeometry
from .packed_indices import packed_sparse_metadata


def standard_sparse_attention(
    query, window, compressed, sink, selected, *, ratio, window_size
):
    """Use the A2 physical addresses with Core's FlashMLA/DSA autograd.

    This entry covers CP1 full documents. Packed CP wiring and its fixed halo
    remain owned by the shared packed attention integration.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils.packed_sparse_attention import (
        csa_sparse_attn,
    )

    batch, rows, heads, dim = query.shape
    if rows != window.shape[1] or rows == 0:
        raise ValueError('Standard sparse attention expects a non-empty CP1 document')
    if query.dtype != torch.bfloat16 or window.dtype != query.dtype or dim != 512:
        raise ValueError(
            'Core sparse attention requires BF16 Q/KV with head dimension 512'
        )
    if compressed is None:
        compressed = window[:, :0]
    if compressed.dtype != query.dtype:
        raise ValueError('Compressed keys must use the query dtype')
    geometry = PackedCPGeometry((0, rows), device=query.device)
    kv = torch.cat((compressed, window), 1)
    outputs = []
    for item in range(batch):
        indices, lengths = packed_sparse_metadata(
            geometry,
            ratio=ratio,
            window_size=window_size,
            selected=None if selected is None else selected[item],
            kv_bounds=(0, compressed.shape[1]) if ratio else None,
        )
        output = csa_sparse_attn(
            query[item].contiguous(),
            kv[item].contiguous(),
            sink.float(),
            indices,
            dim**-0.5,
            topk_length=lengths,
        )
        outputs.append(output.reshape(rows, heads, dim))
    return torch.stack(outputs)


@torch.no_grad()
def standard_indexer_scores(query, keys, weights, *, ratio):
    """Reuse Core's noncompact DSA score kernel without a head-score tensor.

    Keep final per-key scores for DS41's existing candidate build/reuse policy;
    Core's compact TopK does not accept that policy's external candidate mask.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        fused_sparse_attention as core,
    )

    batch, rows = query.shape[:2]
    columns = keys.shape[1]
    if not columns:
        return query.new_empty((batch, rows, 0), dtype=torch.float32)
    if any(t.dtype != torch.bfloat16 for t in (query, keys, weights)):
        raise ValueError('DSA indexer requires BF16 query, keys and scaled weights')
    core._ensure_dsa_namespace()
    cu_q = torch.tensor((0, rows), dtype=torch.int32, device=query.device)
    cu_k = torch.tensor((0, columns), dtype=torch.int32, device=query.device)
    return torch.stack(
        [
            core._DSA.indexer_forward_wrapper(
                query[item].contiguous(),
                keys[item].unsqueeze(1).contiguous(),
                weights[item].contiguous(),
                ratio=ratio,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=rows,
                max_seqlen_k=columns,
            )['scores']
            for item in range(batch)
        ]
    )

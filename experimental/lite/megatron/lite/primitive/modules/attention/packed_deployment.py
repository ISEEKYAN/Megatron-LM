# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment-math cross-layer CSA over a whole packed CP slice.

DS4's packed CP attention, with DS4.1's codecs and compression ratios 1/2:
window keys are the rank's own rows plus a fixed left halo exchanged with the
previous rank; compressed groups are made by the rank holding their last token
and gathered into sequence-major rows by Core's row map; Core lowers the
document-relative selections to physical addresses. The packed state keeps the
AttentionState THD layout (compressed rows at ``kv_bounds``, query rows at
``query_bounds``), so later consumer layers read it unchanged.

Every collective runs on every rank, including ranks with no real rows. A
rank whose width is below the halo cannot get it from its neighbour alone;
those slices run the per-document path instead (``fits``/``note_fallback``).
"""

import logging
from collections import Counter

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# Packed slices that ran the per-document path because width < halo.
FALLBACKS = Counter()


def halo_rows(attn):
    return max(attn.config.window, max(attn.ratio - 1, 0))


def fits(attn, geometry):
    """True when the single-hop halo covers this slice on every rank."""
    return geometry.cp_size == 1 or geometry.width >= halo_rows(attn)


def note_fallback(attn, geometry):
    key = (geometry.total, geometry.cp_size, geometry.width, halo_rows(attn))
    FALLBACKS[key] += 1
    if FALLBACKS[key] == 1:
        logger.warning(
            'Packed CSA uses the per-document path: T=%d, cp=%d, width=%d < halo=%d',
            *key,
        )


def _gather_rows(local, row_map, geometry):
    """Rank-major gather and sequence-major selection of compressed rows."""
    from dataclasses import fields, is_dataclass, replace

    from .packed_transport import gather_packed_compressed_rows

    if is_dataclass(local):
        return replace(
            local,
            **{
                f.name: _gather_rows(getattr(local, f.name), row_map, geometry)
                for f in fields(local)
            },
        )
    _, rows = gather_packed_compressed_rows(local[0], row_map, geometry)
    return rows.unsqueeze(0)


def _pad_cat(parts, pad):
    width = max(p.shape[-1] for p in parts)
    return torch.cat([F.pad(p, (0, width - p.shape[-1]), value=pad) for p in parts], 1)


def _window(attn, unrotated_q, local_kv, halo_kv, positions, halo_positions, cos_sin):
    """Native Q/SWA writer over the local rows and the halo rows (row-wise)."""
    from ..deployment_math.autograd import visible_forward
    from ..deployment_math.cache import _swa_insert
    from .csa import rotate

    c, ratio = attn.config, attn.ratio

    def reference(query, values, halo):
        return (
            rotate(query, positions, c, ratio, deployment_math=True),
            rotate(values, positions, c, ratio, deployment_math=True),
            rotate(halo, halo_positions, c, ratio, deployment_math=True),
        )

    def visible(query, values, halo):
        if query.shape[1]:
            rotated, window = _swa_insert(query, values, positions, cos_sin, c.eps, 64)
        else:
            # A rank may own no real rows; the joint native entry has no
            # zero-token early exit.
            rotated = query.new_empty(query.shape)
            window = reference(query, values, halo)[1]
        if halo.shape[1]:
            placeholder = query.new_zeros((1, halo.shape[1], *query.shape[2:]))
            _, halo_window = _swa_insert(
                placeholder, halo, halo_positions, cos_sin, c.eps, 64
            )
        else:
            halo_window = halo
        return rotated, window, halo_window

    return visible_forward(visible, reference, unrotated_q, local_kv, halo_kv)


def _compress(attn, x, boundary, geometry, cos_sin):
    """Compressed main KV and index keys made here, gathered sequence-major."""
    from ..deployment_math.cache import compressed_cache, index_cache
    from .csa import AttentionState, _indexer_linear, rotate

    c, ratio = attn.config, attn.ratio
    rows = x.shape[1]
    hidden = F.pad(x[0], (0, 0, 0, geometry.width - rows))  # Core needs width rows
    capacity = geometry.width // ratio + 1
    gathered, _, starts, cu_compressed, row_map = geometry.compressor_input(
        hidden,
        boundary[-(ratio - 1) :] if ratio > 1 else boundary[:0],
        ratio,
        halo=ratio - 1,
        capacity=capacity,
    )
    latent = attn.compressor(gathered.unsqueeze(0))
    writer = (starts.long() + ratio - 1).to(latent.device)  # group's last token
    index_k = index_cache(
        _indexer_linear(attn.indexer.wk, latent, True),
        attn.indexer.k_norm.weight,
        attn.indexer.k_norm.eps,
        ratio,
        cos_sin,
        positions=writer,
    )

    def main_vjp(values):
        return rotate(values, starts.long(), c, ratio, deployment_math=True)

    main = compressed_cache(latent, ratio, cos_sin, main_vjp, positions=writer)
    kv_bounds = tuple(int(v) for v in cu_compressed.tolist())
    row_map = row_map[: kv_bounds[-1]]
    return AttentionState(
        kv_owner=attn.layer_id,
        main_kv=_gather_rows(main, row_map, geometry),
        index_k=_gather_rows(index_k, row_map, geometry),
        kv_bounds=kv_bounds,
    )


def _select(attn, x, qr, state, geometry, positions, cos_sin, candidate_mask):
    """Index owner: score each document's keys and select per query row."""
    from dataclasses import replace

    from .. import deployment_math
    from ..deployment_math.cache import index_query
    from .csa import _indexer_linear, _rows, candidate_blocks

    c, ratio = attn.config, attn.ratio
    iq = attn.indexer.wq_b(qr).unflatten(-1, (c.index_heads, c.index_dim))
    iq, weights = index_query(
        iq, _indexer_linear(attn.indexer.weights_proj, x, True), positions, cos_sin
    )
    lengths = ((positions + 1) // ratio).unsqueeze(-1)  # as the per-document path
    kv, query = state.kv_bounds, []
    for doc in range(geometry.num_documents):
        start, stop = geometry.owned(doc)
        query.append(query[-1] + stop - start if query else stop - start)
    query = (0, *query)
    selections, pools = [], []
    for doc in range(geometry.num_documents):
        q0, q1, k0, k1 = query[doc], query[doc + 1], kv[doc], kv[doc + 1]
        scores = deployment_math.indexer_scores(
            _rows(iq, q0, q1),
            _rows(state.index_k, k0, k1),
            weights[:, q0:q1],
            lengths[q0:q1],
        )
        pool = None
        if state.candidates is not None:
            pool = state.candidates[:, q0:q1, : k1 - k0]
        if attn.candidate_mode == 'build':
            pool = candidate_blocks(
                scores,
                lengths[q0:q1],
                topk_blocks=c.candidate_blocks,
                block_size=c.block_size,
            )
        injected = pool if candidate_mask is None else candidate_mask[:, q0:q1]
        if attn.candidate_mode == 'reuse' and injected is None:
            raise ValueError('Reindex requires the layer-20 candidate pool')
        selections.append(
            deployment_math.indexer_select_positions(
                scores,
                lengths[q0:q1],
                c.topk,
                candidates=injected if attn.candidate_mode == 'reuse' else None,
            )
        )
        pools.append(pool)
    candidates = None
    if all(p is not None for p in pools):
        candidates = _pad_cat(pools, False)
    return replace(
        state,
        index_owner=attn.layer_id,
        indices=_pad_cat(selections, -1),
        candidates=candidates,
        query_bounds=query,
    )


def forward(attn, x, state, geometry, *, candidate_mask=None):
    """Packed counterpart of the deployment branch of ``_forward_cross_layer``."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
        thd_cp_position_ids,
    )

    from ..deployment_math.cache import rotary_cache
    from ..deployment_math.sparse import sparse_attention_from_indices
    from .packed_indices import packed_sparse_metadata
    from .packed_transport import packed_boundary_hidden

    c, ratio = attn.config, attn.ratio
    if x.shape[0] != 1:
        raise ValueError('Packed CSA expects a single packed row batch')
    if not (c.swa_fp8 and c.main_qat and c.index_qat):
        raise ValueError('Deployment cache requires enabled KV codecs')
    rows = x.shape[1]
    positions = geometry.positions[:rows]
    halo = halo_rows(attn)
    halo_positions = thd_cp_position_ids(
        geometry.cu_seqlens_padded, geometry.global_start - halo, halo
    ).long()
    cos_sin = rotary_cache(c, ratio, x.device)

    qr = attn.q_norm(attn.wq_a(x))
    unrotated_q = attn.wq_b(qr).unflatten(-1, (c.heads, c.head_dim))
    boundary = packed_boundary_hidden(x[0], geometry, window_size=c.window, ratio=ratio)
    local_kv = attn.kv_norm(attn.wkv(x))
    halo_kv = attn.kv_norm(attn.wkv(boundary.unsqueeze(0)))
    q, window, halo_window = _window(
        attn, unrotated_q, local_kv, halo_kv, positions, halo_positions, cos_sin
    )

    selected, main, kv_bounds = None, None, None
    if ratio:
        if attn.owns_kv:
            state = _compress(attn, x, boundary, geometry, cos_sin)
        if (
            state.kv_owner != attn.kv_owner
            or state.main_kv is None
            or state.index_k is None
            or state.kv_bounds is None
        ):
            raise ValueError('Missing or incorrect CSA2 KV source state')
        if attn.owns_index:
            state = _select(
                attn, x, qr, state, geometry, positions, cos_sin, candidate_mask
            )
        elif state.index_owner != attn.index_owner or state.indices is None:
            raise ValueError('Missing or incorrect CSA2 index source state')
        selected, main, kv_bounds = state.indices[0], state.main_kv[0], state.kv_bounds

    width = (
        ((c.topk + c.window + 127) // 128) * 128
        if ratio
        else ((c.window + 127) // 128) * 128
    )
    indices, lengths = packed_sparse_metadata(
        geometry,
        selected,
        ratio=ratio,
        window_size=c.window,
        lookback_rows=halo,  # Core's fixed right-aligned halo, rank 0 included
        width=width,
        kv_bounds=kv_bounds,
        query_bounds=state.query_bounds if ratio else None,
    )
    parts = [halo_window[0], window[0]]
    if main is not None:
        parts.insert(0, main)
    kv = torch.cat(parts, 0).unsqueeze(0)
    output = sparse_attention_from_indices(
        q,
        kv,
        attn.attn_sink,
        indices.unsqueeze(0),
        lengths.unsqueeze(0),
        scale=c.head_dim**-0.5,
    )
    return attn._project_output(output, positions, x), state

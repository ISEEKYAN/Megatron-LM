# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 forward in vLLM BI=1 prefill order, over plain parameter namespaces.

One packed THD batch, TP1, PP1. Order per layer (``DeepseekV41DecoderLayer``):

* entry: layer 0 broadcasts the embedding into the streams and runs
  ``mhc_pre_delayed_overlap`` with the summed ``hc_attn_fn``; an Engram layer
  runs ``mhc_post_tilelang`` -> Engram -> ``mhc_pre_delayed_overlap``; any other
  layer runs the fused ``mhc_shifted_post_pre`` on the previous FFN output.
* attention: :mod:`.attention` projections/compressor/indexer, the attention
  core (callable), ``wo_a``/``wo_b``.
* FFN entry: ``mhc_shifted_post_pre`` on the attention output; MoE.
* end: ``mhc_post_tilelang`` -> ``hc_collapse`` (pre of the last FFN mix) ->
  RMSNorm -> head.

``tap(name, value)`` sees every module boundary under the dump's key names.
"""

from types import SimpleNamespace

import torch

from megatron.lite.primitive.parallel.linear import _EmbeddingAccumulatingIntoMainGrad

from . import attention as A
from .engram import engram
from .mhc import collapse, joint, post_mix
from .norm import rms_norm

IMAGE_SENTINEL = 129264  # Engram gate is shut on image-span tokens (vLLM HAS_MASK path)


def _none(*_):
    return None


def attention_layer(
    x, positions, p, layer, cfg, state, core, tap=_none, cu_seqlens=None, layout="mega", cp=None
):
    """``p``: per-layer attention namespace (see :func:`attention_params`).

    ``layout``: "mega" (fused wq_b rows / wo_a columns of the mega kernel) or
    "flashmla" (release order). ``cp``: :class:`.attention.ContextParallel` or None;
    ``cu_seqlens`` are then global and the compressor latent spans all ranks."""
    fused = layout == "mega"
    eps = cfg["rms_norm_eps"]
    ratio = cfg["compress_ratios"][layer]
    q, kv, (q_dec, shared) = A.projections(x, p, eps, fused=fused)
    tap("attn.wq_b.out", q)
    tap("attn.kv", kv)
    latent, topk = None, None
    if ratio:
        source = max(s for s in cfg["kv_source_layer_ids"] if s <= layer)
        if source == layer:
            state.latent[layer] = A.compressor(
                x, positions, p.c_wkv, p.c_wgate, p.c_norm, eps, ratio, cu_seqlens, cp
            )
        latent = state.latent[source]
        if cp is not None:
            latent = latent[cp.offset : cp.offset + x.shape[0]]
        if source == layer:
            tap("compressor.out", latent)
        if layer in cfg["index_source_layer_ids"]:
            candidate_source = cfg.get("candidate_source_layer_id", -1)
            candidate = (
                "write"
                if layer == candidate_source
                else ("read" if 0 <= candidate_source < layer and ratio == 1 else None)
            )
            A.indexer_topk(
                x,
                positions,
                (q_dec, shared),
                state.latent[source],
                p.indexer,
                ratio=ratio,
                cos_sin_cache=state.rope(ratio),
                eps=eps,
                state=state.indexer,
                source=source,
                owns_k=source == layer,
                candidate=candidate,
                cu_seqlens=cu_seqlens,
                cp=cp,
            )
            tap("indexer.q", state.indexer.q)
        topk = state.indexer.topk
        tap("attn.topk", topk)
    out = core(
        layer=layer,
        q=q,
        kv=kv,
        latent=latent,
        topk=topk,
        positions=positions,
        attn_sink=p.attn_sink,
        ratio=ratio,
        cos_sin_cache=state.rope(ratio),
        cu_seqlens=cu_seqlens,
        cp_group=None if cp is None else cp.group,
    )
    tap("attn.core.out", out)
    o = A.o_projection(out, p.wo_a, p.wo_b, fused=fused)
    tap("attn.out", o)
    return o


def layer_forward(
    layer,
    lp,
    carry,
    ids,
    positions,
    cfg,
    state,
    *,
    core,
    moe,
    engram_rows,
    tap=_none,
    cu_seqlens=None,
    layout="mega",
    keep=None,
    cp=None,
):
    """One decoder layer. ``carry`` is the embedding for layer 0, else the previous
    layer's ``(y, stream, post, comb, pre)``; returns this layer's carry."""
    copies = cfg["hc_mult"]
    if layer == 0:
        if lp.engram is not None:
            raise NotImplementedError("Engram on layer 0")
        stream = carry.unsqueeze(-2).expand(-1, copies, -1).contiguous()
        dummy = stream.new_empty(stream.shape[0], copies, dtype=torch.float32)
        _, pre, post, comb, x = joint(stream, dummy, lp.attn_norm, lp.attn_mixes, broadcast=True)
    elif lp.engram is not None:
        y, stream, post, comb, pre = carry
        stream = post_mix(y, stream, post, comb)
        tap("mhc_post_tilelang.out", stream)
        stream = engram(
            stream,
            engram_rows(layer),
            lp.engram.wkv,
            lp.engram.q_weight,
            lp.engram.k_weight,
            keep,
            eps=cfg["rms_norm_eps"],
        )
        tap("engram.out", stream)
        _, pre, post, comb, x = joint(stream, pre, lp.attn_norm, lp.attn_mixes)
    else:
        y, stream, post, comb, pre = carry
        stream, pre, post, comb, x = joint(
            stream, pre, lp.attn_norm, lp.attn_mixes, pending=(y, stream, post, comb)
        )
    tap("attn.in", x)
    o = attention_layer(x, positions, lp.attn, layer, cfg, state, core, tap, cu_seqlens, layout, cp)
    stream, pre, post, comb, x = joint(
        stream, pre, lp.ffn_norm, lp.ffn_mixes, pending=(o, stream, post, comb)
    )
    tap("moe.in", x)
    y = moe(x, ids, layer)
    tap("moe.out", y)
    carry = (y, stream, post, comb, pre)
    tap("layer.out", carry)
    return carry


def model_forward(
    ids,
    positions,
    params,
    cfg,
    state,
    *,
    core,
    moe,
    engram_rows,
    tap=_none,
    cu_seqlens=None,
    layout="mega",
    keep=None,
    cp=None,
):
    """Returns the final normed hidden ``[T, H]`` (input of the head).

    ``params``: ``embed``, ``norm``, ``layers[i]`` with ``attn_norm``, ``ffn_norm``,
    ``attn_mixes``, ``ffn_mixes``, ``attn``, ``engram`` (or None).
    ``moe(x, ids, layer)``, ``engram_rows(layer)`` and ``core(**kw)`` are callables.
    """
    # Lookup whose backward accumulates into the FP32 main_grad (as the native embedding
    # does) instead of a per-rank BF16 grad: under CP each rank's partial sum would be
    # rounded to BF16 before the cross-rank reduction.
    embedding = _EmbeddingAccumulatingIntoMainGrad.apply(params.embed, ids)
    keep = (ids != IMAGE_SENTINEL) if keep is None else keep & (ids != IMAGE_SENTINEL)
    tap("embed.out", embedding)
    carry = embedding
    for layer, lp in enumerate(params.layers):

        def ltap(name, value, prefix=f"L{layer}."):
            tap(prefix + name, value)

        carry = layer_forward(
            layer,
            lp,
            carry,
            ids,
            positions,
            cfg,
            state,
            core=core,
            moe=moe,
            engram_rows=engram_rows,
            tap=ltap,
            cu_seqlens=cu_seqlens,
            layout=layout,
            keep=keep,
            cp=cp,
        )
    y, stream, post, comb, pre = carry
    stream = post_mix(y, stream, post, comb)
    collapsed = collapse(stream, pre)
    tap("hc_collapse.out", collapsed)
    hidden = rms_norm(collapsed, params.norm, cfg["rms_norm_eps"])
    tap("model.out", hidden)
    return hidden


def interface_core(kind):
    """The tracks.md attention core of ``kind`` ("mega": M's ``mega_attention``;
    "flashmla": F's ``flashmla_attention_core``) as an :func:`attention_layer` core.

    ``comp_latent`` is passed as the boundary rows (``(pos+1) % cr == 0``) in token order.
    """
    if kind == "mega":
        from .mega_attention import mega_attention as fn

        def call(q, kv, comp, topk, positions, sink, ratio, cache, cu_seqlens, cp_group):
            return fn(
                q,
                kv,
                comp,
                topk,
                positions,
                sink,
                compress_ratio=ratio,
                cos_sin_cache=cache,
                cu_seqlens=cu_seqlens,
                cp_group=cp_group,
            )

    elif kind == "flashmla":
        from .flashmla_attention import FlashMLALayerConfig, flashmla_attention_core as fn

        def call(q, kv, comp, topk, positions, sink, ratio, cache, cu_seqlens, cp_group):
            return fn(
                q,
                kv,
                comp,
                topk,
                positions,
                sink,
                FlashMLALayerConfig(compress_ratio=ratio),
                cache,
                cu_seqlens=cu_seqlens,
                cp_group=cp_group,
            )

    else:
        raise ValueError(f"Unknown DS4.1 attention core {kind!r}")

    def core(
        *,
        layer,
        q,
        kv,
        latent,
        topk,
        positions,
        attn_sink,
        ratio,
        cos_sin_cache,
        cu_seqlens,
        cp_group=None,
    ):
        comp = None if latent is None else latent[(positions + 1) % ratio == 0]
        return call(
            q, kv, comp, topk, positions, attn_sink, ratio, cos_sin_cache, cu_seqlens, cp_group
        )

    return core


def attention_params(get, layer):
    """Per-layer attention namespace from ``get(release_key) -> tensor | None``."""
    p = f"layers.{layer}.attn."
    return SimpleNamespace(
        wq_a=get(p + "wq_a.weight"),
        wkv=get(p + "wkv.weight"),
        q_norm=get(p + "q_norm.weight"),
        kv_norm=get(p + "kv_norm.weight"),
        wq_b=get(p + "wq_b.weight"),
        wo_a=get(p + "wo_a.weight"),
        wo_b=get(p + "wo_b.weight"),
        attn_sink=get(p + "attn_sink"),
        c_wkv=get(p + "compressor.wkv.weight"),
        c_wgate=get(p + "compressor.wgate.weight"),
        c_norm=get(p + "compressor.norm.weight"),
        indexer=SimpleNamespace(
            wq_b=get(p + "indexer.wq_b.weight"),
            weights_proj=get(p + "indexer.weights_proj.weight"),
            wk=get(p + "indexer.wk.weight"),
            k_norm=get(p + "indexer.k_norm.weight"),
        ),
    )


class ForwardState(SimpleNamespace):
    """Cross-layer state of one forward: compressor latents, indexer K/candidates/top-k."""

    def __init__(self, rope):
        super().__init__(latent={}, indexer=A.IndexerState(), rope=rope)

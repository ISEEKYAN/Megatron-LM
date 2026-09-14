"""Single-process virtual CP owners, built independently of tested CP tensors.

Only configuration, complete initial state and original batches enter this model.
Physical ownership is calculated here; no product CP context/collective is called.
"""

import torch
from megatron.lite.model.qwen3_8_flash_next.math import qsa_routes, sparse_attention
from megatron.lite.primitive.utils.rope import _apply_rotary_pos_emb_bshd as rope
from torch.nn import functional as F


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right):
        ctx.width = left.shape[1]
        full = torch.cat((left, right), 1)
        return full.clone(), full.clone()

    @staticmethod
    def backward(ctx, left, right):
        total = left + right
        return total[:, : ctx.width], total[:, ctx.width :]


def gdn(modules, inputs, cu):
    projections = [
        m.in_proj(x.transpose(0, 1)).transpose(0, 1).contiguous()
        for m, x in zip(modules, inputs)
    ]
    sections = modules[0]._qkvzba_sections()
    # Partition each semantic projection section before exchanging token owners.
    local = [torch.split(x, sections, -1) for x in projections]
    outputs = []
    for rank, m in enumerate(modules):
        projected = torch.cat(
            [torch.cat([part.chunk(2, -1)[rank] for part in row], -1) for row in local],
            1,
        ).contiguous()
        q, k, v, gate, beta, alpha = m._split_proj(projected, 2)
        b, s = projected.shape[:2]
        qkv = torch.cat([x.reshape(b, s, -1) for x in (q, k, v)], -1)
        weights = torch.cat(
            [x.chunk(2, 0)[rank] for x in m.conv1d.weight.split(m._conv_sections(), 0)],
            0,
        )
        qkv = m._causal_conv1d(
            qkv, s, cu_seqlens=cu, conv_weight=weights, cp_div=2, cp_context=None
        )
        q, k, v, gate, beta, alpha = m._prepare_qkv(qkv, gate, beta, alpha, b, s, 2)
        g, beta = m._compute_g_and_beta(
            m.A_log.chunk(2)[rank], m.dt_bias.chunk(2)[rank], alpha, beta
        )
        out, _ = m._gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu,
            cp_context=None,
        )
        outputs.append(m._apply_gated_norm(out, gate).reshape(b, s, -1))
    return [
        m.o_proj(
            torch.cat([x[:, r * 8 : (r + 1) * 8] for x in outputs], -1)
            .transpose(0, 1)
            .contiguous()
        ).transpose(0, 1)
        for r, m in enumerate(modules)
    ]


def ple(modules, inputs, ids, real_mask, cu):
    values, normalized = [], []
    for r, (m, x) in enumerate(zip(modules, inputs)):
        hashes = m.ple_embedding.hash_ids(
            ids.masked_fill(~real_mask, m.ple_embedding.eos_token_id), cu
        )
        embeddings = (
            m.ple_embedding.ngram_embedding(hashes[:, r * 8 : (r + 1) * 8])
            .flatten(-2)
            .to(m.key_proj.weight.dtype)
        )
        key = m.key_proj(embeddings).unflatten(-1, (m.hc_count, m.hidden_size))
        query = x.unflatten(-1, (m.hc_count, m.hidden_size))
        dot = (m._norm(key, m.norm_key) * m._norm(query, m.norm_query)).sum(
            -1, keepdim=True
        ) / m.hidden_size**0.5
        gate = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt())
        value = gate * m.value_proj(embeddings).unsqueeze(-2)
        values.append(value)
        normalized.append(
            m._norm(value, m.norm_conv)
            .flatten(-2)
            .to(m.conv1d.weight.dtype)
            .masked_fill(~real_mask[:, r * 8 : (r + 1) * 8, None], 0)
        )
    gathered = Gather.apply(*normalized)
    outputs = []
    for r, m in enumerate(modules):
        history = 3 * (m.conv1d.kernel_size[0] - 1)
        left = gathered[r][:, max(0, r * 8 - history) : r * 8]
        halo = (
            F.pad(left, (0, 0, history - left.shape[1], 0)) + gathered[r][:, :0].sum()
        )
        extended = torch.cat((halo, normalized[r]), 1)
        positions = torch.arange(r * 8, (r + 1) * 8, device=ids.device)
        starts = cu[torch.bucketize(positions, cu[1:], right=True)]
        windows = [
            extended[:, 3 * i : 3 * i + 8]
            * (positions - history + 3 * i >= starts)[None, :, None]
            for i in range(m.conv1d.kernel_size[0])
        ]
        windows = torch.stack(windows, -1).reshape(
            -1, inputs[r].shape[-1], m.conv1d.kernel_size[0]
        )
        convolution = F.silu(
            F.conv1d(windows, m.conv1d.weight, groups=inputs[r].shape[-1])
        ).reshape_as(inputs[r])
        outputs.append((values[r].flatten(-2) + convolution).to(inputs[r].dtype))
    return outputs


def qsa(modules, inputs, angles, cu):
    qs, ks, vs, gates, indexes = [], [], [], [], []
    c = modules[0].config
    for r, (m, x) in enumerate(zip(modules, inputs)):
        q, gate = (
            m.q_proj(x)
            .reshape(1, 8, c.num_attention_heads, 2 * c.head_dim)
            .chunk(2, -1)
        )
        k = m.k_proj(x).reshape(1, 8, c.num_key_value_heads, c.head_dim)
        qs.append(rope(m._norm(q, m.q_norm), angles[:, r * 8 : (r + 1) * 8]))
        ks.append(rope(m._norm(k, m.k_norm), angles[:, r * 8 : (r + 1) * 8]))
        vs.append(m.v_proj(x).reshape_as(k))
        gates.append(gate)
        with torch.no_grad():
            indexes.append(
                m.indexer.index_qk_proj(x).reshape(
                    1, 8, c.indexer_n_heads + 1, c.indexer_head_dim
                )
            )
    q, k, v = [torch.cat(parts, 1) for parts in (qs, ks, vs)]
    index = torch.cat(indexes, 1)
    outputs = []
    m = modules[0]
    for a, z in zip(cu.tolist(), cu.tolist()[1:]):
        with torch.no_grad():
            iq = rope(
                m._norm(index[:, a:z, :-1], m.indexer.q_layernorm), angles[:, a:z]
            )
            blocks = (z - a) // c.indexer_compress_ratio
            raw = index[:, a : a + blocks * c.indexer_compress_ratio, -1:]
            pooled = (
                raw.reshape(1, blocks, c.indexer_compress_ratio, 1, c.indexer_head_dim)
                .float()
                .mean(2)
                .to(index.dtype)
            )
            ik = rope(
                m._norm(pooled, m.indexer.k_layernorm),
                angles[
                    :,
                    a : a
                    + blocks * c.indexer_compress_ratio : c.indexer_compress_ratio,
                ],
            )
            routes = qsa_routes(
                iq,
                ik,
                torch.tensor([z - a], device=q.device),
                token_budget=c.indexer_budget,
                compress_ratio=c.indexer_compress_ratio,
            )
        # Product QSA deliberately retains complete-query native dk/dv order.
        outputs.append(
            sparse_attention(
                q[:, a:z],
                k[:, a:z],
                v[:, a:z],
                routes[..., : min(z - a, routes.shape[-1])],
            )
        )
    output = torch.cat(outputs, 1)
    return [
        m.o_proj((output[:, r * 8 : (r + 1) * 8] * gates[r].sigmoid()).flatten(-2))
        for r, m in enumerate(modules)
    ]


def moe(modules, inputs, mask):
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
    from megatron.lite.primitive.modules.router import _ordered_topk_from_routing_map
    from megatron.lite.primitive.utils.moe import (
        compute_routing_scores_for_aux_loss,
        router_gating_linear,
        switch_load_balancing_loss_func,
        topk_routing_with_score_function,
    )

    routing = []
    for m, x in zip(modules, inputs):
        r = m.router
        logits = router_gating_linear(
            x.reshape(-1, x.shape[-1]), r.gate.weight, None, r.router_dtype
        )
        probs, mapping = topk_routing_with_score_function(
            logits,
            r.topk,
            use_pre_softmax=r.use_pre_softmax,
            score_function='softmax',
            fused=False,
        )
        scores, indices = _ordered_topk_from_routing_map(probs, mapping, r.topk)
        aux_map, aux_scores = compute_routing_scores_for_aux_loss(
            logits, r.topk, score_function='softmax', fused=False
        )
        routing.append((scores, indices, aux_map, aux_scores))
    counts = [
        (item[2] & mask[:, r * 8 : (r + 1) * 8].reshape(-1, 1)).sum(0).long()
        for r, item in enumerate(routing)
    ]
    total = counts[0] + counts[1]
    outputs = []
    for rank, (m, x, item) in enumerate(zip(modules, inputs, routing)):
        scores, indices, _, aux = item
        r = m.router
        aux = aux * mask[:, rank * 8 : (rank + 1) * 8].reshape(-1, 1)
        loss = (
            switch_load_balancing_loss_func(
                aux, total, 13, r.topk, r.num_experts, r.aux_loss_coeff, fused=False
            )
            * 2
        )
        scores = MoEAuxLossAutoScaler.apply(scores, loss)
        original = r.forward
        try:
            r.forward = lambda x, scores=scores, indices=indices: (scores, indices)
            outputs.append(m(x))
        finally:
            r.forward = original
    return outputs


def forward(models, batch):
    """Fixed CP2 proxy: 13 real rows, aligned to 16, documents [0,5,13]."""
    assert batch.seq_lens.tolist() == [5, 8], 'CP_REFERENCE_DOCUMENTS'
    ids = F.pad(batch.input_ids.reshape(1, -1), (0, 3))
    mask = torch.arange(16, device=ids.device)[None, :] < 13
    cu = torch.tensor([0, 5, 13, 16], device=ids.device, dtype=torch.int32)
    positions = torch.tensor(
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0], device=ids.device
    )
    c = models[0].config
    rotary = int(c.head_dim * c.partial_rotary_factor)
    inv = c.rope_theta ** (
        -torch.arange(0, rotary, 2, device=ids.device).float() / rotary
    )
    half = positions[None, :, None].float() * inv
    angles = torch.cat((half, half), -1).unsqueeze(-2)
    hidden = [
        m.embed_tokens(ids[:, r * 8 : (r + 1) * 8]).repeat(1, 1, c.hc_count)
        for r, m in enumerate(models)
    ]
    for i in range(c.num_hidden_layers):
        layers = [m.layers[i] for m in models]
        if layers[0].ple is not None:
            extra = ple([b.ple for b in layers], hidden, ids, mask, cu)
            hidden = [x + y for x, y in zip(hidden, extra)]
        mixed = [b.attn_hyper_connection.mix(x) for b, x in zip(layers, hidden)]
        branches = [x[0] for x in mixed]
        fn = gdn if layers[0].linear_attn is not None else qsa
        arguments = (cu,) if fn is gdn else (angles, cu)
        branch = fn(
            [b.linear_attn if fn is gdn else b.self_attn for b in layers],
            branches,
            *arguments
        )
        hidden = [
            b.attn_hyper_connection.combine(y, pair[1])
            for b, y, pair in zip(layers, branch, mixed)
        ]
        mixed = [b.mlp_hyper_connection.mix(x) for b, x in zip(layers, hidden)]
        branch = moe([b.mlp for b in layers], [x[0] for x in mixed], mask)
        hidden = [
            b.mlp_hyper_connection.combine(y, pair[1])
            for b, y, pair in zip(layers, branch, mixed)
        ]
    logits = [
        m.lm_head(m.hyper_connection_mixer.mix(x)[0]) for m, x in zip(models, hidden)
    ]
    labels = torch.full((16,), -100, device=ids.device, dtype=torch.long)
    for a, z in ((0, 5), (5, 13)):
        labels[a : z - 1] = batch.labels[a + 1 : z].masked_fill(
            ~batch.loss_mask[a + 1 : z].bool(), -100
        )
    denominator = (labels != -100).sum()
    losses = [
        F.cross_entropy(
            x.float().flatten(0, 1), labels[r * 8 : (r + 1) * 8], reduction='none'
        ).sum()
        / denominator
        * 2
        for r, x in enumerate(logits)
    ]
    return logits, losses

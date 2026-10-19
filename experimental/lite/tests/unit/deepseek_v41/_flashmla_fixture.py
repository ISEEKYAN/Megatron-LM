# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inputs, serving harness and FP64 reference for the FlashMLA attention core tests.

The serving harness runs vLLM's own ``DeepseekV4FlashMLAAttention.forward_mqa``
under real SWA / compressed metadata builders on paged caches with shuffled
block tables, written by the ops the serving layer calls. It is the forward
the training core must equal bit for bit.
"""

from dataclasses import replace
from types import SimpleNamespace

import torch

HEADS, DIM, ROPE, WINDOW, TOP_K = 64, 512, 64, 128, 512
SWA_PAGE, COMPRESSED_PAGE_TOKENS = 64, 128


def v41_rope(compress_ratio, device, max_position=1048576):
    """The layer's ``rotary_emb.cos_sin_cache`` for DSv4.1-Flash's rope config."""
    from vllm.config import set_current_vllm_config
    from vllm.config.vllm import VllmConfig
    from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope

    config = SimpleNamespace(
        rope_parameters={
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        },
        rope_theta=10000,
        compress_rope_theta=160000,
    )
    with set_current_vllm_config(VllmConfig()), torch.device(device):
        rope = build_deepseek_v4_rope(
            config,
            head_dim=DIM,
            rope_head_dim=ROPE,
            max_position_embeddings=max_position,
            compress_ratio=compress_ratio,
        )
    return rope.cos_sin_cache.to(device=device, dtype=torch.float32).contiguous()


def make_inputs(lengths, compress_ratio, seed, device):
    """Random layer inputs for a packed sequence of documents ``lengths``."""
    g = torch.Generator(device=device).manual_seed(seed)
    total = sum(lengths)
    q = torch.randn(total, HEADS, DIM, device=device, generator=g).bfloat16()
    kv = torch.randn(total, DIM, device=device, generator=g).bfloat16()
    positions = torch.cat([torch.arange(n, device=device) for n in lengths])
    sink = torch.randn(HEADS, device=device, generator=g)
    latent = topk = None
    if compress_ratio:
        entries = sum(n // compress_ratio for n in lengths)
        latent = torch.randn(entries, DIM, device=device, generator=g).bfloat16()
        topk = torch.full((total, TOP_K), -1, dtype=torch.int32, device=device)
        cpu = torch.Generator().manual_seed(seed)
        for row, pos in enumerate(positions.tolist()):
            candidates = (pos + 1) // compress_ratio
            if candidates <= TOP_K:
                chosen = torch.arange(candidates)
            else:
                chosen = torch.randperm(candidates, generator=cpu)[:TOP_K].sort().values
            topk[row, : chosen.numel()] = chosen.to(device=device, dtype=torch.int32)
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return SimpleNamespace(
        q=q, kv=kv, latent=latent, topk=topk, positions=positions, sink=sink, cu=cu, lengths=lengths
    )


class Serving:
    """vLLM's FLASHMLA_SPARSE_DSV41 layer forward on one layer's paged caches."""

    def __init__(self, inputs, compress_ratio, cos_sin, device, seed=5):
        from vllm.config import CacheConfig, set_current_vllm_config
        from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert
        from vllm.models.deepseek_v41.nvidia.flashmla import (
            DeepseekSparseSWAFlashMLAMetadataBuilder,
            DeepseekV4FlashMLAAttention,
        )
        from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadataBuilder
        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
        from vllm.v1.kv_cache_interface import MLAAttentionSpec
        from vllm.v1.worker.workspace import init_workspace_manager

        self.inputs, self.cr, self.cos_sin, self.device = inputs, compress_ratio, cos_sin, device
        lengths = inputs.lengths
        requests, max_len = len(lengths), max(lengths)
        self.max_tokens = sum(lengths)
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    sliding_window=WINDOW, compress_ratios=[0, 0, 2, 1], index_topk=TOP_K
                ),
                max_model_len=max_len,
            ),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=self.max_tokens, max_num_seqs=max(requests, 8)
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1, prefill_context_parallel_size=1
            ),
            speculative_config=None,
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        with set_current_vllm_config(config):
            swa = DeepseekV4SWACache(
                head_dim=DIM,
                window_size=WINDOW,
                dtype=torch.uint8,
                prefix="swa",
                cache_config=CacheConfig(cache_dtype="nvfp4_ds_mla"),
                backend_cls=DeepseekV4FlashMLAAttention.swa_backend_cls,
                block_size=SWA_PAGE,
                packed_bytes_per_token=528,
                packed_page_alignment=512,
            )
        self.config, self.swa = config, swa
        self.builder = DeepseekSparseSWAFlashMLAMetadataBuilder(
            swa.get_kv_cache_spec(config), [swa.prefix], config, device
        )
        cpu = torch.Generator().manual_seed(seed)
        pages = -(-max_len // SWA_PAGE)
        self.tables = (
            torch.randperm(requests * pages + 3, generator=cpu)[: requests * pages]
            .view(requests, pages)
            .to(device=device, dtype=torch.int32)
        )
        swa_cache = torch.zeros(
            requests * pages + 3, SWA_PAGE * 528, device=device, dtype=torch.uint8
        )
        swa.kv_cache = swa_cache.view(-1, SWA_PAGE, 528)

        layer = DeepseekV4FlashMLAAttention.__new__(DeepseekV4FlashMLAAttention)
        torch.nn.Module.__init__(layer)
        layer.compress_ratio = compress_ratio
        layer.use_pcp = False
        layer.compressed_cache_prefix = "compressed" if compress_ratio else None
        layer.swa_cache_layer = swa
        layer.window_size = WINDOW
        layer.max_num_batched_tokens = self.max_tokens
        layer.topk_indices_buffer = torch.full(
            (self.max_tokens, TOP_K), -1, device=device, dtype=torch.int32
        )
        layer.scale = DIM**-0.5
        layer.attn_sink = inputs.sink
        self.layer = layer

        # Publish every token's KV with the serving op, Q RoPE in the same launch.
        positions, slots = inputs.positions, self._swa_slots()
        self.q_rope = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            inputs.q,
            inputs.kv,
            swa_cache,
            slots,
            positions,
            cos_sin,
            HEADS,
            1e-20,
            SWA_PAGE,
            False,
            True,
            True,
            False,
        )

        self.compressed_builder = None
        if compress_ratio:
            per_page = COMPRESSED_PAGE_TOKENS // compress_ratio
            cpages = -(-(max_len // compress_ratio) // per_page) + 1
            self.compressed_tables = (
                torch.randperm(requests * cpages + 2, generator=cpu)[: requests * cpages]
                .view(requests, cpages)
                .to(device=device, dtype=torch.int32)
            )
            compressed = torch.zeros(
                requests * cpages + 2, per_page, 288, device=device, dtype=torch.uint8
            )
            layer.is_kv_source = True
            layer.kv_cache = compressed
            layer._static_forward_context = {"compressed": layer}
            spec = MLAAttentionSpec(
                block_size=COMPRESSED_PAGE_TOKENS,
                num_kv_heads=1,
                head_size=DIM,
                dtype=torch.uint8,
                tokens_per_state=compress_ratio,
                state_content_bytes=288,
                cache_dtype_str="nvfp4_ds_mla",
                model_version="deepseek_v4",
            )
            self.compressed_builder = DeepseekV4FlashMLAMetadataBuilder(
                spec, ["compressed"], config, device
            )
            # The compressor emits one latent row per token; only group
            # boundaries are read. Fill the others with junk.
            token_latent = torch.randn(self.max_tokens, DIM, device=device).bfloat16()
            boundary = (positions + 1) % compress_ratio == 0
            token_latent[boundary] = inputs.latent
            rope_quant_insert(
                token_latent,
                positions,
                cos_sin,
                compressed,
                self._compressed_slots(),
                compress_ratio,
            )
        init_workspace_manager(device)

    def _rows(self):
        for r, (a, b) in enumerate(zip(self.inputs.cu, self.inputs.cu[1:])):
            yield r, a, b

    def _swa_slots(self):
        slots = []
        for r, a, b in self._rows():
            pos = torch.arange(b - a, device=self.device)
            slots.append(self.tables[r, pos // SWA_PAGE].long() * SWA_PAGE + pos % SWA_PAGE)
        return torch.cat(slots)

    def _compressed_slots(self):
        per_page = COMPRESSED_PAGE_TOKENS // self.cr
        slots = []
        for r, a, b in self._rows():
            entry = torch.arange(b - a, device=self.device) // self.cr
            table = self.compressed_tables[r]
            slots.append(table[entry // per_page].long() * per_page + entry % per_page)
        return torch.cat(slots)

    def run(self, schedule):
        """Attention output rows of ``schedule`` = [(request, start, end)]."""
        from vllm.forward_context import ForwardContext, override_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        device, cu = self.device, self.inputs.cu
        sizes = [end - start for _, start, end in schedule]
        starts_cpu = torch.tensor([0, *sizes], dtype=torch.int32).cumsum(0).int()
        rows = torch.cat(
            [torch.arange(cu[r] + s, cu[r] + e, device=device) for r, s, e in schedule]
        )
        q = self.q_rope[rows]
        positions = self.inputs.positions[rows]
        ends = [end for _, _, end in schedule]
        common = CommonAttentionMetadata(
            query_start_loc=starts_cpu.to(device),
            query_start_loc_cpu=starts_cpu,
            seq_lens=torch.tensor(ends, device=device, dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor(ends, dtype=torch.int32),
            num_reqs=len(schedule),
            num_actual_tokens=sum(sizes),
            max_query_len=max(sizes),
            max_seq_len=max(ends),
            block_table_tensor=torch.stack([self.tables[r] for r, _, _ in schedule]),
            slot_mapping=torch.arange(sum(sizes), device=device, dtype=torch.int64),
            causal=True,
        )
        metadata = {"swa": self.builder.build(0, common)}
        if self.cr:
            metadata["compressed"] = self.compressed_builder.build(
                0,
                replace(
                    common,
                    block_table_tensor=torch.stack(
                        [self.compressed_tables[r] for r, _, _ in schedule]
                    ),
                ),
            )
            self.layer.topk_indices_buffer[: rows.numel()] = self.inputs.topk[rows]
        output = torch.empty_like(q)
        with override_forward_context(ForwardContext({}, metadata, {})):
            self.layer.forward_mqa(q, q[:, 0], positions, output)
        return rows, output

    def close(self):
        from vllm.v1.worker.workspace import reset_workspace_manager

        torch.cuda.synchronize()
        reset_workspace_manager()


def output_quant(out, positions, cos_sin):
    """The FP8 head of serving's ``dsv41_o_proj`` (``deep_gemm_fp8_o_proj``)."""
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    return fused_inv_rope_fp8_quant(
        out,
        positions,
        cos_sin,
        n_groups=8,
        heads_per_group=8,
        nope_dim=448,
        rope_dim=64,
        quant_group_size=32,
        tma_aligned_scales=True,
        quantize=True,
    )


# ---------------------------------------------------------------------------
# FP64 reference VJP (no cuDNN, no closed-form RoPE transposes)
# ---------------------------------------------------------------------------


def rope64(x, positions, cos_sin, inverse=False):
    """GPT-J RoPE on the last 64 dims in FP64, written forward-only (autograd)."""
    cos = cos_sin[positions, :32].double()
    sin = cos_sin[positions, 32:64].double()
    shape = [cos.shape[0]] + [1] * (x.dim() - 2) + [32]
    cos, sin = cos.view(shape), sin.view(shape)
    pairs = x[..., -64:].unflatten(-1, (32, 2))
    e, o = pairs[..., 0], pairs[..., 1]
    if inverse:
        re, ro = e * cos + o * sin, o * cos - e * sin
    else:
        re, ro = e * cos - o * sin, o * cos + e * sin
    return torch.cat([x[..., :-64], torch.stack((re, ro), -1).flatten(-2)], -1)


class Bridge(torch.autograd.Function):
    """Value from the visible (serving) tensor, gradient into the surrogate."""

    @staticmethod
    def forward(ctx, visible, surrogate):
        return visible

    @staticmethod
    def backward(ctx, grad):
        return None, grad


def reference_grads(inputs, cfg, cos_sin, visible, d_o, chunk=32):
    """FP64 gradients of <inverse_rope(attention), d_o> under the STE contract.

    ``visible`` holds the serving-visible forward values (q after RoPE, gathered
    KV, indices, lengths) mapped back to tokens: ``q_rope [S,H,D]``,
    ``swa [S,D]``, ``comp [N,D]``, ``keys [S,K]`` (``('s', row)`` / ``('c', row)``
    encoded as row / N_total + row ... see ``key_rows``) and ``lens [S]``.
    """
    q = inputs.q.double().requires_grad_()
    kv = inputs.kv.double().requires_grad_()
    sink = inputs.sink.double().requires_grad_()
    latent = inputs.latent.double().requires_grad_() if inputs.latent is not None else None
    positions = inputs.positions

    q_used = Bridge.apply(visible.q_rope.double(), rope64(q, positions, cos_sin))
    swa_used = Bridge.apply(visible.swa.double(), rope64(kv, positions, cos_sin))
    parts = [swa_used]
    if latent is not None:
        comp_pos = visible.comp_positions // cfg.compress_ratio * cfg.compress_ratio
        parts.append(Bridge.apply(visible.comp.double(), rope64(latent, comp_pos, cos_sin)))
    table = torch.cat(parts)  # rows: S swa tokens then N compressed entries

    q_leaf = q_used.detach().requires_grad_()
    t_leaf = table.detach().requires_grad_()
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for a in range(0, q.shape[0], chunk):
        b = min(a + chunk, q.shape[0])
        keys = visible.keys[a:b]  # [c, K] rows of ``table`` or -1
        valid = keys >= 0
        k = t_leaf[keys.clamp(min=0)]  # [c, K, D]
        logits = torch.einsum("chd,ckd->chk", q_leaf[a:b], k) * cfg.softmax_scale
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        logits = torch.cat([logits, sink.view(1, -1, 1).expand(b - a, -1, 1)], -1)
        p = torch.softmax(logits, -1)[..., :-1]
        o = torch.einsum("chk,ckd->chd", p, k)
        o_rot = rope64(o, positions[a:b], cos_sin, inverse=True)
        (o_rot * d_o[a:b].double()).sum().backward()
        out[a:b] = o.detach()
    torch.autograd.backward([q_used, table], [q_leaf.grad, t_leaf.grad])
    return SimpleNamespace(
        q=q.grad,
        kv=kv.grad,
        latent=None if latent is None else latent.grad,
        sink=sink.grad,
        out=out,
    )

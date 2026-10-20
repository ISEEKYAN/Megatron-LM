# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 on vLLM BI=1 prefill operators, over the native Lite parameter tree.

The native model owns construction, release loading, DistOpt and checkpoints;
this subclass only replaces ``forward`` with :func:`.forward.model_forward`,
reading the live native parameters by their release names. The attention core
comes from M (mega) or F (FlashMLA) per tracks.md; select with
``DSV41_ATTENTION_CORE=mega|flashmla`` (default mega, the B_bi1_mega rollout).

Scope: PP=1, TP=1; CP any (contiguous equal THD slices, as the native protocol
packs them); EP any (MegaMoE symmetric buffer over the EP group).
"""

import os
from types import SimpleNamespace

import torch

from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.lite.model.deepseek_v41.lite.checkpoint import source_name
from megatron.lite.model.deepseek_v41.lite.components.config import validate_runtime_cp
from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model as NativeModel

from . import forward as F
from .attention import ContextParallel
from .head import head_log_probs, head_logits
from .moe import MegaMoE
from .norm import ModelNorm


class Head(ColumnParallelLinear):
    def forward(self, x, weight=None, **kwargs):
        return head_logits(x, self.weight if weight is None else weight), None


def doc_positions(prefix, rows):
    """Positions within each packed document."""
    starts = prefix[:-1].to(torch.int64)
    index = torch.arange(rows, device=prefix.device)
    owner = torch.searchsorted(prefix[1:].to(torch.int64), index, right=True)
    return index - starts[owner]


class DeepseekV41Model(NativeModel):
    deployment_providers = {
        'attention': 'vLLM BI projections/compressor/indexer/wo_a/wo_b; core per tracks.md (M/F)',
        'engram': 'frozen native FP8 rows; vLLM MXFP8 wkv + fused post-wkv gate',
        'moe': 'vLLM bf16_mega_gate + DeepGEMM fp8_fp4_mega_moe over the EP group',
        'mhc': 'vLLM pre_delayed / shifted_post_pre / post_tilelang / hc_collapse',
        'head': 'vLLM BI BF16 GEMM -> BF16 logits -> native_log_softmax',
    }

    def _build_mlp(self, config, release, index):
        spec = get_mlp_module_spec(
            use_te=True, num_experts=release.n_routed_experts, moe_grouped_gemm=True
        )
        return MegaMoE(
            config=config,
            submodules=spec.keywords['submodules'],
            layer_number=index + 1,
            pg_collection=self.context.groups,
            hash_moe_layer_threshold=release.num_hash_layers,
            vision_routing=release.vision_routing,
        )

    def _build_head(self, config, release):
        norm = ModelNorm(config, release.hidden_size, release.rms_norm_eps)
        head = Head(
            release.hidden_size,
            release.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            tp_group=self.ps.tp_group,
        )
        head.weight.is_embedding_or_output_parameter = True
        return norm, head

    # ---- deployment forward ------------------------------------------------------

    def _release_tensors(self):
        """``release key -> live native tensor`` (scale splits are read via mixes)."""
        table = {}
        for name, tensor in list(self.named_parameters()) + list(self.named_buffers()):
            if name.startswith(('engrams.', 'hash.')) or '.ffn.' in name:
                continue
            if name.endswith(('.alpha_pre', '.alpha_post', '.alpha_res', '.ape')):
                continue
            try:
                key = source_name(name)
            except KeyError:
                continue
            if not key.endswith(('hc_attn_fn', 'hc_ffn_fn', 'hc_attn_base', 'hc_ffn_base')):
                table.setdefault(key, tensor)
        return table

    def _params(self):
        table = self._release_tensors()
        layers = []
        for index in range(self.release.num_hidden_layers):
            block = self.layers[str(index)]
            for mixes in (block.attn_mixes, block.ffn_mixes):
                if mixes.mapping_proj.weight.dtype != torch.float32:
                    raise TypeError('vLLM mHC kernels take the FP32 hc_fn of the release')
            engram = self.engrams[str(index)] if str(index) in self.engrams else None
            layers.append(
                SimpleNamespace(
                    attn_norm=block.attn_norm.weight,
                    ffn_norm=block.ffn_norm.weight,
                    attn_mixes=block.attn_mixes,
                    ffn_mixes=block.ffn_mixes,
                    attn=F.attention_params(table.get, index),
                    engram=None
                    if engram is None
                    else SimpleNamespace(
                        wkv=engram.wkv.weight, q_weight=engram.q_weight, k_weight=engram.k_weight
                    ),
                )
            )
        return SimpleNamespace(embed=self.embed.weight, norm=self.norm.weight, layers=layers)

    def _rope(self):
        if not hasattr(self, '_rope_cache'):
            self._rope_cache = {}

        def get(ratio):
            key = ratio > 0
            if key not in self._rope_cache:
                from vllm.config import VllmConfig, set_current_vllm_config
                from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
                from vllm.transformers_utils.config import get_config

                hf = get_config(self.impl.hf_path, trust_remote_code=False)
                text = getattr(hf, 'text_config', hf)
                # Built on the GPU like the serving model: CPU cos/sin differ in ulps.
                with set_current_vllm_config(VllmConfig()), torch.device('cuda'):
                    rope = build_deepseek_v4_rope(
                        text,
                        head_dim=text.head_dim,
                        rope_head_dim=text.qk_rope_head_dim,
                        max_position_embeddings=text.max_position_embeddings,
                        compress_ratio=ratio,
                    )
                self._rope_cache[key] = rope.cos_sin_cache.to(torch.cuda.current_device())
            return self._rope_cache[key]

        return get

    def _core(self):
        kind = os.environ.get('DSV41_ATTENTION_CORE', 'mega')
        return F.interface_core(kind), kind

    def forward(
        self,
        input_ids,
        *,
        packed_seq_params,
        labels=None,
        loss_mask=None,
        full_input_ids=None,
        token_mask=None,
        temperature=1.0,
        calculate_entropy=False,
        hidden_states=None,
    ):
        if not (self.layout.has_embed and self.layout.has_head) or hidden_states is not None:
            raise NotImplementedError('Deployment DS4.1 forward is PP=1')
        if self.ps.tp_size > 1:
            raise NotImplementedError('Deployment DS4.1 forward is TP=1')
        params = packed_seq_params
        ids = input_ids.reshape(-1)
        rows = ids.shape[0]
        prefix = params.cu_seqlens_q_padded
        if prefix is None:
            prefix = params.cu_seqlens_q
        cp = None
        offset = self.ps.cp_rank * rows if self.ps.cp_size > 1 else 0
        if self.ps.cp_size > 1:
            validate_runtime_cp(self.release, self.ps.cp_group, params)
            if full_input_ids is None or full_input_ids.numel() != rows * self.ps.cp_size:
                raise ValueError('DS4.1 CP needs the full packed IDs over equal slices')
            cp = ContextParallel(
                group=self.ps.cp_group,
                offset=offset,
                positions=doc_positions(prefix, rows * self.ps.cp_size),
            )
            positions = cp.positions[offset : offset + rows]
        else:
            full_input_ids = ids
            positions = doc_positions(prefix, rows)
        hashes = None
        if self.engrams:
            hashes = self.hash(full_input_ids.reshape(1, -1), cu_seqlens=prefix)
            hashes = hashes[:, offset : offset + rows].transpose(0, 1)

        def engram_rows(layer):
            module = self.engrams[str(layer)]
            return module.embed(hashes[:, 0, module.layer_hash_index]).flatten(-2)

        def moe(x, tokens, layer):
            return self.layers[str(layer)].ffn(x, input_ids=tokens)[0]

        core, kind = self._core()
        state = F.ForwardState(self._rope())
        hidden = F.model_forward(
            ids,
            positions,
            self._params(),
            vars(self.release),
            state,
            core=core,
            moe=moe,
            engram_rows=engram_rows,
            cu_seqlens=prefix,
            layout='mega' if kind == 'mega' else 'flashmla',
            keep=None if token_mask is None else token_mask.reshape(-1).bool(),
            cp=cp,
            recompute=self.impl.recompute,
        )
        output = {'rows': torch.arange(rows, device=ids.device) + offset}
        if labels is None:
            output['logits'] = head_logits(hidden, self.head.weight)
            return output
        picked, spread = head_log_probs(
            hidden, self.head.weight, labels.reshape(-1), temperature, entropy=calculate_entropy
        )
        mask = (
            torch.ones_like(picked) if loss_mask is None else loss_mask.reshape(-1).to(picked.dtype)
        )
        output.update(
            log_probs=picked,
            loss=-(picked * mask).sum() / mask.sum().clamp_min(1),
            valid_mask=mask.bool(),
        )
        if calculate_entropy:
            output['entropy'] = spread
        return output

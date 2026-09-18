# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text assembly with local expert shards and explicit archival owners.

Vision/aligner have live differentiable owners; DSpark remains archival.
The floating diagnostic mode is explicit; it is not native quantized parity.
"""

from functools import partial
from types import SimpleNamespace

import torch
from megatron.lite.primitive.config_fields import project_fields
from megatron.lite.primitive.modules import engram_lookup as memory
from megatron.lite.primitive.modules import paired_stream as stream
from megatron.lite.primitive.modules import vision_training as visual
from megatron.lite.primitive.modules.attention.csa import (
    AttentionState,
    CompressedSparseAttention,
    Linear,
)
from megatron.lite.primitive.modules.image_data import validate_image_spans
from megatron.lite.primitive.modules.mlp import SwiGLUMLP
from megatron.lite.primitive.modules.native_fp32_linear import FP4Linear
from megatron.lite.primitive.modules.row_memory_build import (
    build_row_memories,
    sequence_hashes,
)
from megatron.lite.primitive.modules.vision import Aligner, ViT
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.primitive.utils import ensure_divisible
from torch import nn

from ..codecs import CODECS, attention_codecs
from .block import DeepseekV41Block, RMSNorm, contract_hc, expand_hc
from .checkpoint import DeferredModule, Rule, TensorBinding, validate_execution
from .moe import DeepseekV41MoE, ModalityRouter


class DeepseekV41Model(nn.Module):
    def __init__(
        self,
        config,
        *,
        token_map=None,
        quantized=True,
        trainable_engram=False,
        shard_engram=True,
        gate_temperature=1.0,
        bias_rate=0.001,
        enable_dspark_execution=False,
        layer_range=None,
        parallel_state=None,
    ):
        nn.Module.__init__(self)
        validate_execution(enable_dspark_execution=enable_dspark_execution)
        self.config = config
        self.topology = config.topology
        self.pipeline_cut = next(
            policy.index for policy in self.topology if policy.candidate_mode == "build"
        )
        self.ps = parallel_state or ParallelState()
        self.engram_group = (
            self.ps.dp_cp_group if shard_engram and self.ps.dp_cp_size > 1 else None
        )
        cfg = config.to_hf_dict()
        t, v = (SimpleNamespace(**cfg[key]) for key in ('text_config', 'vision_config'))
        dim, copies, eps = t.hidden_size, t.hc_mult, t.rms_norm_eps
        local_experts = ensure_divisible(t.n_routed_experts, self.ps.ep_size)
        expert_start = self.ps.ep_rank * local_experts
        self.hc_mult = copies
        self.vision_schedule = None
        self.register_buffer(
            '_vision_trainability', torch.full((4,), -1, dtype=torch.int8)
        )
        self.register_load_state_dict_post_hook(self._restore_vision_trainability)
        count = t.num_hidden_layers
        start, end = self.initialize_bindings(layer_range, count)
        self.embed = self.norm = self.head = None
        if start == 0:
            self.embed = nn.Embedding(t.vocab_size, dim, dtype=torch.bfloat16)
        if end == count:
            self.norm = RMSNorm(dim, eps)
            self.head = nn.Linear(dim, t.vocab_size, bias=False, dtype=torch.float32)
        self.layers = nn.ModuleList([None] * count)
        ac = config.attention_config(
            **dict.fromkeys(
                ('linear_fp8', 'main_qat', 'index_qat', 'swa_fp8'), quantized
            )
        )
        for layer_id in range(start, end):
            policy = self.topology[layer_id]
            attention = CompressedSparseAttention(
                ac,
                layer_idx=layer_id,
                ps=self.ps,
                compress_ratio=policy.compress_ratio,
                kv_owner=policy.kv_owner,
                index_owner=policy.index_owner,
                candidate_mode=policy.candidate_mode,
                query_head_rms=False,
                codecs=attention_codecs(),
            )
            router = ModalityRouter(
                t,
                SimpleNamespace(tp_size=1),
                gate_temperature=gate_temperature,
                bias_rate=bias_rate,
            )
            experts = [
                (
                    self._expert(t, quantized, shared=False)
                    if expert_start <= i < expert_start + local_experts
                    else None
                )
                for i in range(t.n_routed_experts)
            ]
            shared = (
                self._expert(t, quantized, shared=True) if t.n_shared_experts else None
            )
            ffn = DeepseekV41MoE(router, experts, shared, ps=self.ps)
            block = DeepseekV41Block(
                dim,
                copies,
                attention,
                ffn,
                norm_eps=eps,
                hc_eps=t.hc_eps,
                iterations=t.hc_sinkhorn_iters,
            )
            block.engram = None
            self.layers[layer_id] = block
        self.engram_layer_ids = tuple(t.engram_layer_ids)
        self.engram_hash, memories = build_row_memories(
            **project_fields(
                vars(t),
                'layer_ids=engram_layer_ids row_counts=engram_num_embeddings '
                'order=engram_max_ngram_size heads=engram_n_heads vocabulary=engram_vocab_size '
                'width=engram_head_dim compressed_vocabulary=engram_compressed_vocab_size '
                'pad_id=engram_pad_token_id token_count=vocab_size hidden_size '
                'copies=hc_mult eps=rms_norm_eps',
            ),
            token_map=token_map,
            trainable=trainable_engram,
            group=self.engram_group,
            group_size=self.ps.dp_cp_size,
            local_range=(start, end),
            projection=partial(
                Linear,
                fp8=quantized,
                fp8_operator=CODECS[("linear", 32, "e8m0", "e4m3")],
            ),
            constructors=memory.MEMORY_FACTORIES,
        )
        for index, module in memories.items():
            self.layers[index].engram = module
        self.vision = None
        self.aligner = None
        if start == 0:
            vision_args = SimpleNamespace(
                dim=dim,
                **project_fields(
                    vars(v),
                    'vision_dim=hidden_size vision_n_heads=num_attention_heads '
                    'vision_n_layers=num_hidden_layers vision_inter_dim=intermediate_size '
                    'vision_patch_size=patch_size vision_rope_theta=rope_theta '
                    'vision_downsample_ratio=downsample_ratio',
                ),
            )
            self.vision = ViT(vision_args)
            self.aligner = Aligner(vision_args)
            for key in ('image_start', 'image_end', 'image_newline'):
                self.register_parameter(key, nn.Parameter(torch.zeros(dim)))
        self.mtp = DeferredModule('DSpark')
        self.archival_bindings = {
            key: TensorBinding(key, self.mtp, None, 'archival')
            for key in self._archive_keys(t)
        }
        self._bind_table(t)
        self.validate_parameter_bindings()

    def _scale_binding(self, key, owner, attribute, role, encoding):
        if encoding in ('I8', 'F8_E4M3') and role != 'engram_table':
            return TensorBinding(
                key[:-6] + 'scale', owner, attribute, 'scale', encoding='F8_E8M0'
            )

    @staticmethod
    def _expert(t, quantized, shared):
        dim = t.hidden_size
        width = t.moe_intermediate_size * (t.n_shared_experts if shared else 1)

        projection = (
            partial(
                Linear,
                fp8=quantized,
                fp8_operator=CODECS[("linear", 32, "e8m0", "e4m3")],
            )
            if shared
            else partial(
                FP4Linear,
                quantized=quantized,
                fake_quant=CODECS[("index", 32, "e8m0", "e2m1")],
            )
        )
        return SwiGLUMLP.from_projections(
            *(projection(a, b) for a, b in ((dim, width), (width, dim), (dim, width))),
            swiglu_limit=t.swiglu_limit,
        )

    def _bind_table(self, t):
        # Checkpoint patterns bind objects; optimizer routes independently audit
        # actual owners and logical matrix shapes, never release-name prefixes.
        fp8 = 'F8_E4M3'
        heads = (t.num_attention_heads, t.head_dim, t.q_lora_rank)
        index_heads = (t.index_n_heads, t.index_head_dim, t.q_lora_rank)
        rules = {
            'embed': Rule('weight', 'embedding'),
            'norm': Rule('weight', 'norm'),
            'head': Rule('weight', 'head'),
            'layers.*.attn.wq_b': Rule('weight', 'wq_b', fp8, heads),
            'layers.*.attn.indexer.wq_b': Rule('weight', 'indexer', fp8, index_heads),
            'layers.*.attn': Rule('attn_sink', 'attention_sink'),
            'layers.*.ffn.gate': Rule('bias bias_vl', 'router_bias'),
            'layers.*.ffn.gate.router.gate': Rule(
                'weight', 'router', key='{grandparent}.{a}'
            ),
            'layers.*.ffn.experts.*.w[123]': Rule('weight', 'expert', 'I8'),
            'layers.*.ffn.shared_experts.w[123]': Rule('weight', 'shared_expert', fp8),
            'layers.*.engram.wkv': Rule('weight', 'engram_projection', fp8),
            'layers.*.engram': Rule('q_weight k_weight', 'engram_norm'),
            'layers.*.engram.embed': Rule('scale', 'scale', 'F8_E8M0'),
            '': Rule(
                'image_start image_end image_newline', 'image_delimiter', key='{a}'
            ),
        }
        rules.update(
            (f'layers.*.attn.{n}', Rule('weight', n, fp8))
            for n in ('wq_a', 'wkv', 'wo_a', 'wo_b')
        )
        rules.update(
            (f'layers.*.attn.{n}_norm', Rule('weight', 'norm')) for n in ('q', 'kv')
        )
        for role, names in (
            ('compressor', 'wkv norm wgate'),
            ('indexer', 'weights_proj wk k_norm'),
        ):
            rules.update(
                (f'layers.*.attn.{role}.{n}', Rule('weight', role))
                for n in names.split()
            )
        for side in ('attn', 'ffn'):
            rules[f'layers.*.{side}_norm'] = Rule('weight', 'norm')
            rules[f'layers.*.{side}_mixes'] = Rule(
                'fn base scale', 'hyper_connection', key='{parent}.hc_' + side + '_{a}'
            )
        self.bind_rules(rules, self._extra_binding_rules)

    @staticmethod
    def _extra_binding_rules(path, owner):
        entries = []
        if isinstance(owner, (memory.EngramTable, memory.ShardedEngramTable)):
            attribute = 'weight' if owner.master is None else 'master'
            entries.append(
                Rule(attribute, 'engram_table', 'F8_E4M3', key='{module}.weight')
            )
        if path.split('.')[0] in ('vision', 'aligner'):
            entries.append(
                Rule(
                    ' '.join(dict(owner.named_parameters(recurse=False))),
                    path.split('.')[0],
                )
            )
        return entries

    @staticmethod
    def _archive_keys(t):
        plain = (
            'attn.attn_sink attn.kv_norm.weight attn.q_norm.weight '
            'attn_norm.weight ffn_norm.weight ffn.gate.weight '
            'ffn.gate.bias ffn.gate.bias_vl'
        ).split()
        plain += [
            f'hc_{side}_{attr}'
            for side in ('attn', 'ffn')
            for attr in ('fn', 'base', 'scale')
        ]
        experts = [f'experts.{i}' for i in range(t.dspark_n_routed_experts)]
        experts += ['shared_experts'] if t.n_shared_experts else []
        matrices = [f'attn.{n}' for n in ('wkv', 'wo_a', 'wo_b', 'wq_a', 'wq_b')]
        matrices += [f'ffn.{e}.{n}' for e in experts for n in ('w1', 'w2', 'w3')]
        common = plain + [f'{n}.{a}' for n in matrices for a in ('weight', 'scale')]
        for i in range(t.num_nextn_predict_layers):
            keys = common[:]
            if i == 0:
                keys += 'main_norm.weight main_proj.weight main_proj.scale'.split()
            if i == t.num_nextn_predict_layers - 1:
                keys += (
                    'confidence_head.proj.weight markov_head.embed.weight '
                    'markov_head.head.weight norm.weight'
                ).split()
            yield from (f'mtp.{i}.{key}' for key in keys)

    def _sequence(
        self,
        hidden,
        pre,
        *,
        input_ids,
        image_mask=None,
        modality_loads=None,
        cp_context=None,
    ):
        start, end = self.local_layer_range
        token_mask = None if image_mask is None else ~image_mask
        hashes = (
            sequence_hashes(self.engram_hash, input_ids, token_mask, cp_context)
            if any(start <= i < end for i in self.engram_layer_ids)
            else None
        )
        state = AttentionState()
        for index in range(start, end):
            layer = self.layers[index]
            if layer.engram is not None:
                hidden = layer.engram(
                    hidden, hashes[:, :, self.topology[index].engram_slot], token_mask
                )
            hidden, pre, state = layer.forward_with_state(
                hidden,
                pre,
                state,
                attention_kwargs={'cp_context': cp_context},
                ffn_kwargs={
                    'image_mask': image_mask,
                    'load_sink': (
                        None if modality_loads is None else modality_loads[index]
                    ),
                },
            )
        return hidden, pre

    def set_input_tensor(self, input_tensor):
        """Receive the FP32 paired HC carrier through the shared PP interface."""
        if self.local_layer_range[0] != self.pipeline_cut:
            raise RuntimeError(
                'V4.1_PP_INPUT_STAGE: only the receiving stage accepts input'
            )
        if self._input_tensor is not None:
            raise RuntimeError('V4.1_PP_INPUT_PENDING: previous input was not consumed')
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids,
        *,
        cu_seqlens=None,
        images=None,
        token_types=None,
        cp_context=None,
        return_head_hidden=False,
    ):
        from .protocol import packed_paired_forward as packed_forward

        local_start, local_end = self.local_layer_range
        cut, count = self.pipeline_cut, len(self.topology)
        if (local_start, local_end) not in ((0, count), (0, cut), (cut, count)):
            raise RuntimeError(
                'V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED: use the range protocol'
            )
        if self.ps.pp_size > 1 and (images is not None or token_types is not None):
            raise NotImplementedError(
                'V4.1_PP_TEXT_ONLY: use PP=1 for multimodal training'
            )
        stream.validate_packed_input(
            input_ids, cu_seqlens, cp_context, self.ps.cp_size, self.engram_group
        )
        loads = None
        image_mask = None
        if local_start == cut:
            carrier, self._input_tensor = self._input_tensor, None
            hidden, pre = stream.unpack_pair(
                carrier,
                input_ids.shape,
                self.hc_mult,
                self.config.hidden_size,
                self.pipeline_residual_dtype,
                'V4.1_PP_PAIRED_INPUT: expected FP32 packed hidden/pre_mix',
            )
        else:
            embeddings = self.embed(input_ids)
            if hasattr(self, 'residual_dtype'):
                embeddings = embeddings.to(self.residual_dtype)
            image_mask = validate_image_spans(
                input_ids, images, token_types, cu_seqlens
            )
            if images is not None:
                embeddings = self.merge_image_embeddings(images, embeddings)
            hidden, pre = expand_hc(embeddings, self.hc_mult)
            loads = [[] for _ in self.layers]
        sequence = partial(self._sequence, modality_loads=loads)
        if cu_seqlens is not None:
            sequence = partial(
                packed_forward, sequence, cu_seqlens=cu_seqlens, cp_context=cp_context
            )
        hidden, pre = sequence(hidden, pre, input_ids=input_ids, image_mask=image_mask)
        if local_end == cut:
            return {'hidden_states': stream.pack_pair(hidden, pre)}
        head_hidden = self.norm(contract_hc(hidden, pre)).float()
        result = (
            {'head_hidden': head_hidden}
            if return_head_hidden
            else {
                'logits': torch.nn.functional.linear(
                    head_hidden, self.head.weight.float()
                )
            }
        )
        if loads is not None:
            result['modality_loads'] = tuple(tuple(entries) for entries in loads)
        return result

    _restore_vision_trainability = staticmethod(visual.restore_vision_trainability)
    encode_image = visual.encode_image
    merge_image_embeddings = visual.merge_image_inputs

    def forward_spec(self, *args, **kwargs):
        raise NotImplementedError('DSpark execution is not implemented')

    def parameter_bindings(self):
        return (
            b
            for b in self.tensor_bindings.values()
            if b.role != 'scale' and isinstance(b.tensor, nn.Parameter)
        )

    from megatron.lite.primitive.ckpt.binding_records import (
        _bind,
        bind_rules,
        initialize_bindings,
        validate_parameter_bindings,
    )

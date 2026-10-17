# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text assembly with local expert shards and explicit archival owners.

Vision/aligner have live differentiable owners; DSpark remains archival.
The floating diagnostic mode is explicit; it is not native quantized parity.
"""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import megatron.lite.primitive.modules.attention.csa as attention_primitives
import torch
from megatron.lite.primitive.config_fields import project_fields
from megatron.lite.primitive.modules import engram_lookup as memory
from megatron.lite.primitive.modules import paired_stream as stream
from megatron.lite.primitive.modules import row_memory_build as row_memories
from megatron.lite.primitive.modules import vision_training as visual
from megatron.lite.primitive.modules.image_data import validate_image_spans
from megatron.lite.primitive.modules.mlp import SwiGLUMLP
from megatron.lite.primitive.modules.native_fp32_linear import FP4Linear
from megatron.lite.primitive.modules.vision import Aligner, ViT
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.primitive.utils import ensure_divisible
from torch import nn

from ..codecs import CODECS, attention_codecs
from ..topology import pipeline_cut
from . import block as block_primitives
from . import checkpoint as checkpoint_primitives
from .moe import DeepseekV41MoE, ModalityRouter

DeferredModule = checkpoint_primitives.DeferredModule
Rule = checkpoint_primitives.Rule
TensorBinding = checkpoint_primitives.TensorBinding
validate_execution = checkpoint_primitives.validate_execution
DeepseekV41Block = block_primitives.DeepseekV41Block
RMSNorm = block_primitives.RMSNorm
contract_hc = block_primitives.contract_hc
expand_hc = block_primitives.expand_hc
AttentionState = attention_primitives.AttentionState
CompressedSparseAttention = attention_primitives.CompressedSparseAttention
Linear = attention_primitives.Linear
build_row_memories = row_memories.build_row_memories
sequence_hashes = row_memories.sequence_hashes


def _hashes(hash_module, input_ids, token_mask, cp_context, geometry):
    """Engram n-gram hashes; a packed slice hashes each document separately."""
    if geometry is None:
        return sequence_hashes(hash_module, input_ids, token_mask, cp_context)
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    sequence = None
    if geometry.cp_size > 1:
        sequence = ContiguousCPSequence(
            geometry.total, geometry.cp_rank, geometry.cp_size, geometry.cp_group
        )
    parts, offset = [], 0
    for doc in range(geometry.num_documents):
        start, stop = geometry.owned(doc)
        rows = slice(offset, offset + stop - start)
        context = None
        if sequence is not None:
            context = sequence.document(
                geometry.boundaries[doc], geometry.boundaries[doc + 1]
            )
        parts.append(
            sequence_hashes(
                hash_module,
                input_ids[:, rows],
                None if token_mask is None else token_mask[:, rows],
                context,
            )
        )
        offset = rows.stop
    return torch.cat(parts, 1)


_FP8_LINEAR = partial(Linear, fp8_operator=CODECS[("linear", 32, "e8m0", "e4m3")])


def sentinel_routing_mask(input_ids, routes_image_sentinels):
    """Tokens routed with ``bias_vl`` under deployment math.

    Native routing treats all five sentinel IDs as visual, including generated
    placeholders, but only for checkpoints with a vision tower: the reference
    gate has no ``bias_vl`` when ``vision_n_layers == 0`` and vLLM then routes
    the sentinels with the text bias.
    """
    if not routes_image_sentinels:
        return None
    return (input_ids >= 129264) & (input_ids < 129269)


_STATE_TENSORS = ('main_kv', 'index_k', 'indices', 'candidates')


def owns_attention_state(layer):
    """KV/index source layers publish CSA state that later layers consume."""
    return bool(
        getattr(layer.attn, 'owns_kv', False)
        or getattr(layer.attn, 'owns_index', False)
    )


def _first_call_only_sink(ffn_kwargs, calls):
    # Router statistics are published by the original forward only, never by
    # the backward-time recomputation.
    return ffn_kwargs if calls == 1 else {**ffn_kwargs, 'load_sink': None}


def recomputed_layer(layer, hidden, pre, state, attention_kwargs, ffn_kwargs, extra):
    """Full activation recompute of a layer that only consumes CSA state.

    The reentrant checkpoint routes gradients through positional tensor inputs
    only, so the consumed state tensors and the deployment HC carrier are passed
    explicitly; their gradients then reach the owner layer through this node
    instead of re-entering the owner's graph during recomputation.
    """
    from megatron.lite.primitive.recompute import CheckpointFunction
    from torch.utils._pytree import tree_flatten, tree_unflatten

    has_previous = 'previous_post' in extra
    # State fields may be containers (deployment IndexCache) and the carrier a
    # tuple; checkpoint the tensor leaves and rebuild the structure inside.
    consumed = (
        [getattr(state, name) for name in _STATE_TENSORS],
        extra.get('previous_post'),
    )
    leaves, spec = tree_flatten(consumed)
    slots = [i for i, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
    calls = [0]

    def run(h, p, *tensors):
        calls[0] += 1
        values = list(leaves)
        for slot, tensor in zip(slots, tensors):
            values[slot] = tensor
        fields, previous = tree_unflatten(values, spec)
        inner = replace(state, **dict(zip(_STATE_TENSORS, fields)))
        carrier = {'previous_post': previous} if has_previous else {}
        result = layer.forward(
            h,
            p,
            inner,
            attention_kwargs=attention_kwargs,
            ffn_kwargs=_first_call_only_sink(ffn_kwargs, calls[0]),
            **carrier,
        )
        if result[2] is not inner:
            raise RuntimeError(
                'V4.1_RECOMPUTE_OWNER: a state-consuming layer changed CSA state'
            )
        return (result[0], result[1], *(result[3] if has_previous else ()))

    outputs = CheckpointFunction.apply(
        run, False, hidden, pre, *(leaves[slot] for slot in slots)
    )
    if has_previous:
        return outputs[0], outputs[1], state, tuple(outputs[2:])
    return outputs[0], outputs[1], state


def recompute_ffn(module):
    """FFN-only recompute for CSA state owners (their attention keeps its graph)."""
    from megatron.lite.primitive.recompute import CheckpointFunction

    original = module.forward

    def forward(x, **kwargs):
        calls = [0]

        def run(value):
            calls[0] += 1
            return original(value, **_first_call_only_sink(kwargs, calls[0]))

        return CheckpointFunction.apply(run, False, x)

    module.forward = forward


class DeepseekV41Model(nn.Module):

    def __init__(
        self,
        config,
        *,
        token_map=None,
        quantized=True,
        w4a8_experts=False,
        deployment_math=False,
        use_deepep=False,
        trainable_engram=False,
        shard_engram=True,
        gate_temperature=1.0,
        bias_rate=0.001,
        enable_dspark_execution=False,
        layer_range=None,
        parallel_state=None,
        pipeline_split_layer=None,
    ):
        nn.Module.__init__(self)
        if deployment_math and (not w4a8_experts or not quantized):
            raise ValueError('Deployment math requires quantized W4A8 experts')
        validate_execution(enable_dspark_execution=enable_dspark_execution)
        self.config = config
        self.deployment_math = deployment_math
        self.topology = config.topology
        self.pipeline_cut = pipeline_cut(self.topology, pipeline_split_layer)
        self.ps = parallel_state or ParallelState()
        self.engram_group = (
            self.ps.dp_cp_group if shard_engram and self.ps.dp_cp_size > 1 else None
        )
        cfg = config.to_hf_dict()
        t, v = (SimpleNamespace(**cfg[key]) for key in ('text_config', 'vision_config'))
        # The reference gate owns bias_vl only with a vision tower.
        self.routes_image_sentinels = getattr(v, 'num_hidden_layers', 0) > 0
        # Same text_config key as the rollout: 32 is the official expert
        # contract (group-32 activations, routing weight before FC2 quant).
        act_group = getattr(t, 'expert_activation_group_size', 128)
        if act_group not in (32, 128):
            raise ValueError(f'Unsupported expert_activation_group_size {act_group}')
        official_expert_contract = act_group == 32
        wo_a_dtype = getattr(t, 'wo_a_dtype', 'fp8')
        if wo_a_dtype not in ('fp8', 'bf16'):
            raise ValueError(f'Unsupported wo_a_dtype {wo_a_dtype!r}')
        # Both contracts are part of the model, not of the deployment math: the
        # standard path runs them too (BF16 wo_a through the plain grouped
        # projection; experts through the official group-32 MLP and FMA combine).
        self.wo_a_bf16 = wo_a_dtype == 'bf16'
        if official_expert_contract and not w4a8_experts:
            raise ValueError('The official expert contract requires W4A8 experts')
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
            self.norm.deployment_math = deployment_math
            self.head = nn.Linear(dim, t.vocab_size, bias=False, dtype=torch.float32)
        self.layers = nn.ModuleList([None] * count)
        ac = config.attention_config(
            **dict.fromkeys(
                ('linear_fp8', 'main_qat', 'index_qat', 'swa_fp8'), quantized
            ),
            wo_a_bf16=self.wo_a_bf16,
        )
        for layer_id in range(start, end):
            policy = self.topology[layer_id]
            attention = CompressedSparseAttention(
                ac,
                layer_idx=layer_id,
                ps=self.ps,
                **project_fields(
                    vars(policy), 'compress_ratio kv_owner index_owner candidate_mode'
                ),
                codecs=attention_codecs(),
            )
            attention.deployment_math = deployment_math
            if attention.compressor is not None:
                attention.compressor.deployment_math = deployment_math
            for norm in attention.modules():
                if isinstance(norm, RMSNorm):
                    norm.deployment_math = deployment_math
            router = ModalityRouter(
                t,
                SimpleNamespace(tp_size=1),
                gate_temperature=gate_temperature,
                bias_rate=bias_rate,
                bias_group=self.ps.dp_cp_group if self.ps.pp_size > 1 else None,
            )
            router.deployment_math = deployment_math
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
            if shared is not None:
                shared.deployment_math = deployment_math
            ffn = DeepseekV41MoE(
                router,
                experts,
                shared,
                ps=self.ps,
                use_deepep=use_deepep,
                w4a8=w4a8_experts,
                official_expert_contract=official_expert_contract,
            )
            block = DeepseekV41Block(
                dim,
                copies,
                attention,
                ffn,
                norm_eps=eps,
                hc_eps=t.hc_eps,
                iterations=t.hc_sinkhorn_iters,
            )
            block.attn_norm.deployment_math = deployment_math
            block.ffn_norm.deployment_math = deployment_math
            block.attn_mixes.deployment_math = deployment_math
            block.attn_mixes.broadcast_projection = policy.broadcasts_embedding
            block.ffn_mixes.deployment_math = deployment_math
            block.engram = None
            self.layers[layer_id] = block
        self.engram_layer_ids = tuple(t.engram_layer_ids)
        memory_options = {
            'constructors': (
                memory.prime_buckets,
                memory.hash_multipliers,
                memory.NgramHash,
                memory.RowLookup,
                memory.EngramTable,
                memory.ShardedEngramTable,
                memory.Engram,
            )
        }
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
            projection=partial(_FP8_LINEAR, fp8=quantized),
            hash_layer_ids=getattr(t, 'engram_hash_layer_ids', None),
            **memory_options,
        )
        for index, module in memories.items():
            module.deployment_math = deployment_math
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
        # A BF16 wo_a exports no scale but still accepts the release FP8 pair,
        # which loads as its exact BF16 value.
        if (encoding in ('I8', 'F8_E4M3') and role != 'engram_table') or role == 'wo_a':
            return TensorBinding(
                key[:-6] + 'scale', owner, attribute, 'scale', encoding='F8_E8M0'
            )

    @staticmethod
    def _expert(t, quantized, shared):
        dim = t.hidden_size
        width = t.moe_intermediate_size * (t.n_shared_experts if shared else 1)

        projection = (
            partial(_FP8_LINEAR, fp8=quantized)
            if shared
            else partial(
                FP4Linear,
                quantized=quantized,
                fake_quant=CODECS[("index", 32, "e8m0", "e2m1")],
                activation_fake_quant=CODECS[("linear_activation", 32, "e8m0", "e4m3")],
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
        if self.wo_a_bf16:
            rules['layers.*.attn.wo_a'] = Rule('weight', 'wo_a')
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
        geometry=None,
    ):
        start, end = self.local_layer_range
        token_mask = None if image_mask is None else ~image_mask
        routing_mask = image_mask
        if self.deployment_math:
            routing_mask = sentinel_routing_mask(input_ids, self.routes_image_sentinels)
            # Engram only blocks the actual image ID.
            token_mask = input_ids != 129264
        hashes = (
            _hashes(self.engram_hash, input_ids, token_mask, cp_context, geometry)
            if any(start <= i < end for i in self.engram_layer_ids)
            else None
        )
        state = AttentionState()
        previous_post = None
        for index in range(start, end):
            layer = self.layers[index]
            if layer.engram is not None:
                hidden = layer.engram(
                    hidden, hashes[:, :, self.topology[index].engram_slot], token_mask
                )
            if self.deployment_math and layer.engram is not None:
                # Engram consumes the already-materialized previous HC post.
                # Do not let fused post/pre recompute and discard its injection.
                previous_post = None
            extra = (
                {'previous_post': previous_post}
                if self.deployment_math and hidden.is_cuda
                else {}
            )
            attention_kwargs = (
                {'cp_context': cp_context}
                if geometry is None
                else {'geometry': geometry}
            )
            ffn_kwargs = {
                'image_mask': routing_mask,
                'load_sink': (
                    None if modality_loads is None else modality_loads[index]
                ),
            }
            if (
                getattr(self, 'recompute', None) == 'full'
                and torch.is_grad_enabled()
                and not owns_attention_state(layer)
            ):
                result = recomputed_layer(
                    layer, hidden, pre, state, attention_kwargs, ffn_kwargs, extra
                )
            else:
                result = layer.forward(
                    hidden,
                    pre,
                    state,
                    attention_kwargs=attention_kwargs,
                    ffn_kwargs=ffn_kwargs,
                    **extra,
                )
            hidden, pre, state = result[:3]
            if extra:
                previous_post = result[3]
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
        if self.deployment_math and images is not None:
            raise NotImplementedError('Deployment math is text-only')
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
        # Stats are stage-local, just like the DS4 stage-local layer inventory.
        # _sequence indexes by global layer id; publish only the local interval.
        loads = [[] for _ in self.layers]
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
        sequence = partial(self._sequence, modality_loads=loads)
        if cu_seqlens is not None:
            sequence = partial(
                packed_forward, sequence, cu_seqlens=cu_seqlens, cp_context=cp_context
            )
        hidden, pre = sequence(hidden, pre, input_ids=input_ids, image_mask=image_mask)
        if local_end == cut and cut < count:
            return {
                'hidden_states': stream.pack_pair(hidden, pre),
                'modality_loads': tuple(
                    tuple(entries) for entries in loads[local_start:local_end]
                ),
            }
        if self.deployment_math and hidden.is_cuda:
            from megatron.lite.primitive.modules import deployment_math

            collapsed = deployment_math.rms_norm(
                deployment_math.hc_collapse(hidden, pre),
                self.norm.weight,
                self.norm.normalized_shape,
                self.norm.eps,
            )
        else:
            collapsed = self.norm(contract_hc(hidden, pre))
        head_hidden = collapsed.float()
        if self.deployment_math and collapsed.is_cuda:
            if return_head_hidden:
                raise NotImplementedError(
                    'Deployment math uses the owned head forward, not an external fused head'
                )
            from megatron.lite.primitive.modules import deployment_math

            result = {
                'logits': deployment_math.bf16_fp32_linear(collapsed, self.head.weight)
            }
            if loads is not None:
                result['modality_loads'] = tuple(
                    tuple(entries) for entries in loads[local_start:local_end]
                )
            return result

        result = (
            {
                'head_hidden': head_hidden,
                # A view gives DDP's unused-parameter traversal a grad_fn root.
                'head_weight': self.head.weight.view_as(self.head.weight).float(),
            }
            if return_head_hidden
            else {
                'logits': torch.nn.functional.linear(
                    head_hidden, self.head.weight.float()
                )
            }
        )
        if loads is not None:
            result['modality_loads'] = tuple(
                tuple(entries) for entries in loads[local_start:local_end]
            )
        return result

    _restore_vision_trainability = staticmethod(visual.restore_vision_trainability)
    encode_image = visual.encode_image
    merge_image_embeddings = visual.merge_image_inputs

    def parameter_bindings(self):
        return (
            b
            for b in self.tensor_bindings.values()
            if b.role != 'scale' and isinstance(b.tensor, nn.Parameter)
        )

    from megatron.lite.primitive.ckpt import binding_records

    _bind = binding_records._bind
    bind_rules = binding_records.bind_rules
    initialize_bindings = binding_records.initialize_bindings
    validate_parameter_bindings = binding_records.validate_parameter_bindings

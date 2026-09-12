# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-rank text assembly with explicit live tensor and archival owners.

Vision/aligner have live differentiable owners; DSpark remains archival.
The floating diagnostic mode is explicit; it is not native quantized parity.
"""

from collections import namedtuple
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import partial
from types import SimpleNamespace

import torch
from megatron.lite.primitive.modules.native_fp32_linear import FP4Linear
from torch import nn
from torch.nn import functional as F

from .attention import AttentionState, CSA2Attention, Linear
from .block import DeepseekV41Block, RMSNorm, contract_hc, expand_hc
from .checkpoint import validate_execution
from .engram import Engram, EngramTable, NgramHash, hash_multipliers, prime_buckets
from .image_data import TEXT, merge_image_embeddings
from .moe import DeepseekV41MoE, ModalityRouter, SwiGLUExpert
from .packing import packed_forward
from .vision import Aligner, ViT


@dataclass(frozen=True)
class TensorBinding:
    release_key: str
    owner: nn.Module
    attribute: str | None
    role: str
    head_count: int | None = None
    encoding: str | None = None
    header: object = None
    store: object = None
    matrix_shape: tuple | None = None

    @property
    def tensor(self):
        return None if self.attribute is None else getattr(self.owner, self.attribute)


class DeferredModule(nn.Module):
    """Archival subtree. F3 supplies vision/aligner computation at these interfaces."""

    def __init__(self, scope):
        super().__init__()
        self.scope = scope

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f'{self.scope} execution is not implemented in text-only mode'
        )


Rule = namedtuple(
    'Rule', 'attributes role encoding shape key', defaults=(None, None, None)
)


# Attention state -> PP carrier; native shapes and frozen selection are preserved.
_STATE_PAYLOAD = dict(
    zip(
        'latent main_kv index_k indices candidates'.split(),
        'latent kv index_k topk candidates'.split(),
    )
)


class DeepseekV41Model(nn.Module):
    def __init__(
        self,
        config,
        *,
        token_map=None,
        quantized=True,
        trainable_engram=False,
        gate_temperature=1.0,
        bias_rate=0.001,
        enable_dspark_execution=False,
        layer_range=None,
    ):
        super().__init__()
        validate_execution(enable_dspark_execution=enable_dspark_execution)
        self.config = config
        cfg = config.to_hf_dict()
        t, v = (SimpleNamespace(**cfg[key]) for key in ('text_config', 'vision_config'))
        dim, copies, eps = t.hidden_size, t.hc_mult, t.rms_norm_eps
        self.hc_mult = copies
        self.vision_schedule = None
        self.register_buffer(
            '_vision_trainability', torch.full((4,), -1, dtype=torch.int8)
        )
        self.register_load_state_dict_post_hook(self._restore_vision_trainability)
        count = t.num_hidden_layers
        start, end = (0, count) if layer_range is None else layer_range
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= count
        ):
            raise ValueError('Invalid local pipeline stage interval')
        self.local_layer_range = (start, end)
        self.tensor_bindings = {}
        self.archival_bindings = {}
        self.archival_store = None
        self.checkpoint_bindings = None
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
            attention = CSA2Attention(ac, layer_id)
            router = ModalityRouter(
                t,
                SimpleNamespace(tp_size=1),
                gate_temperature=gate_temperature,
                bias_rate=bias_rate,
            )
            experts = [
                self._expert(t, quantized, shared=False)
                for _ in range(t.n_routed_experts)
            ]
            shared = (
                self._expert(t, quantized, shared=True) if t.n_shared_experts else None
            )
            ffn = DeepseekV41MoE(router, experts, shared)
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
        self.engram_hash = None
        self.engram_layer_ids = tuple(t.engram_layer_ids)
        if self.engram_layer_ids:
            if token_map is not None:
                # Integer layout construction stays on CPU even for meta allocation.
                with torch.device('cpu'):
                    primes = prime_buckets(
                        self.engram_layer_ids,
                        t.engram_max_ngram_size,
                        t.engram_n_heads,
                        t.engram_vocab_size,
                    )
                    if primes.flatten(1).sum(1).tolist() != t.engram_num_embeddings:
                        raise ValueError(
                            'engram_num_embeddings disagrees with prime layout'
                        )
                    if (
                        len(token_map) != t.vocab_size
                        or min(token_map) < 0
                        or max(token_map) >= t.engram_compressed_vocab_size
                    ):
                        raise ValueError('token_map disagrees with Engram vocabulary')
                    multipliers = hash_multipliers(
                        self.engram_layer_ids,
                        t.engram_max_ngram_size,
                        t.engram_compressed_vocab_size,
                    )
                    self.engram_hash = NgramHash(
                        token_map, t.engram_pad_token_id, multipliers, primes
                    )
            for offset, layer_id in enumerate(self.engram_layer_ids):
                if not start <= layer_id < end:
                    continue
                rows, width = t.engram_num_embeddings[offset], t.engram_head_dim
                table = EngramTable(
                    torch.zeros(rows, width, dtype=torch.float8_e4m3fn),
                    torch.ones(rows, width // 32, dtype=torch.float8_e8m0fnu),
                    trainable=trainable_engram,
                )
                projection = Linear(
                    (t.engram_max_ngram_size - 1) * t.engram_n_heads * width,
                    (copies + 1) * dim,
                    fp8=quantized,
                )
                module = Engram(dim, copies, table, projection, eps=eps)
                self.layers[layer_id].engram = module
        self.vision = None
        self.aligner = None
        if start == 0:
            vision_args = SimpleNamespace(
                vision_dim=v.hidden_size,
                vision_n_heads=v.num_attention_heads,
                vision_n_layers=v.num_hidden_layers,
                vision_inter_dim=v.intermediate_size,
                vision_patch_size=v.patch_size,
                vision_rope_theta=v.rope_theta,
                vision_downsample_ratio=v.downsample_ratio,
                dim=dim,
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

    def _bind(
        self,
        key,
        owner,
        attribute,
        role,
        head_count=None,
        encoding=None,
        matrix_shape=None,
    ):
        if key in self.tensor_bindings:
            raise ValueError(f'duplicate binding: {key}')
        self.tensor_bindings[key] = TensorBinding(
            key, owner, attribute, role, head_count, encoding, matrix_shape=matrix_shape
        )
        if encoding in ('I8', 'F8_E4M3') and role != 'engram_table':
            scale = key[:-6] + 'scale'
            self.tensor_bindings[scale] = TensorBinding(
                scale, owner, attribute, 'scale', encoding='F8_E8M0'
            )

    @staticmethod
    def _expert(t, quantized, shared):
        dim = t.hidden_size
        width = t.moe_intermediate_size * (t.n_shared_experts if shared else 1)

        projection = (
            partial(Linear, fp8=quantized)
            if shared
            else partial(FP4Linear, quantized=quantized)
        )
        return SwiGLUExpert(
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
            ('indexer', 'wq_b weights_proj wk k_norm'),
        ):
            for n in names.split():
                indexed = role == 'indexer' and n == 'wq_b'
                rules[f'layers.*.attn.{role}.{n}'] = Rule(
                    'weight',
                    role,
                    fp8 if indexed else None,
                    index_heads if indexed else None,
                )
        for side in ('attn', 'ffn'):
            rules[f'layers.*.{side}_norm'] = Rule('weight', 'norm')
            rules[f'layers.*.{side}_mixes'] = Rule(
                'fn base scale', 'hyper_connection', key='{parent}.hc_' + side + '_{a}'
            )
        for path, owner in self.named_modules():
            entries = [
                rule for pattern, rule in rules.items() if fnmatchcase(path, pattern)
            ]
            if isinstance(owner, EngramTable):
                attribute = 'weight' if owner.master is None else 'master'
                entries.append(
                    Rule(attribute, 'engram_table', fp8, key='{module}.weight')
                )
            if path.split('.')[0] in ('vision', 'aligner'):
                entries.append(
                    Rule(
                        ' '.join(dict(owner.named_parameters(recurse=False))),
                        path.split('.')[0],
                    )
                )
            for attributes, role, encoding, shape, key in entries:
                for attribute in attributes.split():
                    tensor = getattr(owner, attribute, None)
                    if tensor is None:
                        continue
                    name = (key or '{module}.{a}').format(
                        module=path,
                        a=attribute,
                        parent=path.rsplit('.', 1)[0],
                        grandparent=path.rsplit('.', 2)[0],
                    )
                    axes = tuple(tensor.shape) if shape is None else shape
                    self._bind(
                        name,
                        owner,
                        attribute,
                        role,
                        None if shape is None else shape[0],
                        encoding,
                        axes,
                    )

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
        scopes = (
            (range(t.num_nextn_predict_layers), common),
            ((0,), ('main_norm.weight', 'main_proj.weight', 'main_proj.scale')),
            (
                (t.num_nextn_predict_layers - 1,),
                (
                    'confidence_head.proj.weight',
                    'markov_head.embed.weight',
                    'markov_head.head.weight',
                    'norm.weight',
                ),
            ),
        )
        return (
            f'mtp.{i}.{key}' for layers, keys in scopes for i in layers for key in keys
        )

    def parameter_bindings(self):
        return (
            b
            for b in self.tensor_bindings.values()
            if b.role != 'scale' and isinstance(b.tensor, nn.Parameter)
        )

    def validate_parameter_bindings(self):
        ids = [id(b.tensor) for b in self.parameter_bindings()]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters()}:
            raise ValueError('Every parameter must have exactly one binding')

    def _layers(
        self,
        hidden,
        pre,
        input_ids,
        start,
        end,
        state,
        ced,
        image_mask=None,
        modality_loads=None,
    ):
        token_mask = None if image_mask is None else ~image_mask
        hashes = None
        if any(start <= layer < end for layer in self.engram_layer_ids):
            if self.engram_hash is None:
                raise ValueError(
                    'Engram execution requires an explicit tokenizer token_map'
                )
            hashes = self.engram_hash(input_ids, token_mask)
        for index in range(start, end):
            layer = self.layers[index]
            if layer.engram is not None:
                hidden = layer.engram(
                    hidden, hashes[:, :, self.engram_layer_ids.index(index)], token_mask
                )
            hidden, pre, state = layer.forward_with_state(
                hidden,
                pre,
                state,
                ffn_kwargs={
                    'image_mask': image_mask,
                    'load_sink': (
                        None if modality_loads is None else modality_loads[index]
                    ),
                },
            )
            if index == 19:
                ced = hidden, pre
        return hidden, pre, state, ced

    def _sequence(
        self, hidden, pre, *, input_ids, image_mask=None, modality_loads=None
    ):
        return self._layers(
            hidden,
            pre,
            input_ids,
            0,
            len(self.layers),
            AttentionState(),
            (None, None),
            image_mask,
            modality_loads,
        )[:2]

    def forward_pipeline_range(
        self, input_ids, *, start, end, payload=None, owners=(-1, -1)
    ):
        """Execute a contiguous real layer range with explicit, graph-bearing state.

        This is the model boundary used by a PP scheduler, not a scheduler or a
        parameter sharder. Input IDs belong to one unpacked sequence batch; the
        caller must split packed samples before constructing each generation.
        """
        from .pipeline import PairedPayload

        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(self.layers)
        ):
            raise ValueError('Invalid pipeline layer interval')
        local_start, local_end = self.local_layer_range
        if not local_start <= start < end <= local_end:
            raise ValueError('Requested range is outside this pipeline stage')
        if (
            input_ids.ndim != 2
            or input_ids.dtype != torch.int64
            or not input_ids.shape[1]
        ):
            raise ValueError('Expected nonempty int64 input_ids [B,S]')
        if start == 0:
            if payload is not None or owners != (-1, -1):
                raise ValueError('First pipeline range must start with fresh state')
            hidden, pre = expand_hc(self.embed(input_ids), self.hc_mult)
            state = AttentionState()
            ced_h = ced_p = None
        else:
            if payload is None or payload.h.shape[:2] != input_ids.shape:
                raise ValueError('Pipeline range requires matching input state')
            if start >= 20 and (payload.ced_h is None or payload.ced_p is None):
                raise ValueError('Decoder pipeline range requires the saved CED pair')
            hidden, pre = payload.h, payload.p
            ced_h, ced_p = payload.ced_h, payload.ced_p
            state = AttentionState(
                kv_owner=None if owners[0] == -1 else owners[0],
                index_owner=None if owners[1] == -1 else owners[1],
                **{
                    name: getattr(payload, field)
                    for name, field in _STATE_PAYLOAD.items()
                },
            )
        # A stage beginning at p20 consumes the transported CED pair. A range
        # crossing p20 already holds this exact h19/p19 pair in its live stream.
        if start == 20:
            hidden, pre = ced_h, ced_p
        hidden, pre, state, (ced_h, ced_p) = self._layers(
            hidden, pre, input_ids, start, end, state, (ced_h, ced_p)
        )
        fields = {field: getattr(state, name) for name, field in _STATE_PAYLOAD.items()}
        # O12: index selection is discrete and has no indexer objective.
        if fields['index_k'] is not None:
            fields['index_k'] = fields['index_k'].detach()
        return PairedPayload(hidden, pre, ced_h, ced_p, **fields), tuple(
            -1 if owner is None else owner
            for owner in (state.kv_owner, state.index_owner)
        )

    def finish_pipeline(self, payload):
        """Apply the actual final shifted HC contraction, norm and output head."""
        if self.head is None or self.norm is None:
            raise RuntimeError('Only the final pipeline stage owns the output head')
        hidden = self.norm(contract_hc(payload.h, payload.p))
        return F.linear(hidden.float(), self.head.weight.float())

    def forward(self, input_ids, *, cu_seqlens=None, images=None, token_types=None):
        if self.local_layer_range != (0, len(self.layers)):
            raise RuntimeError('A local pipeline stage requires the range protocol')
        if (
            input_ids.ndim != 2
            or input_ids.dtype != torch.int64
            or not input_ids.shape[1]
        ):
            raise ValueError('Expected nonempty int64 input_ids [B,S]')
        embeddings = self.embed(input_ids)
        if hasattr(self, 'residual_dtype'):
            embeddings = embeddings.to(self.residual_dtype)
        image_mask = None
        if images is not None:
            if len(images) != len(input_ids):
                raise ValueError('Image batch size differs from input IDs')
            expected_types = torch.full_like(input_ids, TEXT)
            for batch, sample in enumerate(images):
                for img in sample or ():
                    if cu_seqlens is not None:
                        boundaries = cu_seqlens.tolist()
                        if not any(
                            a <= img.start and img.start + img.types.numel() <= b
                            for a, b in zip(boundaries, boundaries[1:])
                        ):
                            raise ValueError(
                                'Image span crosses a packed sequence boundary'
                            )
                    expected_types[batch, img.start : img.start + img.types.numel()] = (
                        img.types.to(input_ids.device)
                    )
            if token_types is not None and not torch.equal(
                token_types.to(input_ids.device), expected_types
            ):
                raise ValueError('Token types disagree with image spans')
            embeddings = self.merge_image_embeddings(images, embeddings)
            image_mask = expected_types >= 0
        elif token_types is not None:
            if token_types.shape != input_ids.shape or (token_types != TEXT).any():
                raise ValueError('Image token types require image inputs')
        hidden, pre = expand_hc(embeddings, self.hc_mult)
        loads = [[] for _ in self.layers]
        sequence = partial(self._sequence, modality_loads=loads)
        if cu_seqlens is None:
            hidden, pre = sequence(
                hidden, pre, input_ids=input_ids, image_mask=image_mask
            )
        else:
            hidden, pre = packed_forward(
                sequence,
                hidden,
                pre,
                cu_seqlens,
                input_ids=input_ids,
                image_mask=image_mask,
            )
        hidden = self.norm(contract_hc(hidden, pre))
        # Freeze membership before backward: recompute may revisit a sink, but
        # its statistics must not be submitted as another training microbatch.
        return {
            'logits': F.linear(hidden.float(), self.head.weight.float()),
            'modality_loads': tuple(tuple(entries) for entries in loads),
        }

    @staticmethod
    def _restore_vision_trainability(module, incompatible_keys):
        values = module._vision_trainability.tolist()
        if values == [-1] * 4:
            return
        if any(value not in (0, 1) for value in values):
            raise ValueError('Invalid post-training mask in checkpoint')
        from .training import VisionTrainability

        VisionTrainability(*map(bool, values)).apply(module)

    def encode_image(self, patches, n_vit_h, n_vit_w):
        weight = self.vision.patch_embed.proj.weight
        patches = patches.to(device=weight.device, dtype=weight.dtype)
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h):
        if self.vision_schedule is not None:
            features = self.vision_schedule.forward(images)
        else:
            features = [
                [
                    self.encode_image(img.patches, img.n_vit_h, img.n_vit_w)
                    for img in sample or ()
                ]
                for sample in images
            ]
        return merge_image_embeddings(
            h, images, features, self.image_start, self.image_end, self.image_newline
        )

    def forward_spec(self, *args, **kwargs):
        raise NotImplementedError('DSpark execution is not implemented')

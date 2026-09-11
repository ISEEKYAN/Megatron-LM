# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-rank text assembly with explicit live tensor and archival owners.

Vision/aligner have live differentiable owners; DSpark remains archival.
The floating diagnostic mode is explicit; it is not native quantized parity.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .attention import AttentionState, CSA2Attention, Linear
from .block import DeepseekV41Block, RMSNorm, contract_hc, expand_hc
from .checkpoint_store import validate_execution
from .engram import Engram, EngramTable, NgramHash, hash_multipliers, prime_buckets
from .moe import DeepseekV41MoE, ModalityRouter, SwiGLUExpert
from .packing import packed_forward
from .vision import ViT, Aligner
from .image_data import merge_image_embeddings, TEXT


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

    @property
    def tensor(self):
        return None if self.attribute is None else getattr(self.owner, self.attribute)


class DeferredModule(nn.Module):
    """Archival subtree. F3 supplies vision/aligner computation at these interfaces."""

    def __init__(self, scope):
        super().__init__()
        self.scope = scope

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'{self.scope} execution is not implemented in text-only mode')

    def leaf_owner(self, path):
        owner = self
        for part in path.split('.')[:-1]:
            if part not in owner._modules:
                owner.add_module(part, DeferredModule(f'{owner.scope}.{part}'))
            owner = owner._modules[part]
        return owner


class FP4Linear(Linear):
    """Group32 E8M0/E2M1 numerical provider with STE, not a native FP4 GEMM."""

    def __init__(self, input_size, output_size, *, quantized):
        super().__init__(input_size, output_size)
        self.quantized = quantized

    def forward(self, x):
        if getattr(self, 'native_fp32', False):
            from megatron.lite.primitive.modules.native_fp32_linear import (
                native_fp32_linear,
            )
            from megatron.lite.primitive.quantization.ds41_index import fake_quant_index

            return native_fp32_linear(
                fake_quant_index(x, enabled=self.quantized),
                fake_quant_index(self.weight, enabled=self.quantized),
            )
        if not self.quantized:
            return F.linear(x, self.weight)
        from megatron.lite.primitive.quantization.ds41_index import fake_quant_index

        return F.linear(fake_quant_index(x), fake_quant_index(self.weight))


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
    ):
        super().__init__()
        validate_execution(enable_dspark_execution=enable_dspark_execution)
        self.config = config
        cfg = config.to_hf_dict()
        t, v = cfg['text_config'], cfg['vision_config']
        self.hc_mult = t['hc_mult']
        self.tensor_bindings = {}
        self.archival_bindings = {}
        self.archival_store = None
        self.checkpoint_bindings = None
        self.embed = nn.Embedding(t['vocab_size'], t['hidden_size'], dtype=torch.bfloat16)
        self.norm = RMSNorm(t['hidden_size'], t['rms_norm_eps'])
        self.head = nn.Linear(t['hidden_size'], t['vocab_size'], bias=False, dtype=torch.float32)
        self._bind('embed.weight', self.embed, 'weight', 'embedding')
        self._bind('norm.weight', self.norm, 'weight', 'norm')
        self._bind('head.weight', self.head, 'weight', 'head')
        self.layers = nn.ModuleList()
        flags = dict(
            linear_fp8=quantized, main_qat=quantized, index_qat=quantized, swa_fp8=quantized
        )
        ac = config.attention_config(**flags)
        for layer_id in range(t['num_hidden_layers']):
            prefix = f'layers.{layer_id}'
            attention = CSA2Attention(ac, layer_id)
            router = ModalityRouter(
                SimpleNamespace(**t),
                SimpleNamespace(tp_size=1),
                gate_temperature=gate_temperature,
                bias_rate=bias_rate,
            )
            experts = [
                self._expert(t, quantized, shared=False) for _ in range(t['n_routed_experts'])
            ]
            shared = self._expert(t, quantized, shared=True) if t['n_shared_experts'] else None
            ffn = DeepseekV41MoE(router, experts, shared)
            block = DeepseekV41Block(
                t['hidden_size'],
                t['hc_mult'],
                attention,
                ffn,
                norm_eps=t['rms_norm_eps'],
                hc_eps=t['hc_eps'],
                iterations=t['hc_sinkhorn_iters'],
            )
            block.engram = None
            self.layers.append(block)
            self._bind_attention(prefix + '.attn', attention, t)
            self._bind(prefix + '.ffn.gate.weight', router.router.gate, 'weight', 'router')
            for attr in ('bias', 'bias_vl'):
                self._bind(prefix + '.ffn.gate.' + attr, router, attr, 'router_bias')
            for index, expert in enumerate(experts):
                self._bind_expert(f'{prefix}.ffn.experts.{index}', expert, 'expert', 'I8')
            if shared is not None:
                self._bind_expert(
                    prefix + '.ffn.shared_experts', shared, 'shared_expert', 'F8_E4M3'
                )
            for side in ('attn', 'ffn'):
                self._bind(
                    prefix + f'.{side}_norm.weight',
                    getattr(block, f'{side}_norm'),
                    'weight',
                    'norm',
                )
                mixes = getattr(block, f'{side}_mixes')
                for attr in ('fn', 'base', 'scale'):
                    self._bind(prefix + f'.hc_{side}_{attr}', mixes, attr, 'hyper_connection')
        self.engram_hash = None
        self.engram_layer_ids = tuple(t['engram_layer_ids'])
        if self.engram_layer_ids:
            if token_map is not None:
                # Integer layout construction stays on CPU even for meta allocation.
                with torch.device('cpu'):
                    primes = prime_buckets(
                        self.engram_layer_ids,
                        t['engram_max_ngram_size'],
                        t['engram_n_heads'],
                        t['engram_vocab_size'],
                    )
                    if primes.flatten(1).sum(1).tolist() != t['engram_num_embeddings']:
                        raise ValueError('engram_num_embeddings disagrees with prime layout')
                    if (
                        len(token_map) != t['vocab_size']
                        or min(token_map) < 0
                        or max(token_map) >= t['engram_compressed_vocab_size']
                    ):
                        raise ValueError('token_map disagrees with Engram vocabulary')
                    multipliers = hash_multipliers(
                        self.engram_layer_ids,
                        t['engram_max_ngram_size'],
                        t['engram_compressed_vocab_size'],
                    )
                    self.engram_hash = NgramHash(
                        token_map, t['engram_pad_token_id'], multipliers, primes
                    )
            for offset, layer_id in enumerate(self.engram_layer_ids):
                rows, width = t['engram_num_embeddings'][offset], t['engram_head_dim']
                table = EngramTable(
                    torch.zeros(rows, width, dtype=torch.float8_e4m3fn),
                    torch.ones(rows, width // 32, dtype=torch.float8_e8m0fnu),
                    trainable=trainable_engram,
                )
                projection = Linear(
                    (t['engram_max_ngram_size'] - 1) * t['engram_n_heads'] * width,
                    (t['hc_mult'] + 1) * t['hidden_size'],
                    fp8=quantized,
                )
                module = Engram(
                    t['hidden_size'], t['hc_mult'], table, projection, eps=t['rms_norm_eps']
                )
                self.layers[layer_id].engram = module
                prefix = f'layers.{layer_id}.engram'
                self._bind(
                    prefix + '.embed.weight',
                    table,
                    'master' if trainable_engram else 'weight',
                    'engram_table',
                    encoding='F8_E4M3',
                )
                self._bind(prefix + '.embed.scale', table, 'scale', 'scale', encoding='F8_E8M0')
                self._bind(
                    prefix + '.wkv.weight',
                    projection,
                    'weight',
                    'engram_projection',
                    encoding='F8_E4M3',
                )
                for attr in ('q_weight', 'k_weight'):
                    self._bind(prefix + '.' + attr, module, attr, 'engram_norm')
        vision_args = SimpleNamespace(
            vision_dim=v['hidden_size'],
            vision_n_heads=v['num_attention_heads'],
            vision_n_layers=v['num_hidden_layers'],
            vision_inter_dim=v['intermediate_size'],
            vision_patch_size=v['patch_size'],
            vision_rope_theta=v['rope_theta'],
            vision_downsample_ratio=v['downsample_ratio'],
            dim=t['hidden_size'],
        )
        self.vision = ViT(vision_args)
        self.aligner = Aligner(vision_args)
        for root in ('vision', 'aligner'):
            module = getattr(self, root)
            for name, parameter in module.named_parameters():
                path, attribute = name.rsplit('.', 1)
                self._bind(root + '.' + name, module.get_submodule(path), attribute, root)
        for key in ('image_start', 'image_end', 'image_newline'):
            self.register_parameter(key, nn.Parameter(torch.zeros(t['hidden_size'])))
            self._bind(key, self, key, 'image_delimiter')
        self.mtp = DeferredModule('DSpark')
        for root, key in self._archive_keys(t, v):
            if root != 'mtp':
                continue
            module = getattr(self, root)
            owner = module.leaf_owner(key[len(root) + 1 :])
            self.archival_bindings[key] = TensorBinding(key, owner, None, 'archival')
        self.validate_parameter_bindings()

    def _bind(self, key, owner, attribute, role, head_count=None, encoding=None):
        if key in self.tensor_bindings:
            raise ValueError(f'duplicate binding: {key}')
        self.tensor_bindings[key] = TensorBinding(key, owner, attribute, role, head_count, encoding)
        if encoding in ('I8', 'F8_E4M3') and role != 'engram_table':
            scale = key[:-6] + 'scale'
            self.tensor_bindings[scale] = TensorBinding(
                scale, owner, attribute, 'scale', encoding='F8_E8M0'
            )

    @staticmethod
    def _expert(t, quantized, shared):
        dim = t['hidden_size']
        width = t['moe_intermediate_size'] * (t['n_shared_experts'] if shared else 1)

        def projection(a, b):
            return Linear(a, b, fp8=quantized) if shared else FP4Linear(a, b, quantized=quantized)

        return SwiGLUExpert(
            projection(dim, width),
            projection(width, dim),
            projection(dim, width),
            swiglu_limit=t['swiglu_limit'],
        )

    def _bind_expert(self, prefix, expert, role, encoding):
        for name in ('w1', 'w2', 'w3'):
            self._bind(
                prefix + '.' + name + '.weight',
                getattr(expert, name),
                'weight',
                role,
                encoding=encoding,
            )

    def _bind_attention(self, prefix, a, t):
        for name in ('wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b'):
            heads = t['num_attention_heads'] if name == 'wq_b' else None
            self._bind(
                prefix + '.' + name + '.weight', getattr(a, name), 'weight', name, heads, 'F8_E4M3'
            )
        for name in ('q_norm', 'kv_norm'):
            self._bind(prefix + '.' + name + '.weight', getattr(a, name), 'weight', 'norm')
        self._bind(prefix + '.attn_sink', a, 'attn_sink', 'attention_sink')
        if a.compressor is not None:
            for name in ('wkv', 'norm', 'wgate'):
                if hasattr(a.compressor, name):
                    self._bind(
                        prefix + '.compressor.' + name + '.weight',
                        getattr(a.compressor, name),
                        'weight',
                        'compressor',
                    )
        if a.indexer is not None:
            for name in ('wq_b', 'weights_proj', 'wk', 'k_norm'):
                if hasattr(a.indexer, name):
                    self._bind(
                        prefix + '.indexer.' + name + '.weight',
                        getattr(a.indexer, name),
                        'weight',
                        'indexer',
                        t['index_n_heads'] if name == 'wq_b' else None,
                        'F8_E4M3' if name == 'wq_b' else None,
                    )

    @staticmethod
    def _archive_keys(t, v):
        for index in range(v['num_hidden_layers']):
            for suffix in (
                'attn.wo.bias',
                'attn.wo.weight',
                'attn.wqkv.bias',
                'attn.wqkv.weight',
                'mlp.w1.weight',
                'mlp.w2.weight',
                'norm1.weight',
                'norm2.weight',
            ):
                yield 'vision', f'vision.blocks.{index}.{suffix}'
        for suffix in ('norm.weight', 'patch_embed.proj.bias', 'patch_embed.proj.weight'):
            yield 'vision', 'vision.' + suffix
        for name in ('w1', 'w2'):
            for suffix in ('weight', 'bias'):
                yield 'aligner', f'aligner.{name}.{suffix}'
        for index in range(t['num_nextn_predict_layers']):
            prefix = f'mtp.{index}.'
            for suffix in (
                'attn.attn_sink',
                'attn.kv_norm.weight',
                'attn.q_norm.weight',
                'attn_norm.weight',
                'ffn_norm.weight',
                'ffn.gate.weight',
                'ffn.gate.bias',
                'ffn.gate.bias_vl',
            ):
                yield 'mtp', prefix + suffix
            for name in ('wkv', 'wo_a', 'wo_b', 'wq_a', 'wq_b'):
                for suffix in ('weight', 'scale'):
                    yield 'mtp', prefix + f'attn.{name}.{suffix}'
            for side in ('attn', 'ffn'):
                for suffix in ('fn', 'base', 'scale'):
                    yield 'mtp', prefix + f'hc_{side}_{suffix}'
            experts = [f'experts.{i}' for i in range(t['dspark_n_routed_experts'])]
            if t['n_shared_experts']:
                experts.append('shared_experts')
            for expert in experts:
                for name in ('w1', 'w2', 'w3'):
                    for suffix in ('weight', 'scale'):
                        yield 'mtp', prefix + f'ffn.{expert}.{name}.{suffix}'
            if index == 0:
                for suffix in ('main_norm.weight', 'main_proj.weight', 'main_proj.scale'):
                    yield 'mtp', prefix + suffix
            if index == t['num_nextn_predict_layers'] - 1:
                for suffix in (
                    'confidence_head.proj.weight',
                    'markov_head.embed.weight',
                    'markov_head.head.weight',
                    'norm.weight',
                ):
                    yield 'mtp', prefix + suffix

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

    def _sequence(self, hidden, pre, *, input_ids, image_mask=None):
        hashes = None
        if self.engram_layer_ids:
            if self.engram_hash is None:
                raise ValueError('Engram execution requires an explicit tokenizer token_map')
            hashes = self.engram_hash(input_ids, None if image_mask is None else ~image_mask)
        state = AttentionState()
        for index, layer in enumerate(self.layers):
            if layer.engram is not None:
                hidden = layer.engram(
                    hidden,
                    hashes[:, :, self.engram_layer_ids.index(index)],
                    None if image_mask is None else ~image_mask,
                )
            hidden, pre, state = layer.forward_with_state(
                hidden, pre, state, ffn_kwargs={'image_mask': image_mask}
            )
        return hidden, pre

    def forward(self, input_ids, *, cu_seqlens=None, images=None, token_types=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.int64 or not input_ids.shape[1]:
            raise ValueError('Expected nonempty int64 input_ids [B,S]')
        embeddings = self.embed(input_ids)
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
                            raise ValueError('Image span crosses a packed sequence boundary')
                    expected_types[batch, img.start : img.start + img.types.numel()] = img.types.to(
                        input_ids.device
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
        if hasattr(self, 'residual_dtype'):
            embeddings = embeddings.to(self.residual_dtype)
        hidden, pre = expand_hc(embeddings, self.hc_mult)
        if cu_seqlens is None:
            hidden, pre = self._sequence(hidden, pre, input_ids=input_ids, image_mask=image_mask)
        else:
            hidden, pre = packed_forward(
                self._sequence, hidden, pre, cu_seqlens, input_ids=input_ids, image_mask=image_mask
            )
        hidden = self.norm(contract_hc(hidden, pre))
        return {'logits': F.linear(hidden.float(), self.head.weight.float())}

    def encode_image(self, patches, n_vit_h, n_vit_w):
        weight = self.vision.patch_embed.proj.weight
        patches = patches.to(device=weight.device, dtype=weight.dtype)
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h):
        features = [
            [self.encode_image(img.patches, img.n_vit_h, img.n_vit_w) for img in sample or ()]
            for sample in images
        ]
        return merge_image_embeddings(
            h, images, features, self.image_start, self.image_end, self.image_newline
        )

    def forward_spec(self, *args, **kwargs):
        raise NotImplementedError('DSpark execution is not implemented')

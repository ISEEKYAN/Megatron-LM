# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit Qwen3.8 text configuration; preserve unsupported modality metadata."""
from dataclasses import dataclass, field, fields


@dataclass
class Qwen3_8_FlashNextTextConfig:
    intermediate_size: int = 5632
    hidden_act: str = 'silu'
    initializer_range: float = 0.02
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_cache: bool = True
    mamba_ssm_dtype: str = 'float32'
    decoder_sparse_step: int = 1
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    mlp_only_layers: list | None = None
    rope_scaling: dict | None = None
    rope_parameters: dict | None = None
    mtp: dict | None = None
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int = 248044
    tie_word_embeddings: bool = False
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    hc_count: int = 4
    hc_lowrank: int = 320
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_compress_ratio: int = 4
    indexer_budget: int = 2048
    ngram_vocab_size_base: int = 20000000
    make_ngram_vocab_size_divisible_by: int = 128
    max_position_embeddings: int = 262144
    router_aux_loss_coef: float = 0.001
    ngram_size: int = 3
    heads_per_ngram: int = 8
    split_ngram_parts: int = 128
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ple_layer_ids: list[int] = field(default_factory=lambda: [2])
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    output_gate_type: str = 'sigmoid'
    eos_token_id: int = 248044
    dtype: str = 'bfloat16'
    vision_config: dict = field(default_factory=dict)
    mtp_config: dict = field(default_factory=dict)

    @classmethod
    def from_hf_dict(cls, source):
        text = source.get('text_config', source)
        aliases = {
            'qwen4_exp',
            'qwen4_exp_text',
            'qwen3_8_flash_next',
            'qwen3_8_flash_next_text',
        }
        if (
            source.get('model_type') not in aliases
            or text.get('model_type', source.get('model_type')) not in aliases
        ):
            raise ValueError('QWEN38_MODEL_ALIAS')
        values = {f.name: text[f.name] for f in fields(cls) if f.name in text}
        rope = text.get('rope_parameters') or {}
        for key in ('rope_theta', 'partial_rotary_factor', 'mrope_section'):
            if key in rope:
                values[key] = rope[key]
        values['vision_config'] = dict(source.get('vision_config', {}))
        values['mtp_config'] = dict(text.get('mtp', {}))
        if 'mtp_num_hidden_layers' in text:
            values['mtp_config'].setdefault(
                'num_hidden_layers', text['mtp_num_hidden_layers']
            )
        return cls(**values)

    def __post_init__(self):
        if self.full_attention_interval < 1 or self.num_hidden_layers < 1:
            raise ValueError('QWEN38_LAYER_COUNT')
        if not self.layer_types:
            self.layer_types = [
                (
                    'full_attention'
                    if (i + 1) % self.full_attention_interval == 0
                    else 'linear_attention'
                )
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers or set(self.layer_types) - {
            'linear_attention',
            'full_attention',
        }:
            raise ValueError('QWEN38_LAYER_TYPES')
        if self.hc_count < 2 or self.hc_lowrank < 1 or self.hidden_size < 1:
            raise ValueError('QWEN38_HC')
        if (
            self.indexer_kv_heads != 1
            or self.indexer_compress_ratio < 2
            or self.indexer_budget <= 0
            or self.indexer_budget % self.indexer_compress_ratio
        ):
            raise ValueError('QWEN38_INDEXER')
        if (
            self.output_gate_type != 'sigmoid'
            or self.ngram_size != 3
            or self.heads_per_ngram != 8
            or self.ngram_vocab_size_base != 20000000
            or self.make_ngram_vocab_size_divisible_by != 128
        ):
            raise ValueError('QWEN38_RELEASE_SEMANTICS')
        if any(i < 1 or i > self.num_hidden_layers for i in self.ple_layer_ids):
            raise ValueError('QWEN38_PLE_LAYER')


@dataclass
class Qwen3_8_FlashNextVisionConfig:
    depth: int = 27
    hidden_act: str = 'gelu_pytorch_tanh'
    hidden_size: int = 1152
    in_channels: int = 3
    initializer_range: float = 0.02
    intermediate_size: int = 4304
    num_heads: int = 16
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2560
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    deepstack_visual_indexes: list | None = None


@dataclass
class Qwen3_8_FlashNextConfig:
    text_config: Qwen3_8_FlashNextTextConfig = field(
        default_factory=Qwen3_8_FlashNextTextConfig
    )
    vision_config: Qwen3_8_FlashNextVisionConfig = field(
        default_factory=Qwen3_8_FlashNextVisionConfig
    )
    model_type: str = 'qwen3_8_flash_next'

    @classmethod
    def from_hf_dict(cls, source):
        vision = source.get('vision_config', {})
        return cls(
            Qwen3_8_FlashNextTextConfig.from_hf_dict(source),
            Qwen3_8_FlashNextVisionConfig(
                **{
                    f.name: vision[f.name]
                    for f in fields(Qwen3_8_FlashNextVisionConfig)
                    if f.name in vision
                }
            ),
        )

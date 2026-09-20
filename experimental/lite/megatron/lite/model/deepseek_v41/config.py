# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release policy and one topology table, including inactive archival metadata."""
from copy import deepcopy

from megatron.lite.primitive.config import load_hf_config_dict
from megatron.lite.primitive.config_fields import (
    nested_fields,
    project_fields,
    require_fields,
)

from .topology import TopologySpec, build_topology


class DeepseekV41Config:
    def __init__(self, release):
        self._release = deepcopy(release)
        text, _, quantization = nested_fields(
            self._release,
            'deepseek_v41',
            'text_config vision_config quantization_config',
        )
        self.topology = build_topology(
            TopologySpec(
                **{
                    k: tuple(text[k]) if isinstance(text[k], list) else text[k]
                    for k in TopologySpec.__dataclass_fields__
                }
            )
        )
        require_fields(
            text,
            dict(
                num_key_value_heads=1,
                hidden_act='silu',
                attention_bias=False,
                attention_dropout=0.0,
                tie_word_embeddings=False,
                norm_topk_prob=True,
                topk_method='noaux_tc',
            ),
        )
        if text['rope_scaling']['rope_type'] != 'yarn':
            raise ValueError('Only YaRN rope_scaling is supported')
        if text['num_nextn_predict_layers'] != 3:
            raise ValueError('The archival DSpark contract requires three layers')
        if text['engram_head_dim'] % 32:
            raise ValueError('Engram head width must be divisible by 32')
        if not 0 < text['num_experts_per_tok'] <= text['n_routed_experts']:
            raise ValueError('Invalid num_experts_per_tok')
        if text['sliding_window'] <= 0 or text['index_topk'] <= 0:
            raise ValueError('Attention windows and Top-K must be positive')
        if quantization != dict(
            quant_method='fp8',
            activation_scheme='dynamic',
            weight_block_size=[32, 32],
            scale_fmt='ue8m0',
            expert_dtype='fp4',
        ):
            raise ValueError('Unsupported quantization_config')

    @classmethod
    def from_hf(cls, path):
        return cls(load_hf_config_dict(path))

    @property
    def hidden_size(self):
        return self._release['text_config']['hidden_size']

    def to_hf_dict(self):
        return deepcopy(self._release)

    def attention_config(
        self, *, linear_fp8=True, main_qat=True, index_qat=True, swa_fp8=True
    ):
        from megatron.lite.primitive.modules.attention.csa import (
            CrossLayerAttentionConfig,
        )

        text = self._release['text_config']
        return CrossLayerAttentionConfig(
            **project_fields(
                text,
                'dim=hidden_size heads=num_attention_heads head_dim rope_dim=qk_rope_head_dim '
                'q_rank=q_lora_rank o_rank=o_lora_rank groups=o_groups index_heads=index_n_heads '
                'index_dim=index_head_dim topk=index_topk window=sliding_window '
                'candidate_blocks=candidate_topk_blocks block_size=candidate_block_size '
                'eps=rms_norm_eps rope_theta compress_rope_theta',
            ),
            **project_fields(
                text['rope_scaling'],
                'original_length=original_max_position_embeddings factor beta_fast beta_slow',
            ),
            linear_fp8=linear_fp8,
            main_qat=main_qat,
            index_qat=index_qat,
            swa_fp8=swa_fp8
        )

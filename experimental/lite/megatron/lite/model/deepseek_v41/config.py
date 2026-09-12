# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Nested release configuration and explicit single-rank attention bindings."""

from copy import deepcopy
import json
from pathlib import Path


class DeepseekV41Config:
    """Keep the complete release config, including inactive DSpark metadata.

    The current CSA2 implementation supports the released backbone topology.
    Reject changes to that topology instead of accepting ineffective settings.
    Reduced numerical dimensions are supported independently of topology.
    """

    def __init__(self, release):
        self._release = deepcopy(release)
        if self._release.get('model_type') != 'deepseek_v41':
            raise ValueError('Expected model_type=deepseek_v41')
        for section in ('text_config', 'vision_config', 'quantization_config'):
            if not isinstance(self._release.get(section), dict):
                raise ValueError(f'Missing nested {section}')
        text = self._release['text_config']
        topology = {
            'num_hidden_layers': 40,
            'kv_source_layer_ids': [2, 8, 14, 20],
            'index_source_layer_ids': [2, 8, 14, 20, 24, 28, 32, 36],
            'candidate_source_layer_id': 20,
            'compress_ratios': [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        }
        for key, expected in topology.items():
            if text.get(key) != expected:
                raise ValueError(f'Unsupported {key}: expected {expected}')

        fixed = {
            'num_key_value_heads': 1,
            'hidden_act': 'silu',
            'attention_bias': False,
            'attention_dropout': 0.0,
            'tie_word_embeddings': False,
            'norm_topk_prob': True,
            'topk_method': 'noaux_tc',
        }
        for key, expected in fixed.items():
            if text.get(key) != expected:
                raise ValueError(f'Unsupported {key}: expected {expected}')
        if text['rope_scaling']['rope_type'] != 'yarn':
            raise ValueError('Only YaRN rope_scaling is supported')
        if text['num_nextn_predict_layers'] != 3:
            raise ValueError('The archival DSpark contract requires three layers')
        layers = text['engram_layer_ids']
        if len(layers) != len(set(layers)) or any(i < 0 or i >= 40 for i in layers):
            raise ValueError('Invalid engram_layer_ids')
        if len(layers) != len(text['engram_num_embeddings']):
            raise ValueError('Engram row counts must match layers')
        if text['engram_head_dim'] % 32:
            raise ValueError('Engram head width must be divisible by 32')
        if not 0 < text['num_experts_per_tok'] <= text['n_routed_experts']:
            raise ValueError('Invalid num_experts_per_tok')
        if text['sliding_window'] <= 0 or text['index_topk'] <= 0:
            raise ValueError('Attention windows and Top-K must be positive')
        quantization = self._release['quantization_config']
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
        path = Path(path)
        if path.is_dir():
            path = path / 'config.json'
        return cls._from_hf_dict(json.loads(path.read_text()))

    @classmethod
    def _from_hf_dict(cls, hf):
        return cls(hf)

    def to_hf_dict(self):
        return deepcopy(self._release)

    def attention_config(self, *, linear_fp8=True, main_qat=True, index_qat=True, swa_fp8=True):
        from .lite.attention import CSA2Config

        text = self._release['text_config']
        rope = text['rope_scaling']
        return CSA2Config(
            dim=text['hidden_size'],
            heads=text['num_attention_heads'],
            head_dim=text['head_dim'],
            rope_dim=text['qk_rope_head_dim'],
            q_rank=text['q_lora_rank'],
            o_rank=text['o_lora_rank'],
            groups=text['o_groups'],
            index_heads=text['index_n_heads'],
            index_dim=text['index_head_dim'],
            topk=text['index_topk'],
            window=text['sliding_window'],
            candidate_blocks=text['candidate_topk_blocks'],
            block_size=text['candidate_block_size'],
            eps=text['rms_norm_eps'],
            rope_theta=text['rope_theta'],
            compress_rope_theta=text['compress_rope_theta'],
            original_length=rope['original_max_position_embeddings'],
            factor=rope['factor'],
            beta_fast=rope['beta_fast'],
            beta_slow=rope['beta_slow'],
            linear_fp8=linear_fp8,
            main_qat=main_qat,
            index_qat=index_qat,
            swa_fp8=swa_fp8,
        )

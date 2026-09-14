# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Key-level mapping plan; unsupported keys remain explicit, never silently dropped."""
import re

HF_REVISION = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
HF_INDEX = f'https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{HF_REVISION}/model.safetensors.index.json'


def checkpoint_plan(index, config):
    keys = index['weight_map']
    result = {'source': HF_INDEX, 'mapped': {}, 'vision': [], 'mtp': [], 'unknown': []}
    hc = {
        'hc_norm.weight',
        'input_mix_weight_down.weight',
        'input_mix_weight_up.weight',
        'block_inject_weight.weight',
    }
    gdn = {
        f'{n}.weight'
        for n in (
            'in_proj_qkv',
            'in_proj_z',
            'in_proj_b',
            'in_proj_a',
            'conv1d',
            'norm',
            'out_proj',
        )
    } | {'A_log', 'dt_bias'}
    qsa = {
        f'{n}.weight'
        for n in (
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'q_norm',
            'k_norm',
            'indexer.index_qk_proj',
            'indexer.q_layernorm',
            'indexer.k_layernorm',
        )
    }
    moe = {
        'experts.gate_up_proj',
        'experts.down_proj',
        'gate.weight',
        'shared_expert_gate.weight',
    } | {f'shared_expert.{n}_proj.weight' for n in ('gate', 'up', 'down')}
    ple = {
        f'{n}.weight'
        for n in (
            'key_proj',
            'value_proj',
            'norm_key',
            'norm_query',
            'norm_conv',
            'conv1d',
        )
    } | {
        f'ple_embedding.{n}'
        for n in ('layer_multipliers', 'ngram_heads_vocab_sizes', 'ngram_heads_offsets')
    }
    for key in sorted(keys):
        if key.startswith('model.visual.'):
            result['vision'].append(key)
            continue
        if key.startswith('mtp.'):
            result['mtp'].append(key)
            continue
        local = key.removeprefix('model.language_model.')
        accepted = local in {'embed_tokens.weight', 'lm_head.weight'}
        if local.startswith('hyper_connection_mixer.'):
            accepted = local.removeprefix('hyper_connection_mixer.') in hc - {
                'block_inject_weight.weight'
            }
        match = re.fullmatch(r'layers\.(\d+)\.([^.]+)\.(.+)', local)
        if match:
            layer, part, name = match.groups()
            layer = int(layer)
            if layer < config.num_hidden_layers:
                allowed = {
                    'attn_hyper_connection': hc,
                    'mlp_hyper_connection': hc,
                    'mlp': moe,
                }
                allowed[
                    (
                        'linear_attn'
                        if config.layer_types[layer] == 'linear_attention'
                        else 'self_attn'
                    )
                ] = (gdn if config.layer_types[layer] == 'linear_attention' else qsa)
                if layer + 1 in config.ple_layer_ids:
                    allowed['ple'] = ple | {
                        f'ple_embedding.ngram_embedding.shard_{j}.weight'
                        for j in range(config.split_ngram_parts)
                    }
                accepted = name in allowed.get(part, set())
        if accepted:
            result['mapped'][key] = local
        else:
            result['unknown'].append(key)
    result['counts'] = {
        k: len(result[k]) for k in ('mapped', 'vision', 'mtp', 'unknown')
    }
    return result


def owner_shard_views(weight, start, end, *, total_rows=320001536, parts=128):
    if (
        parts < 1
        or weight.ndim != 2
        or total_rows % parts
        or not 0 <= start <= end <= total_rows
        or weight.shape[0] != end - start
    ):
        raise ValueError('PLE_OWNER_RANGE')
    width = total_rows // parts
    return {
        j: (
            max(start, j * width) - j * width,
            weight[max(start, j * width) - start : min(end, (j + 1) * width) - start],
        )
        for j in range(parts)
        if max(start, j * width) < min(end, (j + 1) * width)
    }

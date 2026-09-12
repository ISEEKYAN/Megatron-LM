import json
import math
import os
import struct
import urllib.request
from pathlib import Path

import pytest
import torch
from megatron.lite.model.qwen3_8_flash_next.checkpoint import (
    HF_INDEX,
    checkpoint_plan,
    owner_shard_views,
)
from megatron.lite.model.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextTextConfig as Qwen38Config,
)
from megatron.lite.model.qwen3_8_flash_next.engram import (
    Qwen3_8_FlashNextNGramEmbedding as NgramEmbedding,
)


def test_config_alias_and_rope():
    c = Qwen38Config.from_hf_dict(
        {
            'model_type': 'qwen4_exp',
            'text_config': {
                'model_type': 'qwen4_exp_text',
                'ple_layer_ids': [2],
                'rope_parameters': {'rope_theta': 123.0},
            },
            'vision_config': {'depth': 27},
        }
    )
    assert c.rope_theta == 123.0 and c.vision_config == {
        'depth': 27
    }, 'CONFIG_NESTED_MAPPING'
    assert c.layer_types.count('full_attention') == 12 and c.ple_layer_ids == [
        2
    ], 'CONFIG_RELEASE_LAYOUT'


@pytest.mark.parametrize(
    'kwargs,tag',
    [
        ({'full_attention_interval': 0}, 'QWEN38_LAYER_COUNT'),
        ({'layer_types': ['wrong']}, 'QWEN38_LAYER_TYPES'),
        ({'hc_count': 1}, 'QWEN38_HC'),
        ({'indexer_budget': 3}, 'QWEN38_INDEXER'),
        ({'output_gate_type': 'silu'}, 'QWEN38_RELEASE_SEMANTICS'),
        ({'ple_layer_ids': [0]}, 'QWEN38_PLE_LAYER'),
    ],
)
def test_config_guards(kwargs, tag):
    with expect_guard(tag):
        Qwen38Config(**kwargs)


def test_config_rejects_other_family():
    with expect_guard('QWEN38_MODEL_ALIAS'):
        Qwen38Config.from_hf_dict({'model_type': 'qwen3_5'})


def test_coverage_does_not_hide_unknown_or_modalities():
    prefix = 'model.language_model.layers.'
    keys = [
        prefix + '1.ple.ple_embedding.ngram_embedding.shard_127.weight',
        prefix + '0.self_attn.q_proj.weight',
        prefix + '0.mlp.experts.gate_up_proj',
        'model.visual.patch_embed.proj.weight',
        'mtp.fc_hidden.weight',
    ]
    p = checkpoint_plan(
        {'weight_map': dict.fromkeys(keys, 'source')}, Qwen38Config(ple_layer_ids=[2])
    )
    assert p['counts'] == dict(
        mapped=2, vision=1, mtp=1, unknown=1
    ), 'CHECKPOINT_COVERAGE_COUNTS'
    assert (
        p['unknown'] == [keys[1]] and p['vision'] == [keys[3]] and p['mtp'] == [keys[4]]
    ), 'CHECKPOINT_EXACT_GAPS'


def test_owner_views_intersect_without_copy():
    x = torch.arange(12).reshape(6, 2)
    views = owner_shard_views(x, 3, 9, total_rows=12, parts=3)
    assert [(j, offset, len(v)) for j, (offset, v) in views.items()] == [
        (0, 3, 1),
        (1, 0, 4),
        (2, 0, 1),
    ], 'OWNER_INTERSECTIONS'
    views[1][1][0, 0] = 99
    assert x[1, 0] == 99, 'OWNER_VIEW_ALIAS'
    with expect_guard('PLE_OWNER_RANGE'):
        owner_shard_views(x, 3, 8, total_rows=12, parts=3)


from contextlib import contextmanager


@contextmanager
def expect_guard(tag):
    try:
        yield
    except Exception as error:
        assert tag in str(error), f"GUARD_{tag}: wrong failure {error!r}"
    else:
        assert False, f"GUARD_{tag}: missing rejection"


@pytest.fixture(scope='module')
def hf_metadata():
    # Opt in with MLITE_QWEN38_HF_METADATA=<cache directory>; never fetch weights.
    cache = os.environ.get('MLITE_QWEN38_HF_METADATA')
    if not cache:
        pytest.skip('Set MLITE_QWEN38_HF_METADATA to fetch/cache real HF metadata')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(name, span=None):
        path = cache / (name + (f'.{span[0]}-{span[1]}' if span else ''))
        limit = span[1] - span[0] + 1 if span else 1_000_000
        if not path.exists():
            headers = {'Range': f'bytes={span[0]}-{span[1]}'} if span else {}
            request = urllib.request.Request(
                HF_INDEX.rsplit('/', 1)[0] + '/' + name, headers=headers
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                if span:
                    assert response.status == 206, 'HF_RANGE_REQUIRED'
                data = response.read(limit + 1)
            assert len(data) <= limit, 'HF_METADATA_SIZE_LIMIT'
            path.write_bytes(data)
        data = path.read_bytes()
        assert len(data) == limit if span else len(data) <= limit
        return data

    def header(shard):
        name = f'model-{shard:05d}-of-00131.safetensors'
        size = struct.unpack('<Q', fetch(name, (0, 7)))[0]
        assert 0 < size < 1_000_000, 'HF_HEADER_SIZE'
        result = json.loads(fetch(name, (8, size + 7)))
        result.pop('__metadata__', None)
        return name, size + 8, result

    return fetch, header


def test_real_hf_index_and_first_shard(hf_metadata):
    from megatron.lite.model.qwen3_8_flash_next.math import (
        Qwen3_8_FlashNextHyperConnection,
    )

    fetch, header = hf_metadata
    index = json.loads(fetch('model.safetensors.index.json'))
    c = Qwen38Config.from_hf_dict(json.loads(fetch('config.json')))
    plan = checkpoint_plan(index, c)
    assert plan['counts'] == dict(mapped=1294, vision=333, mtp=31, unknown=0)
    assert len(index['weight_map']) == 1658
    name, _, tensors = header(1)
    assert set(tensors) == {k for k, v in index['weight_map'].items() if v == name}
    assert len(tensors) == 349
    expected = {}
    with torch.device('meta'):
        for prefix, write in [
            ('hyper_connection_mixer', False),
            ('layers.0.attn_hyper_connection', True),
        ]:
            module = Qwen3_8_FlashNextHyperConnection(
                c.hidden_size, c.hc_count, c.hc_lowrank, write=write
            ).bfloat16()
            for k, v in module.state_dict().items():
                assert v.dtype == torch.bfloat16
                expected[prefix + '.' + k] = list(v.shape)
    assert c.dtype == 'bfloat16'
    h = c.hidden_size
    nk, dk = c.linear_num_key_heads, c.linear_key_head_dim
    nv, dv = c.linear_num_value_heads, c.linear_value_head_dim
    channels = 2 * nk * dk + nv * dv
    gdn = {
        'A_log': [nv],
        'dt_bias': [nv],
        'conv1d.weight': [channels, 1, c.linear_conv_kernel_dim],
        'in_proj_a.weight': [nv, h],
        'in_proj_b.weight': [nv, h],
        'in_proj_qkv.weight': [channels, h],
        'in_proj_z.weight': [nv * dv, h],
        'norm.weight': [dv],
        'out_proj.weight': [h, nv * dv],
    }
    expected.update({'layers.0.linear_attn.' + k: v for k, v in gdn.items()})
    mapped = set()
    for key, tensor in tensors.items():
        assert tensor['dtype'] == 'BF16', key
        a, b = tensor['data_offsets']
        assert b - a == 2 * math.prod(tensor['shape']), key
        if key in plan['vision']:
            assert key not in plan['mapped'], key  # No native vision target exists.
        else:
            target = plan['mapped'][key]
            assert target == key.removeprefix('model.language_model.'), key
            assert tensor['shape'] == expected[target], key
            mapped.add(target)
    assert mapped == set(expected) and len(mapped) == 16


def test_real_hf_ple_buffers(hf_metadata):
    fetch, header = hf_metadata
    index = json.loads(fetch('model.safetensors.index.json'))
    config = Qwen38Config.from_hf_dict(json.loads(fetch('config.json')))
    plan = checkpoint_plan(index, config)
    state = NgramEmbedding(None).state_dict()
    for buffer, shard, count in [
        ('layer_multipliers', 5, 3),
        ('ngram_heads_offsets', 37, 16),
        ('ngram_heads_vocab_sizes', 37, 16),
    ]:
        key = f'model.language_model.layers.1.ple.ple_embedding.{buffer}'
        name, offset, tensors = header(shard)
        assert index['weight_map'][key] == name
        assert plan['mapped'][key] == f'layers.1.ple.ple_embedding.{buffer}'
        tensor = tensors[key]
        assert tensor['dtype'] == 'I64' and tensor['shape'] == [count], key
        a, b = tensor['data_offsets']
        assert b - a == count * 8, key
        values = struct.unpack(
            '<' + 'q' * count, fetch(name, (offset + a, offset + b - 1))
        )
        assert state[buffer].dtype == torch.int64 and list(state[buffer].shape) == [
            count
        ], key
        assert state[buffer].tolist() == list(values), key

import pytest
import torch
from megatron.lite.model.qwen3_8_flash_next.checkpoint import (
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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The official/graph recipe rejects silently reused release configurations."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'ds41_recipe_contract',
    ROOT / 'examples/verl/scripts/ds41_w4a8_proxy/recipe_contract.py',
)
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


def test_official_graph_uses_native_precision_and_breakable_capture():
    result = CONTRACT.rollout_settings(
        {'text_config': {'expert_activation_group_size': 32, 'wo_a_dtype': 'bf16'}},
        {'DS41_PRECISION': 'official', 'DS41_CUDAGRAPH': 'full_and_piecewise'},
    )
    assert result['enforce_eager'] is False and result['breakable']
    assert result['hf_overrides'] == {
        'head_dtype': 'float32',
        'expert_activation_group_size': 32,
        'wo_a_dtype': 'bf16',
    }
    assert result['compilation_config'] == {
        'mode': 0,
        'cudagraph_mode': 'FULL_AND_PIECEWISE',
        'max_cudagraph_capture_size': 16,
    }


def test_official_recipe_rejects_old_prepared_prefix():
    with pytest.raises(ValueError, match='prepared g32/BF16'):
        CONTRACT.rollout_settings({}, {'DS41_PRECISION': 'official'})


@pytest.mark.parametrize(
    'env',
    [
        {'DS41_CUDAGRAPH': 'full'},
        {'DS41_CUDAGRAPH': 'full_and_piecewise', 'DS41_CUDAGRAPH_MAX_CAPTURE': '32'},
        {'DS41_PRECISION': 'guess'},
    ],
)
def test_unsupported_recipe_settings_fail_closed(env):
    with pytest.raises(ValueError):
        CONTRACT.rollout_settings({}, env)


def test_release_defaults_keep_eager_and_head_only_override():
    result = CONTRACT.rollout_settings({}, {})
    assert result['enforce_eager'] and not result['breakable']
    assert result['compilation_config'] == {'mode': 0, 'cudagraph_mode': 'NONE'}
    assert result['hf_overrides'] == {'head_dtype': 'float32'}

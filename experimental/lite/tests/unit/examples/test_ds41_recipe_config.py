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


@pytest.mark.parametrize('layers', [2, 4, 6, 9])
def test_hsg_submit_preserves_variant_and_resource_overrides(tmp_path, layers):
    """The delivered entry submits both stages with the requested resources."""
    import os
    import subprocess

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'paths.env').write_text(
        'DS41_RELEASE=/weights\nDS41_DATA_DIR=/data\nDS41_DATA_FILE=input.parquet\n'
    )
    executable = tmp_path / 'bin'
    executable.mkdir()
    call_log = tmp_path / 'calls'
    stub = executable / 'sbatch'
    stub.write_text(
        '#!/bin/bash\n'
        'printf "%s\\n" "$DS41_LAYERS" "$@" >> "$CALL_LOG"\n'
        'echo 12345\n'
    )
    stub.chmod(0o755)
    env = {
        **os.environ,
        'PATH': str(executable) + ':' + os.environ['PATH'],
        'CALL_LOG': str(call_log),
        'DS41_BUNDLE': str(bundle),
        'DS41_SLURM_ACCOUNT': 'test-account',
        'WANDB_API_KEY': 'test-placeholder',
        'DS41_PARAMETER_COUNTS': '123',
        'DS41_CPU_QOS': 'cpu-short',
        'DS41_CPU_TIME': '00:30:00',
        'DS41_GPU_TIME': '01:30:00',
        'DS41_CPU_CPUS': '24',
        'DS41_GPU_CPUS': '64',
        'DS41_CPU_MEM': '0',
        'DS41_GPU_MEM': '0',
        'DS41_GPU_QOS': '',
    }
    subprocess.run(
        [
            'bash',
            str(ROOT / 'examples/verl/scripts/ds41_w4a8_proxy/hsg/submit.sh'),
            '--layers',
            str(layers),
            str(tmp_path / 'output'),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = call_log.read_text().splitlines()
    assert calls.count(str(layers)) == 2
    assert '--time=00:30:00' in calls and '--time=01:30:00' in calls
    assert '--cpus-per-task=24' in calls and '--cpus-per-task=64' in calls
    assert calls.count('--mem=0') == 2
    assert '--dependency=afterok:12345' in calls
    assert calls.count('--qos') == 1 and 'cpu-short' in calls
    assert calls.count('--nodes=2') == 2

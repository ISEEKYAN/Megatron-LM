# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The official/graph recipe rejects silently reused release configurations."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'ds41_recipe_contract',
    ROOT / 'examples/verl/scripts/ds41_w4a8_proxy/recipe_contract.py',
)
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)


@pytest.mark.parametrize(
    'failure',
    [
        None,
        'rank_failed',
        'rank_missing',
        'rank_duplicate',
        'replay',
        'replay_config',
        'replay_missing',
        'pytest_failed',
    ],
)
def test_cpu_receipt_cli_rejects_failed_or_incomplete_ranks(tmp_path, failure):
    """A successful process exit cannot hide a failed rank or enabled replay."""
    import subprocess
    import sys

    config, output = tmp_path / 'config', tmp_path / 'output'
    config.mkdir()
    output.mkdir()
    result = dict(passed=True, collect_only=True, pytest_exit=0)
    if failure == 'pytest_failed':
        result['passed'] = False
    (output / 'result.json').write_text(json.dumps(result))
    if failure != 'replay_missing':
        (config / 'vllm-cli.json').write_text(
            json.dumps({'swa_bounded_replay': failure == 'replay_config'})
        )
    for prefix in ('model-counts', 'checkpoint-headers'):
        for rank in range(2):
            if failure == 'rank_missing' and rank == 1:
                continue
            row = dict(passed=True, global_rank=rank)
            if failure == 'rank_failed' and rank == 1:
                row['ok'] = False
            if failure == 'rank_duplicate':
                row['global_rank'] = 0
            (config / f'{prefix}-rank-{rank}.json').write_text(json.dumps(row))
    if failure == 'replay':
        (output / 'events-123.jsonl').write_text(
            json.dumps({'kind': 'decoder_replay_enabled', 'window': 128}) + '\n'
        )
    command = [
        sys.executable,
        str(ROOT / 'examples/verl/scripts/ds41_w4a8_proxy/hsg/check-cpu-receipts.py'),
        '--config',
        str(config),
        '--output',
        str(output),
        '--world',
        '2',
        '--pp',
        '2',
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    receipt = json.loads((output / 'receipt-validation.json').read_text())
    assert completed.returncode == (0 if failure is None else 1)
    assert receipt['passed'] is (failure is None)


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


@pytest.mark.parametrize(
    ('layers', 'cpu_only'), [(2, False), (4, False), (6, False), (9, False), (40, True)]
)
@pytest.mark.parametrize('pp', [1, 2])
def test_hsg_submit_preserves_variant_and_resource_overrides(
    tmp_path, layers, cpu_only, pp
):
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
        'printf "TOPOLOGY=%s/%s/%s/%s/%s/%s\\n" "$DS41_PP" '
        '"$DS41_PIPELINE_SPLIT_LAYER" "$DS41_EP" "$DS41_CP" '
        '"$DS41_WORLD" "$DS41_PARAMETER_COUNTS" >> "$CALL_LOG"\n'
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
        'DS41_PARAMETER_COUNTS': '123' if pp == 1 else '101,202',
        'DS41_PP': str(pp),
        'DS41_PIPELINE_SPLIT_LAYER': '2',
        'DS41_EP': '4',
        'DS41_CP': '1',
        'DS41_WORLD': '8',
        'DS41_CPU_QOS': 'cpu-short',
        'DS41_CPU_TIME': '00:30:00',
        'DS41_GPU_TIME': '01:30:00',
        'DS41_CPU_CPUS': '24',
        'DS41_GPU_CPUS': '64',
        'DS41_CPU_MEM': '0',
        'DS41_GPU_MEM': '0',
        'DS41_GPU_QOS': '',
        'DS41_CPU_ONLY': '1' if cpu_only else '0',
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
    expected_jobs = 1 if cpu_only else 2
    counts = '123' if pp == 1 else '101,202'
    assert calls.count(f'TOPOLOGY={pp}/2/4/1/8/{counts}') == expected_jobs
    assert calls.count(str(layers)) == expected_jobs
    assert '--time=00:30:00' in calls and '--cpus-per-task=24' in calls
    assert calls.count('--mem=0') == expected_jobs
    assert calls.count('--qos') == 1 and 'cpu-short' in calls
    assert calls.count('--nodes=2') == expected_jobs
    if cpu_only:
        assert '--gpus-per-node=4' not in calls
        assert not any(arg.startswith('--dependency=') for arg in calls)
    else:
        assert '--time=01:30:00' in calls and '--cpus-per-task=64' in calls
        assert '--dependency=afterok:12345' in calls


@pytest.mark.parametrize('counts', ['665', '1,2,3', '0,2'])
def test_pp_verifier_rejects_non_stage_parameter_counts(monkeypatch, counts):
    spec = importlib.util.spec_from_file_location(
        'ds41_c8_counts', ROOT / 'tests/smoke/workflows/test_ds41_grpo_recipe.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('DS41_PP', '2')
    monkeypatch.setenv('DS41_PARAMETER_COUNTS', counts)
    with pytest.raises(ValueError, match='one positive count per stage'):
        module._parameter_counts(4)


def test_pp_verifier_uses_stage_counts_instead_of_full_model(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'ds41_c8_counts', ROOT / 'tests/smoke/workflows/test_ds41_grpo_recipe.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('DS41_PP', '2')
    monkeypatch.setenv('DS41_PARAMETER_COUNTS', '301,364')
    assert module._parameter_counts(4) == '301,364'
    monkeypatch.setenv('DS41_PP', '1')
    assert module._parameter_counts(4) == '665'


def test_p6_cpu_and_gpu_resolve_full_recompute(tmp_path, monkeypatch):
    """Both production entry paths feed full recompute into real Hydra config."""
    import os
    import subprocess
    import sys

    recipe = ROOT / 'examples/verl/scripts/ds41_w4a8_proxy'
    monkeypatch.delenv('DS41_RECOMPUTE', raising=False)
    for key in (
        'DS41_EP',
        'DS41_CP',
        'DS41_WORLD',
        'DS41_PP',
        'DS41_PIPELINE_SPLIT_LAYER',
    ):
        monkeypatch.delenv(key, raising=False)
    # Run the actual CPU environment branch without creating container paths.
    prefix = (
        (recipe / 'hsg/prepare-container.sh')
        .read_text()
        .split('export DS41_DATA=', 1)[0]
    )
    cpu_recompute = subprocess.check_output(
        ['bash', '-c', 'mkdir() { :; }\n' + prefix + '\nprintf %s "$DS41_RECOMPUTE"'],
        env={**os.environ, 'RECIPE_LAYERS': '6'},
        text=True,
    )
    spec = importlib.util.spec_from_file_location(
        'ds41_p6_production', ROOT / 'tests/smoke/workflows/test_ds41_grpo_recipe.py'
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    inputs = tmp_path / 'inputs'
    env = smoke.production_environment(6, 1, inputs)
    assert cpu_recompute == env['DS41_RECOMPUTE'] == 'full'
    pytest.importorskip(
        'verl', reason='real Hydra resolution requires the recipe VERL runtime'
    )
    from omegaconf import OmegaConf

    model = inputs / '6l/model'
    model.mkdir(parents=True)
    (model / 'config.json').write_text(
        json.dumps(
            {
                'text_config': {
                    'expert_activation_group_size': 32,
                    'wo_a_dtype': 'bf16',
                    'num_hidden_layers': 6,
                }
            }
        )
    )
    (model / 'token-map.json').write_text(json.dumps(list(range(64))))
    Path(env['DS41_DATA']).write_bytes(b'config-only input; no dataset read')
    env.update(
        DS41_OUTPUT=str(tmp_path / 'resolved'),
        DS41_PRECISION='official',
        DS41_CUDAGRAPH='full_and_piecewise',
    )
    subprocess.run(
        [sys.executable, str(recipe / 'command.py'), '--config-only'],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = OmegaConf.load(Path(env['DS41_OUTPUT']) / 'resolved.yaml')
    for role in ('actor', 'ref'):
        assert resolved.actor_rollout_ref[role].engine.impl_cfg.recompute == 'full'

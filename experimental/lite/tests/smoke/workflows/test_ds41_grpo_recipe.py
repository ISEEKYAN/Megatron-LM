# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise real two-update GRPO, publication, and new-process checkpoint resume.

This opt-in suite owns no scheduler or cluster lifecycle. Its caller provides an
idle eight-GPU Ray cluster, prepared release prefixes, data, and output storage.
CPU collection imports neither Ray nor the model and does not load weights.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.optional,
    pytest.mark.smoke,
    pytest.mark.gpus(8, min_architecture='blackwell'),
]
RECIPE = Path(__file__).resolve().parents[3] / 'examples/verl/scripts/ds41_w4a8_proxy'


def _idle_cluster(address):
    import ray

    ray.init(address=address, ignore_reinit_error=True)
    deadline = time.monotonic() + 90
    try:
        assert ray.cluster_resources().get('GPU') == 8
        while ray.available_resources().get('GPU', 0) != 8:
            assert time.monotonic() < deadline, ray.available_resources()
            time.sleep(2)
    finally:
        ray.shutdown()


def _command(command, destination, env, *, timeout=1500):
    with destination.open('w') as stream:
        result = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout
        )
    assert result.returncode == 0, (destination, destination.read_text()[-10000:])


def production_environment(layers, bi, inputs):
    base = os.environ.copy()
    base.pop('DS41_RELEASE', None)
    base.pop('DS41_RESUME', None)
    base.update(
        DS41_PYTHON=sys.executable,
        DS41_MODEL=str(inputs / f'{layers}l/model'),
        DS41_DATA=(
            str(inputs / 'c8-boundary-long-prompts.parquet')
            if layers == 4
            else os.environ['DS41_C8_SHORT_DATA']
        ),
        DS41_LAYERS=str(layers),
        DS41_PROMPT_LENGTH='12800' if layers == 4 else '256',
        VLLM_BATCH_INVARIANT=str(bi),
        DS41_INDEXER_DIAGNOSTIC=str(int(layers == 4)),
        DS41_INDEXER_DIAGNOSTIC_LIMIT='1',
    )
    base['PYTHONPATH'] = str(RECIPE.parents[1]) + ':' + base.get('PYTHONPATH', '')
    return base


def stage_environment(base, run, steps, resume):
    env = dict(
        base, DS41_OUTPUT=str(run), DS41_STEPS=str(steps), DS41_SAVE_FREQ=str(steps)
    )
    if resume is not None:
        env['DS41_RESUME'] = str(resume)
    return env


@pytest.mark.parametrize('layers,bi', [(2, 1), (4, 1), (2, 0)])
def test_two_updates_and_checkpoint_resume(layers, bi):
    """Require raw token equality and exact saved owner/state on a fresh process."""
    if os.environ.get('DS41_C8_ENABLED') != '1':
        pytest.skip('Requires prepared DS41_C8_INPUTS and an eight-GPU Ray cluster')
    address = os.environ['DS41_RAY_ADDRESS']
    inputs = Path(os.environ['DS41_C8_INPUTS'])
    output = Path(os.environ['DS41_C8_OUTPUT']) / f'{layers}l-bi{bi}'
    output.mkdir(parents=True, exist_ok=False)
    base = production_environment(layers, bi, inputs)
    stages = [('fresh', 2, None)]
    if bi:
        stages.append(('resumed', 4, output / 'fresh/checkpoints/global_step_2'))
    receipts = []
    for stage, steps, resume in stages:
        _idle_cluster(address)
        run = output / stage
        run.mkdir()
        if resume is not None:
            assert resume.is_dir(), resume
        env = stage_environment(base, run, steps, resume)
        _command(
            ['bash', str(RECIPE / 'run.sh')],
            run / 'train.log',
            env,
            timeout=2700 if layers == 4 else (2400 if stage == 'resumed' else 1500),
        )
        events = [
            json.loads(line)
            for path in (run / 'audit').glob('events-*.jsonl')
            for line in path.read_text().splitlines()
        ]
        rows = [row for row in events if row['kind'] == 'step']
        assert len(rows) == 2
        assert all(row['wandb_url'].startswith('https://wandb.ai/') for row in rows)
        if bi:
            _command(
                [
                    sys.executable,
                    str(RECIPE / 'verify.py'),
                    str(run),
                    '--steps',
                    '2',
                    '--start-step',
                    '1' if stage == 'fresh' else '3',
                    '--segmented-host',
                    '--parameter-counts',
                    '665' if layers == 4 else '334',
                ],
                run / 'verified.json',
                base,
            )
        if layers == 4:
            coverage = [
                row
                for row in events
                if row['kind'] == 'indexer_topk_call'
                and row['crossed_12288_query_rows']
                and row['selector'] == 'vllm_batch_invariant::top_k_per_row_prefill'
            ]
            assert {row['owner'] for row in coverage} >= {'training', 'inference'}, (
                'Both real callers must cross 12288 query rows',
                coverage,
            )
            (run / 'indexer-boundary-coverage.json').write_text(
                json.dumps(coverage, indent=2) + '\n'
            )
            probes = [
                row for row in events if row['kind'] == 'indexer_fp32_definition_probe'
            ]
            assert probes, 'The long-input BF16/FP32-definition probe did not execute'
            assert all(
                row['native_replay_equal']
                and row['parameter_versions_unchanged']
                and sum(call['truncated_rows'] for call in row['score_calls']) > 0
                for row in probes
            )
            (run / 'indexer-probes.json').write_text(
                json.dumps(probes, indent=2) + '\n'
            )
        receipts.append({'stage': stage, 'steps': rows})
        _idle_cluster(address)
    if bi:
        _command(
            [
                sys.executable,
                str(RECIPE / 'verify_checkpoint.py'),
                str(output / 'fresh'),
                str(output / 'resumed'),
            ],
            output / 'checkpoint-verified.json',
            base,
        )
    (output / 'result.json').write_text(json.dumps(receipts, indent=2) + '\n')

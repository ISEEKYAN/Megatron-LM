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

    gpus = int(os.environ.get('DS41_C8_GPUS', '8'))
    ray.init(address=address, ignore_reinit_error=True)
    deadline = time.monotonic() + 90
    try:
        assert ray.cluster_resources().get('GPU') == gpus
        while ray.available_resources().get('GPU', 0) != gpus:
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


# 6L is the release subset [1, 14, 15, 20, 21, 24]; its prompts exceed the
# 2048x8 candidate pool so the two-level Top-K actually drops blocks.
LONG_PROMPTS = {
    4: 'c8-boundary-long-prompts.parquet',
    6: 'c8-candidate-long-prompts.parquet',
    8: 'c8-candidate-long-prompts.parquet',
    9: 'c8-candidate-long-prompts.parquet',
}
PROMPT_LENGTHS = {4: '12800', 6: '17408', 8: '17408', 9: '17408'}
FROZEN_TABLES = {6: '2', 8: '2', 9: '2', 40: '2'}
# Long-sequence proxy: release subset with every 40L layer role, EP8xCP2 and
# full activation recompute (8L drops layer 15).
LONG_SEQUENCE = {8, 9}


def _parameter_counts(layers):
    # 6L trainable leaves come from its CPU init receipt, not a guess
    # (40L: EP64 CONFIG_ONLY, 1540 on every rank).
    known = {2: '334', 4: '665', 40: '1540'}
    return known.get(layers) or os.environ['DS41_PARAMETER_COUNTS']


def production_environment(layers, bi, inputs):
    base = os.environ.copy()
    base.pop('DS41_RELEASE', None)
    base.pop('DS41_RESUME', None)
    base.update(
        DS41_PYTHON=sys.executable,
        DS41_MODEL=str(inputs / f'{layers}l/model'),
        DS41_DATA=(
            str(inputs / LONG_PROMPTS[layers])
            if layers in LONG_PROMPTS
            else os.environ['DS41_C8_SHORT_DATA']
        ),
        DS41_LAYERS=str(layers),
        DS41_PROMPT_LENGTH=PROMPT_LENGTHS.get(layers, '256'),
        VLLM_BATCH_INVARIANT=str(bi),
        DS41_INDEXER_DIAGNOSTIC=str(int(layers == 4)),
        DS41_INDEXER_DIAGNOSTIC_LIMIT='1',
    )
    if layers in LONG_SEQUENCE:
        base.update(DS41_CP='2', DS41_WORLD='8', DS41_RECOMPUTE='full')
    if layers == 40:
        # Full model: EP64/DP64 over 16 nodes; four sequences per prompt need
        # 16 prompts so that every data-parallel rank receives one sequence.
        base.update(DS41_EP='64', DS41_WORLD='64', DS41_TRAIN_BATCH='16')
    base['PYTHONPATH'] = str(RECIPE.parents[1]) + ':' + base.get('PYTHONPATH', '')
    return base


def stage_environment(base, run, steps, resume):
    env = dict(
        base, DS41_OUTPUT=str(run), DS41_STEPS=str(steps), DS41_SAVE_FREQ=str(steps)
    )
    if resume is not None:
        env['DS41_RESUME'] = str(resume)
    return env


@pytest.mark.parametrize(
    'layers,bi', [(2, 1), (4, 1), (6, 1), (8, 1), (9, 1), (2, 0), (40, 1)]
)
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
            timeout={
                2: 2400 if stage == 'resumed' else 1500,
                4: 2700,
                6: 2700,
                8: 4500,
                9: 4500,
                40: 6600,
            }[layers],
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
                    _parameter_counts(layers),
                    '--frozen-tables',
                    FROZEN_TABLES.get(layers, '1'),
                    *(['--ranks', '64'] if layers == 40 else []),
                ],
                run / 'verified.json',
                base,
            )
        if layers in LONG_PROMPTS:
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
        if layers in LONG_SEQUENCE:
            assert any(
                row['kind'] == 'recompute_backward' for row in events
            ), 'Activation recompute did not run in backward'
        if layers in (6, *LONG_SEQUENCE):
            candidates = [
                row
                for row in events
                if row['kind'] == 'candidate_blocks_call' and row['dropped'] > 0
            ]
            # The two-level Top-K must actually drop blocks on both sides.
            assert {row['owner'] for row in candidates} >= {'training', 'inference'}, (
                'Candidate filtering stayed the identity',
                candidates,
            )
            assert any(
                row['kind'] == 'decoder_replay_batch_used' for row in events
            ), 'vLLM decoder SWA bounded replay did not run'
            (run / 'candidate-coverage.json').write_text(
                json.dumps(candidates, indent=2) + '\n'
            )
        if layers == 4:
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
                *(['--ranks', '64'] if layers == 40 else []),
            ],
            output / 'checkpoint-verified.json',
            base,
        )
    (output / 'result.json').write_text(json.dumps(receipts, indent=2) + '\n')

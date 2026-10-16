# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PP head receipts cover all tokens; every stage still updates and resyncs."""
import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3] / 'examples/verl/scripts/ds41_w4a8_proxy'


def make_evidence(root, pp):
    audit = root / 'audit'
    audit.mkdir()
    events = []
    for update in (1, 2):
        events.append(
            dict(
                kind='step',
                step=update,
                metrics={
                    'training/rollout_probs_diff_max': 0,
                    'rollout_corr/k3_kl': 0,
                    'actor/loss': 0.1,
                    'actor/grad_norm': 1.0,
                },
                wandb_url='https://wandb.ai/test',
            )
        )
        torch.save(
            dict(
                response_mask=torch.ones(8, 1),
                responses=torch.zeros(8, 1),
                old_log_probs=torch.full((8, 1), -2.0),
                rollout_log_probs=torch.full((8, 1), -2.0),
            ),
            audit / f'raw-{update}.pt',
        )
        torch.save(
            dict(
                advantages=torch.tensor([[1.0], [-1.0]]),
                response_mask=torch.ones(2, 1),
                token_level_rewards=torch.tensor([[0.0], [1.0]]),
                index=['a', 'b'],
            ),
            audit / f'adv-{update}.pt',
        )
        for rank in range(8):
            stage = rank // (8 // pp)
            row = dict(
                kind='optimizer',
                pid=100 + rank,
                ordinal=update,
                pp_rank=stage,
                pp_size=pp,
                trainable_parameters=10 + stage,
                success=True,
                changed=True,
                master_before=f'{rank}-{update-1}',
                master_after=f'{rank}-{update}',
                grad_norm=1.0,
            )
            if pp > 1:
                row.update(global_rank=rank, world_size=8)
            events.append(row)
            if stage == pp - 1:
                lp = torch.full((pp,), -2.0)
                torch.save(
                    dict(
                        grad_enabled=True,
                        actor_requires_grad=True,
                        actor_log_probs=lp,
                        rollout_log_probs=lp.clone(),
                        old_log_probs=lp.clone(),
                        input_ids=[torch.tensor([1])],
                        rank=rank,
                        update=update,
                    ),
                    audit / f'grad-{100+rank}-{update}.pt',
                )
    for rank in range(8):
        for generation in (1, 2, 3):
            events.append(
                dict(
                    kind='resync',
                    pid=200 + rank,
                    finished=True,
                    staging_current=0,
                    frozen_tables=1,
                    received_tables=int(generation == 1),
                    frozen_reuse=generation != 1,
                    generation=generation,
                )
            )
    write_events(root, events)
    return events


def write_events(root, events):
    (root / 'audit/events-all.jsonl').write_text(
        ''.join(json.dumps(e) + '\n' for e in events)
    )


def verify(root, pp, monkeypatch):
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'verify.py',
            str(root),
            '--steps',
            '2',
            '--parameter-counts',
            ','.join(str(10 + s) for s in range(pp)),
        ],
    )
    runpy.run_path(str(ROOT / 'verify.py'), run_name='__main__')


@pytest.mark.parametrize('pp', [1, 2])
def test_head_c5_tokens_and_all_stage_updates(tmp_path, monkeypatch, capsys, pp):
    make_evidence(tmp_path, pp)
    verify(tmp_path, pp, monkeypatch)
    out = capsys.readouterr().out
    receipt = json.loads(out)
    assert receipt['grad_forward_rank_updates'] == [
        [r, u] for r in range(8 - 8 // pp, 8) for u in (1, 2)
    ]
    assert (
        receipt['grad_forward_tokens'] == sum(r['tokens'] for r in receipt['raw']) == 16
    )
    assert receipt['optimizer_ranks'] == 8 and receipt['optimizer_steps'] == 16
    assert receipt['resync_replicas'] == 8 and receipt['resync_generations'] == 24
    assert receipt['optimizer_stage_ranks'] == [8 // pp] * pp
    assert out == json.dumps(receipt, indent=2) + '\n'
    if pp == 1:
        # Frozen pre-PP verifier output for the identical legacy evidence fixture.
        assert hashlib.sha256(out.encode()).hexdigest() == (
            '66a987ae0c7bb46099cd01a423e3b8f2b7423c77cdafbde7290a90a0ba4e1c80'
        )


@pytest.mark.parametrize(
    'fault',
    [
        'missing_first_optimizer',
        'missing_first_resync',
        'rank_duplicate',
        'stage_wrong',
        'no_identity',
        'missing_head_c5',
        'wrong_pid',
        'first_stage_c5',
        'token_deficit',
        'ulp',
    ],
)
def test_pp2_fails_closed(tmp_path, monkeypatch, fault):
    events = make_evidence(tmp_path, 2)
    audit = tmp_path / 'audit'
    if fault == 'missing_first_optimizer':
        events = [
            e for e in events if not (e['kind'] == 'optimizer' and e['pid'] == 100)
        ]
    elif fault == 'missing_first_resync':
        events = [e for e in events if not (e['kind'] == 'resync' and e['pid'] == 200)]
    elif fault in ('rank_duplicate', 'stage_wrong', 'no_identity'):
        for e in events:
            if e['kind'] == 'optimizer' and e['pid'] == 100:
                if fault == 'rank_duplicate':
                    e['global_rank'] = 1
                elif fault == 'stage_wrong':
                    e.update(pp_rank=1, trainable_parameters=11)
                else:
                    del e['global_rank']
    else:
        path = audit / 'grad-104-1.pt'
        payload = torch.load(path, weights_only=False)
        if fault == 'missing_head_c5':
            path.unlink()
        elif fault == 'wrong_pid':
            path.rename(audit / 'grad-100-1.pt')
        elif fault == 'first_stage_c5':
            payload['rank'] = 0
            torch.save(payload, path)
        elif fault == 'token_deficit':
            for key in ('actor_log_probs', 'rollout_log_probs', 'old_log_probs'):
                payload[key] = payload[key][:1]
            torch.save(payload, path)
        else:
            payload['old_log_probs'][0] = torch.nextafter(
                payload['old_log_probs'][0], torch.tensor(0.0)
            )
            torch.save(payload, path)
    write_events(tmp_path, events)
    with pytest.raises((AssertionError, KeyError)):
        verify(tmp_path, 2, monkeypatch)

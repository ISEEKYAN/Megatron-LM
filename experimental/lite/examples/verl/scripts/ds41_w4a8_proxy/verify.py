# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independently recompute strict gates from actual VERL observer tensors/events."""
import argparse
import json
import math
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('--steps', type=int, default=5)
p.add_argument('--start-step', type=int, default=1)
p.add_argument('--ranks', type=int, default=8)
p.add_argument(
    '--parameter-counts',
    default='334',
    help='Expected owned trainable tensors by PP stage, from CPU init receipt',
)
args = p.parse_args()
parameter_counts = [int(value) for value in args.parameter_counts.split(',')]
root = args.run
events = []
for path in sorted((root / 'audit').glob('events-*.jsonl')):
    events.extend(json.loads(line) for line in path.read_text().splitlines())
steps = sorted([e for e in events if e['kind'] == 'step'], key=lambda e: e['step'])
assert [e['step'] for e in steps] == list(
    range(args.start_step, args.start_step + args.steps)
), [e['step'] for e in steps]
raw = []
for path in sorted((root / 'audit').glob('raw-*.pt')):
    j = torch.load(path, map_location='cpu', weights_only=False)
    mask = j['response_mask'][:, -j['responses'].size(1) :].bool()
    a = j['old_log_probs'][mask]
    b = j['rollout_log_probs'][mask]
    assert (
        a.numel()
        and torch.isfinite(a).all()
        and torch.isfinite(b).all()
        and torch.equal(a, b)
    ), path
    raw.append(
        {
            'file': path.name,
            'tokens': a.numel(),
            'max_abs': (a - b).abs().max().item(),
            'mismatch': (a != b).sum().item(),
        }
    )
assert len(raw) == args.steps, len(raw)
advs = []
for path in sorted((root / 'audit').glob('adv-*.pt')):
    j = torch.load(path, map_location='cpu', weights_only=False)
    a = j['advantages']
    m = j['response_mask'].bool()
    s = j['token_level_rewards'].sum(-1)
    groups = {}
    for i, uid in enumerate(j['index']):
        groups.setdefault(str(uid), []).append(i)
    n = sum(bool((a[rows][m[rows]] != 0).any()) for rows in groups.values())
    assert torch.isfinite(a).all() and n > 0 and s.max() > s.min(), path
    assert torch.isfinite(s).all() and ((s >= 0) & (s <= 1)).all(), (
        'explicit numeric-character-fraction training proxy',
        s.tolist(),
    )
    advs.append(
        {
            'file': path.name,
            'groups': len(groups),
            'nonzero_groups': n,
            'reward_min': s.min().item(),
            'reward_max': s.max().item(),
            'reward_std': s.std().item(),
        }
    )
assert len(advs) == args.steps, len(advs)
for row in steps:
    d = row['metrics']
    assert d['training/rollout_probs_diff_max'] == 0 and d['rollout_corr/k3_kl'] == 0
    assert math.isfinite(float(d['actor/loss'])) and math.isfinite(
        float(d['actor/grad_norm'])
    )
    assert 'megatron-core-moe-dev' in row['wandb_url']
optimizers = {}
resync = {}
for row in events:
    if row['kind'] == 'optimizer':
        stage = row.get('pp_rank', 0)
        assert row.get('pp_size', 1) == len(parameter_counts)
        assert row['trainable_parameters'] == parameter_counts[stage]
        assert (
            row['success']
            and row['changed']
            and row['master_before'] != row['master_after']
            and math.isfinite(row['grad_norm'])
        )
        optimizers.setdefault(row['pid'], []).append(row)
    if row['kind'] == 'resync':
        assert row['finished'] and row['staging_current'] == 0
        assert row['frozen_tables'] == 1
        assert row['received_tables'] == (0 if row['frozen_reuse'] else 1)
        resync.setdefault(row['pid'], []).append(row)
assert len(optimizers) == args.ranks, len(optimizers)
stage_counts = [
    sum(rows[0].get('pp_rank', 0) == stage for rows in optimizers.values())
    for stage in range(len(parameter_counts))
]
assert stage_counts == [args.ranks // len(parameter_counts)] * len(
    parameter_counts
), stage_counts
for rows in optimizers.values():
    assert [r['ordinal'] for r in rows] == list(range(1, args.steps + 1))
    for a, b in zip(rows, rows[1:]):
        assert a['master_after'] == b['master_before']
assert len(resync) == args.ranks, len(resync)
for rows in resync.values():
    assert [r['generation'] for r in rows] == list(range(1, args.steps + 2)), [
        r['generation'] for r in rows
    ]
    assert not rows[0]['frozen_reuse'] and all(r['frozen_reuse'] for r in rows[1:])
print(
    json.dumps(
        {
            'reward_protocol': 'numeric-character-fraction-proxy-v1; DAPO correctness recorded separately, not used for training',
            'all_strict_zero': True,
            'actual_verl_steps': args.steps,
            'raw': raw,
            'advantages': advs,
            'steps': steps,
            'optimizer_ranks': len(optimizers),
            'optimizer_steps': sum(map(len, optimizers.values())),
            'resync_replicas': len(resync),
            'resync_generations': sum(map(len, resync.values())),
        },
        indent=2,
    )
)

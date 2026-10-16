# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare actual per-rank save/load receipts from distinct worker processes."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('saved', type=Path)
p.add_argument('resumed', type=Path)
args = p.parse_args()


def events(root, kind):
    return [
        e
        for f in (root / 'audit').glob('events-*.jsonl')
        for line in f.read_text().splitlines()
        if (e := json.loads(line))['kind'] == kind
    ]


saved = events(args.saved, 'checkpoint_save')
loaded = events(args.resumed, 'checkpoint_load')
assert len(saved) == len(loaded) == 8, (len(saved), len(loaded))
left = {x['rank']: x for x in saved}
right = {x['rank']: x for x in loaded}
assert set(left) == set(right) == set(range(8))
rows = []
for rank in sorted(left):
    a, b = left[rank], right[rank]
    assert a['pid'] != b['pid'], ('distinct process', rank)
    for key in ('masters', 'optimizer', 'scheduler', 'tables'):
        assert a[key] == b[key], (rank, key, a[key], b[key])
    rows.append(
        {
            'rank': rank,
            'saved_pid': a['pid'],
            'loaded_pid': b['pid'],
            'masters': a['masters'],
            'optimizer': a['optimizer'],
            'scheduler': a['scheduler'],
            'tables': a['tables'],
        }
    )
print(
    json.dumps(
        {
            'all_pass': True,
            'qualification': 'exact numerical owner/state digests across new processes; actual RNG/dataloader restored by upstream checkpoint APIs, trajectories require separate evidence',
            'ranks': rows,
        },
        indent=2,
    )
)

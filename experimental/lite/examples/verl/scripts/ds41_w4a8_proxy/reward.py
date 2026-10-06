"""Explicit deterministic pipeline proxy approved for the true release2L prefix.

Training score = digit/non-whitespace character fraction, exactly the tiny rule.
This tests on-policy updates, not mathematical correctness or model quality.
The unchanged pinned DAPO scorer runs on every real response and is recorded;
it is excluded from the training score. No seed/rank/group/ground-truth leaks.
"""

import hashlib
import json
import os
from pathlib import Path

from verl.utils.reward_score.math_dapo import compute_score as actual_math_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    chars = [c for c in solution_str if not c.isspace()]
    numeric = sum(c.isdigit() for c in chars)
    score = float(numeric / max(1, len(chars)))
    dapo = actual_math_score(solution_str, ground_truth)
    record = {
        'protocol': 'numeric-character-fraction-proxy-v1',
        'solution_str': solution_str,
        'response_sha256': hashlib.sha256(solution_str.encode('utf-8')).hexdigest(),
        'ground_truth': ground_truth,
        'data_source': data_source,
        'proxy_score': score,
        'numeric_characters': numeric,
        'response_characters': len(chars),
        'dapo_score': float(dapo['score']),
        'dapo_acc': bool(dapo['acc']),
        'dapo_pred': dapo['pred'],
    }
    audit = Path(os.environ['W4_GRPO_OUT']) / 'reward-audit'
    audit.mkdir(parents=True, exist_ok=True)
    with (audit / f'responses-{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    return {
        'score': score,
        'proxy_score': score,
        'numeric_characters': numeric,
        'response_characters': len(chars),
        'dapo_score': record['dapo_score'],
        'dapo_acc': record['dapo_acc'],
        'dapo_pred': record['dapo_pred'],
    }

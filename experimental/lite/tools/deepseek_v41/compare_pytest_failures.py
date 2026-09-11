# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare completed pytest -q -rs logs, including collection/setup errors.

Exit 0 means no candidate-only failures; it does not mean either suite passed
or that both logs exercised the same test inventory.
"""

import hashlib
import json
import re
import sys
from pathlib import Path


def read(path):
    raw = Path(path).read_bytes()
    text = raw.decode()
    outcomes = []
    for match in re.finditer(
        r'^_+ (test_.*?|ERROR (?:collecting|at setup of|at teardown of) .*?) _+$',
        text,
        re.M,
    ):
        title = match.group(1)
        if title.startswith('ERROR collecting '):
            # Test directories were reorganized in the reference tree.
            title = (
                'ERROR collecting ' + Path(title.removeprefix('ERROR collecting ')).name
            )
        outcomes.append(title)
    summary = re.findall(r'^.*\d+ failed, .*\d+ passed.*$', text, re.M)
    if len(summary) != 1:
        raise ValueError(f'Expected one completed pytest summary in {path}: {summary}')
    failures = int(re.search(r'(\d+) failed', summary[0])[1])
    errors_match = re.search(r'(\d+) errors?', summary[0])
    expected = failures + (int(errors_match[1]) if errors_match else 0)
    if len(outcomes) != expected or len(set(outcomes)) != expected:
        raise ValueError(
            f'Incomplete or ambiguous failure extraction: {len(outcomes)} vs {expected}: {outcomes}'
        )
    return {
        'path': str(path),
        'sha256': hashlib.sha256(raw).hexdigest(),
        'summary': summary[0],
        'failures_and_errors': sorted(outcomes),
    }


baseline, candidate = map(read, sys.argv[1:3])
added = sorted(
    set(candidate['failures_and_errors']) - set(baseline['failures_and_errors'])
)
print(
    json.dumps(
        {
            'baseline': baseline,
            'candidate': candidate,
            'candidate_minus_baseline': added,
            'regression_gate_passed': not added,
        },
        indent=2,
    )
)
sys.exit(bool(added))

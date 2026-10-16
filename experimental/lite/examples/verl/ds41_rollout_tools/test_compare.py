#!/usr/bin/env python3
"""CPU self-test of compare.py: one positive case and fail-closed negatives."""
import copy
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from common import save_records


def next_ulp(x):
    as_int = struct.unpack('<I', struct.pack('<f', x))[0]
    return struct.unpack('<f', struct.pack('<I', as_int - 1))[0]


from compare import compare

BASE = [
    dict(
        id='p',
        sample=s,
        index=0,
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        logprobs=[-0.125, -2.5, -7.078326225280762],
        meta={},
    )
    for s in range(2)
]


def case(edit):
    records = copy.deepcopy(BASE)
    edit(records)
    return records


def setitem(i, field, value):
    return lambda r: r[i].__setitem__(field, value)


CASES = {
    'identical': (BASE, 0, 'ZERO_DIFF'),
    'one_ulp': (
        case(lambda r: r[1]['logprobs'].__setitem__(2, next_ulp(-7.078326225280762))),
        1,
        'DIFF',
    ),
    'token_differs': (
        case(lambda r: r[0]['response_ids'].__setitem__(1, 9)),
        1,
        'DIFF',
    ),
    'missing_record': (BASE[:1], 1, 'DIFF'),
    'short_logprobs': (case(setitem(0, 'logprobs', [-0.125, -2.5])), 2, 'MALFORMED'),
    'long_logprobs': (
        case(setitem(0, 'logprobs', [-0.125, -2.5, -7.078326225280762, -1.0])),
        2,
        'MALFORMED',
    ),
    'duplicate_key': (BASE + [copy.deepcopy(BASE[0])], 2, 'MALFORMED'),
    'nan': (case(lambda r: r[0]['logprobs'].__setitem__(0, math.nan)), 2, 'MALFORMED'),
    'inf': (case(lambda r: r[0]['logprobs'].__setitem__(0, -math.inf)), 2, 'MALFORMED'),
    'not_fp32': (case(lambda r: r[0]['logprobs'].__setitem__(0, -0.1)), 2, 'MALFORMED'),
    'missing_field': (case(lambda r: r[0].pop('logprobs')), 2, 'MALFORMED'),
    'empty_response': (
        case(
            lambda r: (
                r[0].__setitem__('response_ids', []),
                r[0].__setitem__('logprobs', []),
            )
        ),
        2,
        'MALFORMED',
    ),
}


def main():
    failed = 0
    for name, (other, code, verdict) in CASES.items():
        report, got = compare(BASE, other)
        ok = (got, report['verdict']) == (code, verdict)
        failed += not ok
        print('PASS' if ok else 'FAIL', name, got, report['verdict'])
    report, got = compare([], [])
    failed += got != 2
    print('PASS' if got == 2 else 'FAIL', 'both_empty', got, report['verdict'])
    with tempfile.TemporaryDirectory() as d:  # CLI exit codes and JSONL/.pt round trip
        save_records(BASE, Path(d, 'a.jsonl'))
        save_records(BASE, Path(d, 'a.pt'))
        save_records(CASES['short_logprobs'][0], Path(d, 'bad.jsonl'))
        here = Path(__file__).with_name('compare.py')
        for b, want in (('a.pt', 0), ('bad.jsonl', 2)):
            rc = subprocess.run(
                [sys.executable, here, Path(d, 'a.jsonl'), Path(d, b)],
                capture_output=True,
            ).returncode
            failed += rc != want
            print('PASS' if rc == want else 'FAIL', 'cli', b, rc)
    print('SELFTEST', 'GREEN' if not failed else f'RED {failed}')
    sys.exit(bool(failed))


if __name__ == '__main__':
    main()

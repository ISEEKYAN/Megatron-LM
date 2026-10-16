#!/usr/bin/env python3
"""Bitwise logprob comparison of two record files (rollout or reference).

Fail-closed: exit 0 (ZERO_DIFF) only if both files are well formed and every
token's logprob has identical FP32 bits. A file is malformed if any record
lacks a field, repeats an (id, sample) key, has len(logprobs) !=
len(response_ids), or holds a non-finite or non-FP32-representable logprob.
Records are matched by (id, sample); prompt and response ids must agree, and
both files must hold the same key set unless --allow-subset.
Exit codes: 0 zero diff, 1 diff, 2 malformed input.
"""
import argparse
import json
import math
import struct
import sys

FIELDS = ('id', 'sample', 'prompt_ids', 'response_ids', 'logprobs')


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def validate(records, label):
    errors, keyed = [], {}
    for n, r in enumerate(records):
        missing = [f for f in FIELDS if f not in r]
        if missing:
            errors.append(f'{label}[{n}] missing {missing}')
            continue
        key = (str(r['id']), int(r['sample']))
        if key in keyed:
            errors.append(f'{label}[{n}] duplicate key {list(key)}')
            continue
        if not r['response_ids']:
            errors.append(f'{label}{list(key)} empty response')
        if len(r['logprobs']) != len(r['response_ids']):
            errors.append(
                f'{label}{list(key)} {len(r["logprobs"])} logprobs for {len(r["response_ids"])} tokens'
            )
        for i, x in enumerate(r['logprobs']):
            x = float(x)
            if (
                not math.isfinite(x)
                or struct.unpack('<f', struct.pack('<f', x))[0] != x
            ):
                errors.append(
                    f'{label}{list(key)} token {i}: logprob {x!r} is not a finite FP32 value'
                )
                break
        keyed[key] = r
    return keyed, errors


def compare(a_records, b_records, allow_subset=False):
    A, errors_a = validate(a_records, 'A')
    B, errors_b = validate(b_records, 'B')
    report = dict(
        errors=errors_a + errors_b,
        only_a=sorted(map(list, A.keys() - B.keys())),
        only_b=sorted(map(list, B.keys() - A.keys())),
        records=0,
        tokens=0,
        token_mismatch=[],
        logprob_mismatch=0,
        first=None,
        max_abs=0.0,
    )
    for key in sorted(A.keys() & B.keys()):
        a, b = A[key], B[key]
        report['records'] += 1
        if a['prompt_ids'] != b['prompt_ids'] or a['response_ids'] != b['response_ids']:
            n = next(
                (
                    i
                    for i, (x, y) in enumerate(
                        zip(a['response_ids'], b['response_ids'])
                    )
                    if x != y
                ),
                min(len(a['response_ids']), len(b['response_ids'])),
            )
            report['token_mismatch'].append(
                dict(
                    key=list(key),
                    first_token=n,
                    prompt_equal=a['prompt_ids'] == b['prompt_ids'],
                )
            )
            continue
        if len(a['logprobs']) != len(b['logprobs']):
            continue  # already reported as malformed
        for i, (x, y) in enumerate(zip(a['logprobs'], b['logprobs'])):
            report['tokens'] += 1
            if bits(float(x)) != bits(float(y)):
                report['logprob_mismatch'] += 1
                report['max_abs'] = max(report['max_abs'], abs(float(x) - float(y)))
                if report['first'] is None:
                    report['first'] = dict(key=list(key), token=i, a=x, b=y)
    if report['errors'] or report['records'] == 0:
        report['verdict'], code = 'MALFORMED', 2
    elif (
        report['token_mismatch']
        or report['logprob_mismatch']
        or (not allow_subset and (report['only_a'] or report['only_b']))
    ):
        report['verdict'], code = 'DIFF', 1
    else:
        report['verdict'], code = 'ZERO_DIFF', 0
    return report, code


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('a', help='records file A (.jsonl or .pt)')
    ap.add_argument('b', help='records file B (.jsonl or .pt)')
    ap.add_argument('--json', default=None, help='also write the report as JSON here')
    ap.add_argument(
        '--allow-subset',
        action='store_true',
        help='compare only records present in both (default: key sets must match)',
    )
    args = ap.parse_args()
    from common import load_records

    report, code = compare(
        load_records(args.a), load_records(args.b), args.allow_subset
    )
    text = json.dumps(report, indent=1)
    print(text)
    if args.json:
        open(args.json, 'w').write(text)
    sys.exit(code)


if __name__ == '__main__':
    main()

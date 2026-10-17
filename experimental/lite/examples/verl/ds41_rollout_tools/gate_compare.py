#!/usr/bin/env python3
"""Compare two ds41_gate.py receipts; exit 0 only when the gate holds.

Forward only: the token log-probabilities of every global sequence must be
bitwise equal -- for the same configuration (fresh processes, in-process
repeat, recompute, resume) and across parallel partitions (DP, PP, CP, EP)
alike. Gradients are not compared; every step of both runs must report all
finalized gradients finite.

  python gate_compare.py A B --out receipt.json
  python gate_compare.py A A --suffix-b=-repeat1    # in-process repeat
"""
import argparse
import json
import sys
from pathlib import Path

PARTITION = ('world', 'dp_size', 'ep_size', 'pp_size', 'cp_size')


def _logprobs(directory, suffix):
    rows = (Path(directory) / f'logprobs{suffix}.jsonl').read_text().splitlines()
    return {row['sequence']: row['logprobs'] for row in map(json.loads, rows)}


def _record(directory):
    return json.loads((Path(directory) / 'record.json').read_text())


def gradients_finite(record):
    """Per step of a run: gradients checked and the non-finite ones."""
    steps = {k: v for k, v in record.items() if k.startswith('step-')}
    return {
        step: dict(
            checked=value.get('gradients_checked', 0),
            nonfinite=value.get('gradients_nonfinite', ['<not recorded>']),
        )
        for step, value in steps.items()
    }


def compare(lp_a, lp_b, finite_a, finite_b):
    if lp_a.keys() != lp_b.keys():
        raise SystemExit('the two runs scored different sequences')
    unequal = sorted(i for i in lp_a if lp_a[i] != lp_b[i])
    finite = dict(a=finite_a, b=finite_b)
    gradients_ok = all(
        steps and all(s['checked'] > 0 and not s['nonfinite'] for s in steps.values())
        for steps in finite.values()
    )
    return dict(
        logprobs=dict(sequences=len(lp_a), unequal=unequal[:20]),
        gradients_finite=finite,
        passed=not unequal and gradients_ok,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--suffix-a', default='')
    ap.add_argument('--suffix-b', default='')
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    record_a, record_b = _record(args.a), _record(args.b)
    report = compare(
        _logprobs(args.a, args.suffix_a),
        _logprobs(args.b, args.suffix_b),
        gradients_finite(record_a),
        gradients_finite(record_b),
    )
    report.update(
        label=args.label,
        partition=dict(
            fields=list(PARTITION),
            a=[record_a.get(k) for k in PARTITION],
            b=[record_b.get(k) for k in PARTITION],
        ),
        a=f'{args.a}{args.suffix_a}',
        b=f'{args.b}{args.suffix_b}',
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n')
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

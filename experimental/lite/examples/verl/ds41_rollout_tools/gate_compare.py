#!/usr/bin/env python3
"""Compare two ds41_gate.py receipts; exit 0 only when the criterion holds.

Log-probabilities are always compared bitwise, per global sequence.

  --criterion bitwise    a partition change that keeps the DP size (or the same
                         configuration, in one process or fresh ones): every
                         parameter's gradient bit fingerprint equal
  --criterion tolerance  a partition change that changes the DP size (the FP32
                         sums are reordered): B must have been run with
                         --compare-grads against A's saved gradients, and every
                         parameter must satisfy rel <= --rel or abs <= --abs
                         (test_cp_ep_parity: 5e-5 relative to max |A|, 2e-6
                         absolute); both runs must also be bitwise against their
                         own in-process repeat (--repeats 2)
  --criterion auto       (default) bitwise when both records have the same DP
                         size, tolerance otherwise

  python gate_compare.py A B --out receipt.json
  python gate_compare.py A A --suffix-b=-repeat1    # in-process repeat
"""
import argparse
import json
import sys
from pathlib import Path

REL, ABS = 5e-5, 2e-6


def _load(directory, suffix):
    directory = Path(directory)
    lps = {}
    for line in (directory / f'logprobs{suffix}.jsonl').read_text().splitlines():
        row = json.loads(line)
        lps[row['sequence']] = row['logprobs']
    grads = json.loads((directory / f'grads{suffix}.json').read_text())
    return lps, grads


def _dp_size(directory):
    return json.loads((Path(directory) / 'record.json').read_text())['dp_size']


def _quantiles(values):
    if not values:
        return {}
    v = sorted(values)

    def pick(f):
        return v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]

    return dict(p50=pick(0.5), p99=pick(0.99), max=v[-1], count=len(v))


def _bitwise(a, b):
    (lp_a, grad_a), (lp_b, grad_b) = a, b
    lp_unequal = sorted(i for i in lp_a if lp_a[i] != lp_b.get(i))
    grad_unequal = sorted(
        n for n in grad_a if grad_a[n]['fingerprint'] != grad_b[n]['fingerprint']
    )
    return lp_unequal, grad_unequal


def compare(a, b, criterion='bitwise', rel=REL, absolute=ABS, repeats=None):
    (lp_a, grad_a), (lp_b, grad_b) = a, b
    if lp_a.keys() != lp_b.keys():
        raise SystemExit('the two runs scored different sequences')
    if grad_a.keys() != grad_b.keys():
        missing = sorted(grad_a.keys() ^ grad_b.keys())
        raise SystemExit(f'parameter sets differ: {missing[:5]}')
    lp_unequal, bit_unequal = _bitwise(a, b)
    worst = sorted(
        (
            (
                max(abs(p - q) for p, q in zip(x['sketch'], grad_b[n]['sketch']))
                / (max(x['norm'], grad_b[n]['norm']) or 1.0),
                n,
            )
            for n, x in grad_a.items()
        ),
        reverse=True,
    )
    report = dict(
        criterion=criterion,
        logprobs=dict(sequences=len(lp_a), unequal=lp_unequal[:20]),
        grads=dict(
            parameters=len(grad_a),
            bitwise_unequal=len(bit_unequal),
            first_unequal=bit_unequal[:10],
            worst_sketch_rel=[[round(e, 9), n] for e, n in worst[:10]],
        ),
    )
    if criterion == 'bitwise':
        report['passed'] = not lp_unequal and not bit_unequal
        return report
    measured = {
        n: g['versus'] for n, g in grad_b.items() if 'abs' in g.get('versus', {})
    }
    unmeasured = sorted(grad_b.keys() - measured.keys())
    outliers = sorted(
        (
            (v['rel'], v['abs'], n)
            for n, v in measured.items()
            if v['rel'] > rel and v['abs'] > absolute
        ),
        reverse=True,
    )
    experts = [n for n in measured if '.ffn.experts.' in n]
    dense = [n for n in measured if '.ffn.experts.' not in n]
    report['tolerance'] = dict(
        rel=rel,
        abs=absolute,
        unmeasured=len(unmeasured),
        first_unmeasured=unmeasured[:10],
        outliers=len(outliers),
        worst=[[r, e, n] for r, e, n in outliers[:20]],
        rel_distribution=_quantiles([v['rel'] for v in measured.values()]),
        abs_distribution=_quantiles([v['abs'] for v in measured.values()]),
        expert_rel_distribution=_quantiles([measured[n]['rel'] for n in experts]),
        dense_rel_distribution=_quantiles([measured[n]['rel'] for n in dense]),
    )
    report['run_to_run'] = repeats
    repeat_ok = repeats is not None and all(
        not r['logprobs_unequal'] and not r['grads_unequal'] for r in repeats.values()
    )
    report['passed'] = not lp_unequal and not unmeasured and not outliers and repeat_ok
    return report


def run_to_run(directories):
    """Each side against its own in-process repeat (--repeats 2), bitwise."""
    out = {}
    for directory in directories:
        if not (Path(directory) / 'grads-repeat1.json').exists():
            return None
        lp, grads = _bitwise(_load(directory, ''), _load(directory, '-repeat1'))
        out[str(directory)] = dict(logprobs_unequal=lp[:10], grads_unequal=grads[:10])
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--suffix-a', default='')
    ap.add_argument('--suffix-b', default='')
    ap.add_argument(
        '--criterion', choices=('auto', 'bitwise', 'tolerance'), default='auto'
    )
    ap.add_argument('--rel', type=float, default=REL)
    ap.add_argument('--abs', type=float, default=ABS)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    dp = (_dp_size(args.a), _dp_size(args.b))
    criterion = args.criterion
    if criterion == 'auto':
        criterion = 'bitwise' if dp[0] == dp[1] else 'tolerance'
    report = compare(
        _load(args.a, args.suffix_a),
        _load(args.b, args.suffix_b),
        criterion,
        args.rel,
        args.abs,
        run_to_run((args.a, args.b)) if criterion == 'tolerance' else None,
    )
    report.update(
        label=args.label,
        dp_size=list(dp),
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

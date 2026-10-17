#!/usr/bin/env python3
"""Compare two ds41_gate.py receipts; exit 0 only when the criterion holds.

Log-probabilities are always compared bitwise, per global sequence.

  --criterion bitwise          the same configuration (one process or fresh
                               ones; recompute, resume): every parameter's
                               gradient bit fingerprint equal
  --criterion cross-partition  a different parallel partition (DP, PP, CP or EP):
                               both runs bitwise against their own in-process
                               repeat (--repeats 2), log-probs and every
                               gradient; gradient errors across the two are
                               reported (median, max, per layer), not judged --
                               element-wise when B was run with --compare-grads
                               against A's saved gradients (max |a - b| and
                               that over max |A|, with the count beyond
                               test_cp_ep_parity's 5e-5 relative and 2e-6
                               absolute for reference), by sketch otherwise
  --criterion auto             (default) bitwise when both records have the
                               same partition, cross-partition otherwise

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


PARTITION = ('world', 'dp_size', 'ep_size', 'pp_size', 'cp_size')


def _partition(directory):
    record = json.loads((Path(directory) / 'record.json').read_text())
    split = record.get('args', {}).get('pipeline_split_layer')
    return [record[k] for k in PARTITION] + [split]


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


def _by_layer(measured):
    groups = {}
    for name, v in measured.items():
        parts = name.split('.')
        layer = '.'.join(parts[:2]) if parts[0] == 'layers' else parts[0]
        groups.setdefault(layer, []).append(v['rel'])
    return groups


def _errors(grad_a, grad_b):
    """Element-wise errors from B's --compare-grads fields, else sketch errors."""
    measured = {
        n: g['versus'] for n, g in grad_b.items() if 'abs' in g.get('versus', {})
    }
    sketch = {
        n: max(abs(p - q) for p, q in zip(x['sketch'], grad_b[n]['sketch']))
        / (max(x['norm'], grad_b[n]['norm']) or 1.0)
        for n, x in grad_a.items()
    }
    if not measured:
        rel = sketch
        return dict(
            kind='sketch',
            rel_distribution=_quantiles(list(rel.values())),
            expert_rel_distribution=_quantiles(
                [v for n, v in rel.items() if '.ffn.experts.' in n]
            ),
            worst=[[v, n] for n, v in sorted(rel.items(), key=lambda kv: -kv[1])][:20],
        )
    beyond = sorted(
        (
            (v['rel'], v['abs'], n)
            for n, v in measured.items()
            if v['rel'] > REL and v['abs'] > ABS
        ),
        reverse=True,
    )
    return dict(
        kind='element-wise',
        unmeasured=sorted(grad_b.keys() - measured.keys())[:10],
        rel_distribution=_quantiles([v['rel'] for v in measured.values()]),
        abs_distribution=_quantiles([v['abs'] for v in measured.values()]),
        expert_rel_distribution=_quantiles(
            [v['rel'] for n, v in measured.items() if '.ffn.experts.' in n]
        ),
        dense_rel_distribution=_quantiles(
            [v['rel'] for n, v in measured.items() if '.ffn.experts.' not in n]
        ),
        per_layer={
            layer: _quantiles(values)
            for layer, values in sorted(_by_layer(measured).items())
        },
        beyond_parity_tolerance=len(beyond),
        worst=[[r, e, n] for r, e, n in beyond[:20]],
    )


def compare(a, b, criterion='bitwise', repeats=None):
    (lp_a, grad_a), (lp_b, grad_b) = a, b
    if lp_a.keys() != lp_b.keys():
        raise SystemExit('the two runs scored different sequences')
    if grad_a.keys() != grad_b.keys():
        missing = sorted(grad_a.keys() ^ grad_b.keys())
        raise SystemExit(f'parameter sets differ: {missing[:5]}')
    lp_unequal, bit_unequal = _bitwise(a, b)
    report = dict(
        criterion=criterion,
        logprobs=dict(sequences=len(lp_a), unequal=lp_unequal[:20]),
        grads=dict(
            parameters=len(grad_a),
            bitwise_unequal=len(bit_unequal),
            first_unequal=bit_unequal[:10],
        ),
    )
    if criterion == 'bitwise':
        report['passed'] = not lp_unequal and not bit_unequal
        return report
    # A partition change is judged on log-probs and on each side being
    # reproducible; gradient errors across the two are reported only.
    report['errors'] = _errors(grad_a, grad_b)
    report['run_to_run'] = repeats
    repeat_ok = repeats is not None and all(
        not r['logprobs_unequal'] and not r['grads_unequal'] for r in repeats.values()
    )
    report['passed'] = not lp_unequal and repeat_ok
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
        '--criterion', choices=('auto', 'bitwise', 'cross-partition'), default='auto'
    )
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    partitions = (_partition(args.a), _partition(args.b))
    criterion = args.criterion
    if criterion == 'auto':
        same = partitions[0] == partitions[1]
        criterion = 'bitwise' if same else 'cross-partition'
    report = compare(
        _load(args.a, args.suffix_a),
        _load(args.b, args.suffix_b),
        criterion,
        run_to_run((args.a, args.b)) if criterion == 'cross-partition' else None,
    )
    report.update(
        label=args.label,
        partition=dict(fields=[*PARTITION, 'split'], a=partitions[0], b=partitions[1]),
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

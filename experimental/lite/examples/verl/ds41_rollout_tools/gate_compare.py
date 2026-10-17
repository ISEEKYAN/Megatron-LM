#!/usr/bin/env python3
"""Compare two ds41_gate.py receipts; exit 0 only when the configured criterion holds.

  --grads bitwise   every parameter's bit fingerprint equal (same DP size, same
                    process or fresh processes)
  --grads sketch    every parameter's signed-projection sketch within --rel of
                    its gradient norm (a partition that changes the DP size
                    reorders the FP32 sums; a wrong reduction scope is O(1))
Log-probabilities are always compared bitwise, per global sequence.

  python gate_compare.py A B --grads bitwise --out receipt.json
  python gate_compare.py A A --suffix-b -repeat1    # in-process repeat
"""
import argparse
import json
import sys
from pathlib import Path


def _load(directory, suffix):
    directory = Path(directory)
    lps = {}
    for line in (directory / f'logprobs{suffix}.jsonl').read_text().splitlines():
        row = json.loads(line)
        lps[row['sequence']] = row['logprobs']
    grads = json.loads((directory / f'grads{suffix}.json').read_text())
    return lps, grads


def compare(a, b, grads='bitwise', rel=1e-4):
    (lp_a, grad_a), (lp_b, grad_b) = a, b
    report = dict(grads_mode=grads, rel=rel)
    if lp_a.keys() != lp_b.keys():
        raise SystemExit('the two runs scored different sequences')
    lp_unequal = [i for i in lp_a if lp_a[i] != lp_b[i]]
    report['logprobs'] = dict(sequences=len(lp_a), unequal=sorted(lp_unequal)[:20])
    if grad_a.keys() != grad_b.keys():
        missing = sorted(grad_a.keys() ^ grad_b.keys())
        raise SystemExit(f'parameter sets differ: {missing[:5]}')
    bit_unequal, worst = [], []
    for name, x in grad_a.items():
        y = grad_b[name]
        if x['fingerprint'] != y['fingerprint']:
            bit_unequal.append(name)
        scale = max(x['norm'], y['norm']) or 1.0
        error = max(abs(p - q) for p, q in zip(x['sketch'], y['sketch'])) / scale
        worst.append((error, name))
    worst.sort(reverse=True)
    report['grads'] = dict(
        parameters=len(grad_a),
        bitwise_unequal=len(bit_unequal),
        first_unequal=bit_unequal[:10],
        worst_sketch_rel=[[round(e, 9), n] for e, n in worst[:10]],
    )
    grads_ok = not bit_unequal if grads == 'bitwise' else worst[0][0] <= rel
    report['passed'] = not lp_unequal and grads_ok
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--suffix-a', default='')
    ap.add_argument('--suffix-b', default='')
    ap.add_argument('--grads', choices=('bitwise', 'sketch'), default='bitwise')
    ap.add_argument('--rel', type=float, default=1e-4)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    report = compare(
        _load(args.a, args.suffix_a), _load(args.b, args.suffix_b), args.grads, args.rel
    )
    report.update(
        label=args.label, a=f'{args.a}{args.suffix_a}', b=f'{args.b}{args.suffix_b}'
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n')
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

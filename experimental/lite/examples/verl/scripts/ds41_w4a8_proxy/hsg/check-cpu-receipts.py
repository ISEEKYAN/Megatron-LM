"""Reject incomplete or failed CPU receipts before releasing the GPU stage."""

import argparse
import json
from pathlib import Path


def require_success(value, source):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('ok', 'passed') and item is not True:
                raise ValueError(f'{source}: {key}={item!r}')
            require_success(item, source)
    elif isinstance(value, list):
        for item in value:
            require_success(item, source)


def verify(config, output, *, world, pp):
    config, output = Path(config), Path(output)
    result = json.loads((output / 'result.json').read_text())
    require_success(result, output / 'result.json')
    if result.get('passed') is not True or result.get('collect_only') is not True:
        raise ValueError('CPU collection receipt is absent or unsuccessful')
    if result.get('pytest_exit') != 0:
        raise ValueError('CPU pytest did not exit successfully')
    replay = json.loads((config / 'vllm-cli.json').read_text())
    if replay.get('swa_bounded_replay') is not False:
        raise ValueError('Resolved vLLM SWA bounded replay is not False')
    engram = replay.get('engram_config', {})
    if (
        engram.get('cpu_offload') is not True
        or engram.get('dp_shared_memory') is not True
    ):
        raise ValueError('Resolved vLLM Engram host/shared defaults are not True/True')
    for path in sorted(config.glob('*.json')):
        require_success(json.loads(path.read_text()), path)
    census = {}
    for prefix in ('checkpoint-headers', 'model-counts'):
        paths = sorted(config.glob(f'{prefix}-rank-*.json'))
        if pp > 1 or paths:
            ranks = []
            for path in paths:
                row = json.loads(path.read_text())
                if row.get('passed') is not True:
                    raise ValueError(f'{path}: missing successful rank receipt')
                ranks.append(row['global_rank'])
            if sorted(ranks) != list(range(world)):
                raise ValueError(f'{prefix}: incomplete or duplicate ranks {ranks}')
            census[prefix] = len(ranks)
    events = 0
    for path in sorted(output.rglob('events-*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            events += 1
            kind = row.get('kind')
            if kind == 'decoder_replay_batch_used' or (
                kind == 'decoder_replay_enabled' and row.get('enabled') is not False
            ):
                raise ValueError(f'{path}: unexpected {kind}')
    return dict(
        passed=True, ranks=census, audit_events=events, swa_bounded_replay=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--world', required=True, type=int)
    parser.add_argument('--pp', required=True, type=int)
    args = parser.parse_args()
    try:
        receipt = verify(args.config, args.output, world=args.world, pp=args.pp)
    except Exception as exc:
        receipt = dict(passed=False, error=f'{type(exc).__name__}: {exc}')
    (Path(args.output) / 'receipt-validation.json').write_text(
        json.dumps(receipt, indent=2) + '\n'
    )
    print('DS41_CPU_RECEIPTS', json.dumps(receipt), flush=True)
    raise SystemExit(0 if receipt['passed'] else 1)

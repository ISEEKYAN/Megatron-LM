"""CPU assertions for the Phase A contract; no production implementation."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).parent


def expand(contract):
    pairs = []
    for family in contract['families']:
        indices = list(itertools.product(*family['indices']))
        assert len(indices) == family['count']
        assert family['exclusion'] is None
        pairs.extend((family['pattern'].format(*i), family['target'].format(*i))
                     for i in indices)
    assert len(pairs) == len(dict(pairs)) == len({v for _, v in pairs}) == 96085
    return dict(pairs)


def check_keys(actual, expected):
    assert set(actual) == set(expected), 'Missing or unexpected release keys'


def leaves(value, prefix=''):
    for key, item in value.items():
        path = prefix + key
        if isinstance(item, dict):
            yield from leaves(item, path + '.')
        else:
            yield path, item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    contract = json.loads((ROOT / 'weights.json').read_text())
    expected = expand(contract)
    actual = json.loads(Path(args.index).read_text())['weight_map']
    check_keys(actual, expected)
    digest = hashlib.sha256('\n'.join(sorted(actual)).encode()).hexdigest()
    assert digest == contract['key_sha256']
    mtp = sorted(k for k in expected if k.startswith('mtp.'))
    assert len(mtp) == 2401
    for size in (1, 2, 3, 8, 2402):
        shards = [mtp[rank::size] for rank in range(size)]
        restored = list(itertools.chain.from_iterable(shards))
        assert len(restored) == len(set(restored)) == 2401
        assert set(restored) == set(mtp)
    config = dict(leaves(json.loads(Path(args.config).read_text())))
    rows = json.loads((ROOT / 'config.json').read_text())
    assert len(rows) == len({row['field'] for row in rows})
    assert {row['field']: row['value'] for row in rows} == config
    assert all(row['consumer'] and row['status'] == 'planned' for row in rows)
    assert config['text_config.dspark_block_size'] == 5
    for bad in (set(actual) - {mtp[0]}, set(actual) | {'mtp.3.unknown.weight'}):
        try:
            check_keys(bad, expected)
        except AssertionError:
            pass
        else:
            raise AssertionError('Negative control accepted')
    assert {k for k in mtp if '.main_norm.' in k} == {'mtp.0.main_norm.weight'}
    assert {k for k in mtp if '.confidence_head.' in k} == {
        'mtp.2.confidence_head.proj.weight'}
    print(f'keys={len(actual)}/96085 mtp={len(mtp)}/2401 config={len(rows)} '
          'shard_layouts=5 negative_controls=2 PASS (schema only)')


if __name__ == '__main__':
    main()

"""Validate plan structure/declared ownership, not future implementation correctness."""
import argparse
import copy
from graphlib import TopologicalSorter
import json
from pathlib import Path
import re

ROOT = Path(__file__).parent
EXPECTED = set('B1 B2 B3 C1 C2a C2b C2c C3 C4 D1 D2 D3 D4 D5 D6 D7 E0 E1a E1b E1c E1d E1e E2 E3 E4 E5t E5v F1 F2 F3 G1 G2'.split())


def check(text, ownership, decisions, active=()):
    pairs = re.findall(r'^\| ([A-G]\d+[a-z]?)-S:.*?\| ([A-G]\d+[a-z]?)-I:', text, re.M)
    assert len(pairs) == len(EXPECTED)
    assert {s for s, i in pairs} == EXPECTED
    assert all(s == i for s, i in pairs), 'mismatched S/I row'
    graph = {f'{p}-{k}': set() for p in EXPECTED for k in ('S', 'I')}
    for p in EXPECTED:
        graph[f'{p}-I'].add(f'{p}-S')
    def endpoint(value):
        m = re.fullmatch(r'([A-G]\d+[a-z]?)([SI])(?:\[([^]]+)\])?', value)
        assert m, value
        p, kind, label = m.groups()
        node = f'{p}-{kind}'
        assert node in graph and (label is None or label == node), value
        return node
    diagram = text.split('```mermaid\n', 1)[1].split('```', 1)[0]
    for line in diagram.splitlines()[1:]:
        source, target = map(endpoint, line.strip().split(' --> '))
        graph[target].add(source)
    artifacts = ownership['artifacts']
    assert len({a['artifact'] for a in artifacts}) == len(artifacts), 'duplicate producer'
    for a in artifacts:
        assert a['producer'] in graph and a['consumers']
        for consumer in a['consumers']:
            assert consumer in graph
            assert a['producer'] in graph[consumer], (a['artifact'], consumer, 'missing producer edge')
    assert {a['artifact'] for a in artifacts} == {
        'visual_classes', 'injected_attention', 'native_candidates_and_integration',
        'table_representation', 'optimizer_representation', 'optimizer_algorithm', 'indexer_training'}
    assert next(a for a in artifacts if a['artifact'] == 'visual_classes')['producer'] == 'C4-I'
    for node, required in {
        'B1-I': {'B3-I'}, 'C1-I': {'B1-I'}, 'D1-I': {'B1-I', 'B2-I'},
        'D2-I': {'D1-I', 'C2a-I', 'C2b-I', 'C2c-I'},
        'F2-I': {'F1-I', 'C4-I', 'E1e-S', 'E2-S'},
        'E4-I': {'E3-I', 'E2-I', 'F2-I'}, 'E5v-I': {'E4-I', 'F3-I'},
        'G1-I': {'E5t-I', 'E5v-I', 'G2-I', 'F3-I', 'D6-I'},
    }.items():
        assert required <= graph[node], (node, required - graph[node])
    assert len(list(TopologicalSorter(graph).static_order())) == 64
    ids = [d['id'] for d in decisions]
    assert ids == [f'O{i:02}' for i in range(1, 18)]
    for d in decisions:
        for field in ('question', 'candidates', 'evidence_gap', 'authority', 'active_when', 'blocks'):
            assert d[field], (d['id'], field)
        assert len(d['candidates']) >= 2
        assert set(d['blocks']) <= graph.keys()
        assert d['status'] in ('OPEN', 'RESOLVED')
        if d['status'] == 'RESOLVED':
            assert all(d['resolution'].get(k) for k in ('value', 'phase', 'owners', 'approver', 'evidence'))
    for ident in active:
        assert ident in ids, f'unknown active decision {ident}'
        assert decisions[ids.index(ident)]['status'] == 'RESOLVED', f'active decision {ident} remains OPEN'
    review_rows = re.findall(r'^\| (\d+) [^|]+\|', text, re.M)
    assert list(map(int, review_rows)) == list(range(1, 17))


def mutations(text, ownership, decisions):
    cases = []
    swapped = text.replace('B1-I:', 'TEMP-I:').replace('B3-I:', 'B1-I:').replace('TEMP-I:', 'B3-I:')
    cases.append(('swapped S/I rows', swapped, ownership, decisions, ()))
    for name, old, new in [
        ('missing algorithm edge', '  F1I --> F2I[F2-I]\n', ''),
        ('wrong graph label', 'B3I[B3-I]', 'B3I[G1-I]'),
        ('missing cross-S edge', '  E1eS[E1e-S] --> C4I[C4-I]\n', ''),
        ('producer cycle', 'flowchart TD', 'flowchart TD\n  F3I --> C4I'),
    ]:
        assert old in text
        cases.append((name, text.replace(old, new), ownership, decisions, ()))
    changed = copy.deepcopy(ownership)
    changed['artifacts'][0]['producer'] = 'F3-I'
    cases.append(('wrong visual owner', text, changed, decisions, ()))
    duplicate = copy.deepcopy(ownership)
    duplicate['artifacts'].append(dict(duplicate['artifacts'][0], producer='F3-I'))
    cases.append(('duplicate visual owner', text, duplicate, decisions, ()))
    missing = copy.deepcopy(decisions)
    missing[15]['blocks'] = []
    cases.append(('missing normalization blockers', text, ownership, missing, ()))
    cases.append(('unresolved active objective', text, ownership, decisions, ('O16',)))
    for name, t, o, d, active in cases:
        try:
            check(t, o, d, active)
        except (AssertionError, ValueError):
            print(f'REJECTED: {name}')
        else:
            raise AssertionError(f'mutation accepted: {name}')
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--active-decision', action='append', default=[])
    args = parser.parse_args()
    text = (ROOT / 'deepseek_v41_phases_bg_v4.md').read_text()
    ownership = json.loads((ROOT / 'deepseek_v41_dependencies.json').read_text())
    decisions = json.loads((ROOT / 'deepseek_v41_decisions.json').read_text())
    check(text, ownership, decisions, args.active_decision)
    count = mutations(text, ownership, decisions)
    print(f'PLAN_STRUCTURE_OK: 32 row-matched S/I pairs; 64-node acyclic graph; declared artifact ownership; 17 decisions; {count} rejected mutations')
    print('Structural evidence only; OPEN decisions remain unresolved; no implementation/training acceptance.')


if __name__ == '__main__':
    main()

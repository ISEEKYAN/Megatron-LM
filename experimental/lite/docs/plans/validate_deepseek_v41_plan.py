"""Check plan identifiers and dependency structure, not model correctness."""
from graphlib import TopologicalSorter
from pathlib import Path
import re


def main():
    text = Path(__file__).with_name('deepseek_v41_phases_bg_v4.md').read_text()
    expected = set('B1 B2 B3 C1 C2a C2b C2c C3 C4 D1 D2 D3 D4 D5 D6 D7 E0 E1a E1b E1c E1d E1e E2 E3 E4 E5t E5v F1 F2 F3 G1 G2'.split())
    specs = re.findall(r'^\| ([A-G]\d+[a-z]?)-S:', text, re.M)
    implementations = re.findall(r'\| ([A-G]\d+[a-z]?)-I:', text)
    assert len(specs) == len(set(specs)) and set(specs) == expected
    assert len(implementations) == len(set(implementations))
    assert set(implementations) == expected
    graph = {node: set() for node in expected}
    diagram = text.split('```mermaid\n', 1)[1].split('```', 1)[0]
    for line in diagram.splitlines()[1:]:
        source, target = line.strip().split(' --> ')
        source = source.split('[')[0][:-1]
        target = target.split('[')[0][:-1]
        assert source in expected and target in expected
        graph[target].add(source)
    order = list(TopologicalSorter(graph).static_order())
    assert len(order) == len(expected)
    for node, required in {
        'B1': {'B3'}, 'C1': {'B1'}, 'D1': {'B1', 'B2'},
        'D2': {'D1', 'C2a', 'C2b', 'C2c'},
        'E4': {'E3', 'E2', 'F2'}, 'E5v': {'E4', 'F3'},
        'G1': {'E5t', 'E5v', 'G2', 'F3'},
    }.items():
        assert required <= graph[node], (node, required - graph[node])
    review_rows = re.findall(r'^\| (\d+) [^|]+\|', text, re.M)
    assert list(map(int, review_rows)) == list(range(1, 17))
    print(f'PLAN_STRUCTURE_OK: {len(expected)} S/I pairs; acyclic DAG; 16 review items')
    print('Structural checks only; no implementation/training acceptance asserted.')


if __name__ == '__main__':
    main()

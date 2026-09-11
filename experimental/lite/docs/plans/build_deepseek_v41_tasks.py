"""Render and verify task-ready S/I records from the canonical plan and decisions."""
import argparse
import json
from graphlib import TopologicalSorter
from pathlib import Path
import re

ROOT = Path(__file__).parent


def build():
    text = (ROOT / 'deepseek_v41_phases_bg_v4.md').read_text()
    decisions = json.loads((ROOT / 'deepseek_v41_decisions.json').read_text())
    rows = {}
    for line in text.splitlines():
        if re.match(r'^\| [A-G]\d+[a-z]?-S:', line):
            s, i, execution, deferred = [v.strip() for v in line.strip('|').split('|')]
            ident, spec = s.split(': ', 1)
            impl_id, impl = i.split(': ', 1)
            assert impl_id == ident[:-1] + 'I'
            rows[ident[:-2]] = (spec, impl, execution, deferred)
    graph = {p+'-'+k: set() for p in rows for k in ('S', 'I')}
    for p in rows:
        graph[p+'-I'].add(p+'-S')
    diagram = text.split('```mermaid\n', 1)[1].split('```', 1)[0]
    for line in diagram.splitlines()[1:]:
        source, target = line.strip().split(' --> ')
        def node(v):
            m = re.fullmatch(r'([A-G]\d+[a-z]?)([SI])(?:\[[^]]+\])?', v)
            assert m, v
            return m[1]+'-'+m[2]
        graph[node(target)].add(node(source))
    direct = {n: {d['id'] for d in decisions if d['status'] == 'OPEN' and n in d['blocks']} for n in graph}
    inherited = {}
    tasks = []
    for n in TopologicalSorter({n: sorted(deps) for n, deps in graph.items()}).static_order():
        p, kind = n.rsplit('-', 1)
        spec, impl, execution, deferred = rows[p]
        inherited[n] = direct[n] | set().union(*(inherited[d] for d in graph[n]))
        ac = ([spec, 'Publish reviewed interfaces, precision and independent expected fixtures; consume the named A prerequisites and resolved policy without reopening them.',
               'Enumerate active decision IDs and run validate_deepseek_v41_plan.py with each --active-decision ID; unresolved active choices block acceptance.'] if kind == 'S' else
              [impl, execution, 'Run actual implementation against the S fixtures; record commands, exit status, numerical thresholds and applicable discriminating mutations. GPU tests require Slurm job ID, sacct rc=0 and non-skip execution.'])
        constraints = ['Post-training scope only; no DSpark forward or rollout generation.', 'Full-gate dependencies below are mandatory for acceptance; forward-only partial progress is not completion.']
        if p.startswith('E1'):
            constraints.append('Engram tables/scales remain GPU-resident; no CPU table offload or host/RDMA prefetch; account for approved state/workspace in memory budget.')
        if p in ('E1e','E2'):
            ac.append('Obtain evidence or approved policy for post-training Engram trainability before update implementation; O08/O09 remain unresolved. If frozen, obtain explicit scope adjustment and separately audit token embedding/head optimizer needs.')
        if p in ('F2','F3','E5v'):
            ac.append('Use explicit post-training trainability mask; O05/O06 pretraining unfreeze/LR transitions are N/A and do not establish that vision is frozen.')
        if p in ('F1','F2'):
            ac.append('Audit actual DS4 algorithm/group/shape/LR/decay for inherited families; reject unsupported vision mapping or silent numeric defaults. Apply Engram e_proj/e_norm 5x only if trainable.')
        if p in ('E1e','E2','E4','F2'):
            ac.append('Verify native FP32 main_grad rather than widened BF16 gradients, and FP32 momentum; state dtype policy does not resolve table trainability.')
        # Planning ranges include test/qualification effort, not measured LOC promises.
        hours = ([2,6] if kind == 'S' else [8,24])
        if kind == 'I' and p in ('B1','C4','D2','E2','E3','E4','F3','G1'):
            hours = [24,64]
        if p == 'E2' and kind == 'I':
            hours = [24,48]
        task = dict(id=n, title=n+': '+(spec if kind == 'S' else impl), acceptance_criteria=ac,
                    dependencies=sorted(graph[n]), external_prerequisites=['Corrected A1/A3 and A2/A4 specifications named in plan section 0; verify content/ancestry before implementation'],
                    estimate=dict(engineer_hours=hours, basis='Planning range including independent fixtures, focused implementation and qualification; GPU queue excluded', conditional_loc='~1200 lines discussed for Sinkhorn; unvalidated estimate, contingent on trainability' if p=='E2' and kind=='I' else None),
                    direct_open=sorted(direct[n]), blocking_open=sorted(inherited[n]), blocked_by_open=bool(inherited[n]),
                    activation_note='Conservative full training gate; O15 activates only for requested FP4-off diagnostics. An inactive decision requires an explicit profile; it is not automatically resolved.',
                    constraints=constraints, deferred_gate=deferred, source='deepseek_v41_phases_bg_v4.md (row '+p+')')
        tasks.append(task)
    assert len(tasks) == 64
    return tasks


def render(tasks):
    lines = ['# Post-training work items', '', '64 task-ready records (32 S/I pairs), in topological order. Estimates are engineer hours, not elapsed-time promises. OPEN lists include transitive prerequisites; see each activation condition in the ledger. No scheduler nodes or implementations are certified by this list.', '']
    for t in tasks:
        lines += ['## '+t['title'], '', '- Dependencies: '+(', '.join(t['dependencies']) or 'none within B–G'), '- Direct OPEN: '+(', '.join(t['direct_open']) or 'none'), '- Full-gate OPEN (including prerequisites): '+(', '.join(t['blocking_open']) or 'none'), '- Estimated hours: '+'–'.join(map(str,t['estimate']['engineer_hours'])), '- External prerequisites: '+t['external_prerequisites'][0], '- Deferred acceptance: '+t['deferred_gate'], '', 'Acceptance:', '']
        lines += ['- [ ] '+a for a in t['acceptance_criteria']]
        lines += ['', 'Constraints:', '']+['- '+c for c in t['constraints']]+['']
    return '\n'.join(lines)


def verify(tasks):
    expected = build()
    assert tasks == expected, 'task content/dependency/blocker/estimate drift'
    seen = set()
    for t in tasks:
        assert t['id'] not in seen
        assert set(t['dependencies']) <= seen, 'not topologically ordered'
        assert len(t['acceptance_criteria']) >= 3
        seen.add(t['id'])
    assert len(seen) == 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    tasks = build()
    jp, mp = ROOT/'deepseek_v41_tasks.json', ROOT/'deepseek_v41_tasks.md'
    if args.check:
        verify(json.loads(jp.read_text()))
        assert mp.read_text() == render(tasks), 'rendered checklist drift'
        import copy
        for field, value in [('dependencies', []), ('blocking_open', []), ('acceptance_criteria', [])]:
            bad = copy.deepcopy(tasks)
            target = next(t for t in bad if t['dependencies'] and t['blocking_open'])
            target[field] = value
            try:
                verify(bad)
            except AssertionError:
                print('REJECTED: task '+field+' removed')
            else:
                raise AssertionError('mutation accepted')
        import subprocess
        decisions = json.loads((ROOT/'deepseek_v41_decisions.json').read_text())
        import sys
        command = [sys.executable, str(ROOT/'validate_deepseek_v41_plan.py')]
        active = [arg for d in decisions if d['status']=='RESOLVED' for arg in ('--active-decision', d['id'])]
        good = subprocess.run(command+active, capture_output=True, text=True)
        assert good.returncode == 0 and 'PLAN_STRUCTURE_OK' in good.stdout, good.stderr
        print('RESOLVED_GATE_OK: 11 active decisions; rc=0')
        for d in decisions:
            if d['status'] == 'OPEN':
                bad = subprocess.run(command+['--active-decision', d['id']], capture_output=True, text=True)
                assert bad.returncode == 1 and f"active decision {d['id']} remains OPEN" in bad.stderr
                print('OPEN_REJECTED: '+d['id']+'; rc=1')
        print('TASK_MANIFEST_OK: 64 ordered records, complete AC/estimates/dependencies, propagated OPEN blockers; 3 mutations rejected')
    else:
        jp.write_text(json.dumps(tasks, indent=2)+'\n')
        mp.write_text(render(tasks))


if __name__ == '__main__':
    main()

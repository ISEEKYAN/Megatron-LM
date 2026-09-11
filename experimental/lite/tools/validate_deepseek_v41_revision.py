"""Run executable revision checks and verify the unchanged 40-layer owner map."""
import ast
import argparse
import sys
from pathlib import Path
import subprocess
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-model', required=True)
    args = parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    for path,args in [(root/'docs/plans/validate_deepseek_v41_plan.py', []),(root/'tools/validate_deepseek_v41_ced.py',['--official-model',args.official_model])]:
        tree=ast.parse(path.read_text())
        guards=[n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'"]
        assert len(guards)==1 and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='main' for n in ast.walk(guards[0]))
        r=subprocess.run([sys.executable,str(path),*args],capture_output=True,text=True)
        print(r.stdout,end=''); print(f'ENTRYPOINT {path}: rc={r.returncode}')
        assert r.returncode==0 and r.stdout.strip(),r.stderr
    rows=[]
    for line in (root/'docs/deepseek_v41_owner_consumer.md').read_text().splitlines():
        cols=[c.strip() for c in line.split('|')[1:-1]]
        if cols and cols[0].isdigit(): rows.append(cols)
    assert len(rows)==40
    for i,cols in enumerate(rows):
        assert len(cols)==9 and int(cols[0])==i
        if i<2:
            assert cols[4:6]==['—','—']
        else:
            k=max(x for x in [2,8,14,20] if x<=i)
            t=max(x for x in [2,8,14,20,24,28,32,36] if x<=i)
            assert cols[4:6]==[str(k),str(t)]
            assert cols[6]==f'{i}→K{k}→θKV{k}; floating sum at owner'
            assert cols[7]==f'read T{t}; integer, no autograd'
            assert cols[8]==f'OPEN D6-S: contributor {i}, shared indexer owner {t}'
    print('A1_TABLE_OK: 40 unchanged owner assignments; separate floating/integer/auxiliary columns')
    r=subprocess.run([sys.executable,str(root/'docs/plans/validate_deepseek_v41_plan.py'),'--active-decision','O16'],capture_output=True,text=True)
    assert r.returncode!=0 and 'active decision O16 remains OPEN' in r.stderr
    print(f'ACTIVE_OPEN_REJECTED: O16 rc={r.returncode}')


if __name__ == "__main__":
    main()

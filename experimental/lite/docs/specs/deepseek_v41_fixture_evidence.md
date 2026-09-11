# Fixture specification verification

Scope: CPU specification arithmetic and pinned A8 decision validation only.
No generator, model, training step, codec or GPU acceptance was executed.

Run the following from the repository root with Python 3. It uses independent
scalar arithmetic and enumeration on the published cards, not MLite outputs.

```python
import hashlib
import itertools
import json
from pathlib import Path

v = json.loads(Path('experimental/lite/docs/specs/deepseek_v41_fixture_vectors.json').read_text())
c = v['cards']
for name, path in {
    'model.py': '/tmp/ds41-review/model.py',
    'config.json': '/tmp/ds41-review/config.json',
    'engram.py': '/tmp/ds41-review/engram.py',
    'index.json': '/tmp/v41_index.json',
}.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == v['source_sha256'][name]
x = c['dense_weight_recipe']
assert [((17*x['ordinal'] + 13*j + 7) % 101)-50 for j in x['flat_indices']] == x['expected_numerators_over_256']
x = c['ced_collapse_before_norm']
assert [sum(a*b[j] for a,b in zip(x['pre_mix'],x['hc_rows'])) for j in range(2)] == x['expected_collapsed']
x = c['owner_gradient_sum']
sums = {}
for row in x['contributions']:
    dst = sums.setdefault(str(row['owner']), [0,0])
    for j in range(2):
        dst[j] += row['gradient'][j]
assert sums == x['expected']
x = c['positions_ratio2']
assert [(p+1)//2 for p in x['queries']] == x['expected_visible_counts']
assert [2*j for j in range(4)] == x['first_four_group_rope_positions']
assert (3//2)*2 != 3 and not x['length3_trailing_token_visible_as_group']
x = c['swa_boundary']
assert [max(0,p-127) for p in x['queries']] == x['expected_first_inclusive']
assert x['queries'] == x['expected_last_inclusive']
x = c['ranking_reversal']
for who in ('source','reindex'):
    scores = [max(0,sum(a*b for a,b in zip(x[who+'_query'],key))) for key in x['keys']]
    assert scores == x[who+'_scores']
    assert sorted(range(3), key=lambda j:-scores[j])[:1] == x['expected_'+who+'_indices']
    gap = sorted(scores, reverse=True)[0]-sorted(scores, reverse=True)[1]
    assert gap >= x['minimum_cutoff_gap'] and gap > 2*x['score_error_bound']
x = c['tie_membership']
best_sum = max(sum(x['scores'][j] for j in ids) for ids in itertools.combinations(range(4),2))
allowed = [list(ids) for ids in itertools.combinations(range(4),2) if sum(x['scores'][j] for j in ids)==best_sum]
assert allowed == x['allowed_sorted_indices']
assert all(len(ids)==x['expected_cardinality'] and not set(ids)&set(x['forbidden_indices']) for ids in allowed)
x = c['candidate_newest_pin']
assert (x['visible_count']-1)//8 == x['expected_kept_blocks'][0]
assert list(range(16,17)) == x['expected_true_positions']
x = c['candidate_partial_mask']
last_block = (x['visible_count']-1)//x['block_size']
expanded = list(range(last_block*8, min((last_block+1)*8, x['width'])))
assert expanded == x['expected_candidate_true_positions']
assert [j for j in expanded if j<x['visible_count']] == x['expected_causally_valid_positions']
assert c['candidate_empty']['visible_count']==0 and c['candidate_empty']['expected_true_positions']==[]
# Full production cutoffs: forced newest block displaces the worst old block.
blocks = sorted(range(2049), key=lambda b: float('-inf') if b==2048 else -(4096-b))[:2048]
assert sorted(blocks)==list(range(2047))+[2048]
scores = [1024-j for j in range(513)]
assert sorted(range(513), key=lambda j:-scores[j])[:512]==list(range(512))
scores[512]=2048
assert sorted(sorted(range(513), key=lambda j:-scores[j])[:512])==list(range(511))+[512]
x = c['image_hc_copy']
assert [x['input_vector']]*4 == x['expected_copies']
assert [sum(x['upstream_per_copy'])]*2 == x['expected_input_gradient']
x = c['engram_dead_history']
for pos in (3,4):
    history=[]
    blocked=False
    for shift in range(4):
        i=pos-shift
        blocked = blocked or i<0 or x['compressed_tokens'][i]==-1
        history.append(x['pad_id'] if blocked else x['compressed_tokens'][i])
    assert history == x[f'expected_history_at_{pos}']
h=x['expected_history_at_3']; m=x['injected_odd_multipliers']; rolling=h[0]*m[0]; hashes=[]
for i in range(1,4):
    rolling ^= h[i]*m[i]
    hashes.append(rolling)
assert hashes == x['expected_xor_before_mod_at_3']
x = c['sinkhorn_inclusive_threshold']
rho=[abs(row[0]) for row in x['injected_N']]
assert sum(rho)/len(rho)==x['expected_mean_row_norm']
assert [i for i,r in enumerate(rho) if r*1000<=sum(rho)/len(rho)]==x['expected_masked_rows']
print('SPEC_CARD_ARITHMETIC_OK: 13 cards, production cutoffs, four source hashes')
```

Observed: exit 0 with the marker above. This verifies the hand-specified expected
values, not good-versus-mutated implementations; those remain B3-I obligations.

Decision validation command (A8 prerequisite checkout; substitute its location):

```sh
python "$A8/experimental/lite/docs/plans/validate_deepseek_v41_plan.py" --active-decision O01
```

Repeated independently for O01,O02,O03,O04,O07,O10,O11,O13,O14,O16,O17 at
A8 commit `766bc22d1396c30a6c7d08deabe64f0a56e84d8f`: each exit 0. Each run
prints the structural-only evidence disclaimer and executes the plan's negative
controls. O08,O09,O12 were also invoked individually: each exit 1 with
`active decision <ID> remains OPEN`. No active OPEN is silently accepted.

`git diff --check`: exit 0. Source paths at all five cited prerequisite commits
were checked using `git cat-file -e`/`git show`; the A7 branch remains readable
by commit although its former checkout is absent. No prerequisite files were
copied over the current branch or altered by this specification.

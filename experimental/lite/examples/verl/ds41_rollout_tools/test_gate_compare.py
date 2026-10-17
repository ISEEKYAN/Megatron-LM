"""CPU self-test of gate_compare.py."""

import json

import pytest
from gate_compare import compare, run_to_run


def _run(lp_shift=0.0, fp=(1, 2), versus=None, sketch_shift=0.0):
    lps = {0: [-1.0, -2.0], 1: [-0.5 + lp_shift]}
    grads = {
        'w': dict(fingerprint=list(fp), norm=2.0, sketch=[1.0 + sketch_shift, -1.0]),
        'b': dict(fingerprint=[3, 4], norm=1.0, sketch=[0.5, 0.5]),
    }
    for name, (rel, error) in (versus or {}).items():
        grads[name]['versus'] = dict(abs=error, rel=rel, max_reference=1.0)
    return lps, grads


CLEAN = {'a': dict(logprobs_unequal=[], grads_unequal=[])}


def test_identical_runs_pass_bitwise():
    assert compare(_run(), _run())['passed']


def test_bitwise_rejects_a_changed_fingerprint():
    assert not compare(_run(), _run(fp=(1, 9)))['passed']


def test_logprob_difference_always_fails():
    report = compare(_run(), _run(lp_shift=1e-7), 'cross-partition', repeats=CLEAN)
    assert not report['passed'] and report['logprobs']['unequal'] == [1]


def test_cross_partition_reports_gradient_errors_without_judging_them():
    b = _run(fp=(1, 9), versus={'w': (1e-2, 1e-3), 'b': (0.0, 0.0)})
    report = compare(_run(), b, 'cross-partition', repeats=CLEAN)
    assert report['passed']
    errors = report['errors']
    assert errors['kind'] == 'element-wise' and errors['beyond_parity_tolerance'] == 1
    assert errors['rel_distribution']['max'] == 1e-2
    assert set(errors['per_layer']) == {'w', 'b'}
    sketch = compare(
        _run(), _run(fp=(1, 9), sketch_shift=0.5), 'cross-partition', CLEAN
    )
    assert sketch['passed'] and sketch['errors']['kind'] == 'sketch'


def test_cross_partition_needs_clean_repeats_on_both_sides():
    b = _run(fp=(1, 9))
    assert not compare(_run(), b, 'cross-partition', repeats=None)['passed']
    dirty = {'a': dict(logprobs_unequal=[], grads_unequal=['w'])}
    assert not compare(_run(), b, 'cross-partition', repeats=dirty)['passed']


def test_run_to_run_reads_the_in_process_repeat(tmp_path):
    for suffix, fp in (('', (1, 2)), ('-repeat1', (1, 9))):
        lps, grads = _run(fp=fp)
        (tmp_path / f'logprobs{suffix}.jsonl').write_text(
            '\n'.join(json.dumps(dict(sequence=k, logprobs=v)) for k, v in lps.items())
        )
        (tmp_path / f'grads{suffix}.json').write_text(json.dumps(grads))
    report = run_to_run([tmp_path])
    assert report[str(tmp_path)]['grads_unequal'] == ['w']
    assert run_to_run([tmp_path / 'missing']) is None


def test_parameter_sets_must_match():
    a = _run()
    b = _run()
    del b[1]['b']
    with pytest.raises(SystemExit):
        compare(a, b)

"""CPU self-test of gate_compare.py."""

import json

import pytest
from gate_compare import compare, run_to_run


def _run(lp_shift=0.0, fp=(1, 2), versus=None):
    lps = {0: [-1.0, -2.0], 1: [-0.5 + lp_shift]}
    grads = {
        'w': dict(fingerprint=list(fp), norm=2.0, sketch=[1.0, -1.0]),
        'b': dict(fingerprint=[3, 4], norm=1.0, sketch=[0.5, 0.5]),
    }
    for name, (rel, error) in (versus or {}).items():
        grads[name]['versus'] = dict(abs=error, rel=rel, max_reference=1.0)
    return lps, grads


SMALL = {'w': (1e-6, 1e-6), 'b': (0.0, 0.0)}
CLEAN = {'a': dict(logprobs_unequal=[], grads_unequal=[])}


def test_identical_runs_pass_bitwise():
    assert compare(_run(), _run())['passed']


def test_logprob_difference_always_fails():
    report = compare(
        _run(), _run(lp_shift=1e-7, versus=SMALL), 'tolerance', repeats=CLEAN
    )
    assert not report['passed'] and report['logprobs']['unequal'] == [1]


def test_bitwise_rejects_a_changed_fingerprint():
    assert not compare(_run(), _run(fp=(1, 9)))['passed']


def test_tolerance_is_relative_or_absolute():
    a = _run()
    # rel 1e-6 passes; abs 1e-7 passes even at rel 1e-3 (a small, cancelling sum).
    for versus in (SMALL, {'w': (1e-3, 1e-7), 'b': (0.0, 0.0)}):
        b = _run(fp=(1, 9), versus=versus)
        assert compare(a, b, 'tolerance', repeats=CLEAN)['passed']
    b = _run(fp=(1, 9), versus={'w': (1e-3, 1e-5), 'b': (0.0, 0.0)})
    report = compare(a, b, 'tolerance', repeats=CLEAN)
    assert not report['passed'] and report['tolerance']['outliers'] == 1


def test_tolerance_needs_every_parameter_measured_and_clean_repeats():
    a = _run()
    partial = _run(versus={'w': (0.0, 0.0)})
    assert not compare(a, partial, 'tolerance', repeats=CLEAN)['passed']
    b = _run(versus=SMALL)
    assert not compare(a, b, 'tolerance', repeats=None)['passed']
    dirty = {'a': dict(logprobs_unequal=[], grads_unequal=['w'])}
    assert not compare(a, b, 'tolerance', repeats=dirty)['passed']


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

"""CPU self-test of gate_compare.py."""

import pytest
from gate_compare import compare


def _run(lp_shift=0.0, fp=(1, 2), sketch_shift=0.0):
    lps = {0: [-1.0, -2.0], 1: [-0.5 + lp_shift]}
    grads = {
        'w': dict(fingerprint=list(fp), norm=2.0, sketch=[1.0 + sketch_shift, -1.0]),
        'b': dict(fingerprint=[3, 4], norm=1.0, sketch=[0.5, 0.5]),
    }
    return lps, grads


def test_identical_runs_pass_bitwise():
    assert compare(_run(), _run())['passed']


def test_logprob_difference_always_fails():
    report = compare(_run(), _run(lp_shift=1e-7), grads='sketch')
    assert not report['passed'] and report['logprobs']['unequal'] == [1]


def test_bitwise_mode_rejects_a_changed_fingerprint_sketch_mode_bounds_it():
    a, b = _run(), _run(fp=(1, 9), sketch_shift=1e-6)
    assert not compare(a, b, grads='bitwise')['passed']
    assert compare(a, b, grads='sketch', rel=1e-4)['passed']
    assert not compare(a, _run(fp=(1, 9), sketch_shift=1.0), grads='sketch')['passed']


def test_parameter_sets_must_match():
    a = _run()
    b = _run()
    del b[1]['b']
    with pytest.raises(SystemExit):
        compare(a, b)

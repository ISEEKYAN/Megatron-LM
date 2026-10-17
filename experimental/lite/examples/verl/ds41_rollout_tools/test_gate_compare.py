"""CPU self-test of gate_compare.py."""

import json

import pytest
from gate_compare import compare, gradients_finite

FINITE = {'step-0': dict(checked=3, nonfinite=[])}


def _lp(shift=0.0):
    return {0: [-1.0, -2.0], 1: [-0.5 + shift]}


def test_equal_logprobs_and_finite_gradients_pass():
    assert compare(_lp(), _lp(), FINITE, FINITE)['passed']


def test_any_logprob_difference_fails():
    report = compare(_lp(), _lp(1e-7), FINITE, FINITE)
    assert not report['passed'] and report['logprobs']['unequal'] == [1]


def test_nonfinite_or_unchecked_gradients_fail():
    bad = {'step-0': dict(checked=3, nonfinite=['w'])}
    assert not compare(_lp(), _lp(), FINITE, bad)['passed']
    unchecked = {'step-0': dict(checked=0, nonfinite=[])}
    assert not compare(_lp(), _lp(), unchecked, FINITE)['passed']
    assert not compare(_lp(), _lp(), {}, FINITE)['passed']


def test_gradients_finite_reads_every_step_and_flags_old_receipts():
    record = {
        'world': 4,
        'step-0': dict(gradients_checked=5, gradients_nonfinite=[]),
        'step-repeat1': dict(gradients_checked=5, gradients_nonfinite=['b']),
        'step-old': dict(step_s=1.0),
    }
    steps = gradients_finite(json.loads(json.dumps(record)))
    assert steps['step-0'] == dict(checked=5, nonfinite=[])
    assert steps['step-repeat1']['nonfinite'] == ['b']
    assert steps['step-old']['nonfinite'] == ['<not recorded>']


def test_sequence_sets_must_match():
    with pytest.raises(SystemExit):
        compare(_lp(), {0: [-1.0, -2.0]}, FINITE, FINITE)

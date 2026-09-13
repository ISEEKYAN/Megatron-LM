"""Explicit BF16 dgrad arithmetic, independent of tested TP tensors.

MCore 0cd11658f tensor_parallel/layers.py:557,569-571 returns a matmul
in the input dtype and all-reduces that tensor without promoting to FP32.
This is a bitwise reference contract, not a numerical tolerance.
"""

import torch


def bf16_ordered_sum(partials):
    values = [value.to(torch.bfloat16) for value in partials]
    assert values, 'TP_DGRAD_PARTIALS_REQUIRED'
    result = values[0].clone()
    for value in values[1:]:
        result = result + value
    return result


def independent_partials(dy, full_weight, size=2):
    assert dy.dtype == full_weight.dtype == torch.bfloat16, 'TP_DGRAD_DTYPE'
    return [
        grad.contiguous().matmul(weight)
        for grad, weight in zip(
            dy.chunk(size, -1), full_weight.chunk(size, 0), strict=True
        )
    ]


def explained_difference(actual, modeled, serial):
    if isinstance(actual, torch.Tensor):
        return torch.equal(
            actual.double() - serial.double(), modeled.double() - serial.double()
        )
    return actual - serial == modeled - serial


def assert_dgrad_reference(actual, modeled, serial):
    assert actual.dtype == modeled.dtype and torch.equal(
        actual, modeled
    ), 'TP_DGRAD_BF16_MODEL'
    assert explained_difference(actual, modeled, serial), 'TP_DGRAD_SERIAL_EXPLAINED'

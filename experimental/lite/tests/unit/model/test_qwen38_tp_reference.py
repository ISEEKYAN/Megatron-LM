import importlib.util
from pathlib import Path

import pytest
import torch


def reference_module():
    path = Path(__file__).parents[2] / 'smoke/workflows/training/qwen38_tp_reference.py'
    spec = importlib.util.spec_from_file_location('qwen38_tp_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bf16_partial_rounding_is_not_rounding_only_after_sum():
    ref = reference_module()
    parts = [torch.tensor([1.00390625]), torch.tensor([0.00390625])]
    modeled = ref.bf16_ordered_sum(parts)
    unrounded = sum(parts).bfloat16()
    assert not torch.equal(modeled, unrounded)
    ref.assert_dgrad_reference(modeled, modeled, torch.tensor([1.0]))
    with pytest.raises(AssertionError, match='TP_DGRAD_BF16_MODEL'):
        ref.assert_dgrad_reference(unrounded, modeled, torch.tensor([1.0]))


def test_serial_delta_rejects_a_second_error_source():
    ref = reference_module()
    assert not ref.explained_difference(
        torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([0.5])
    )
    assert ref.explained_difference(
        torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([0.5])
    )

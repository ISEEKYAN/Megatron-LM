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


def test_three_addend_order_diagnostic_is_distinct_from_tp2_training():
    # Three effective addends are required to expose non-associativity.
    ref = reference_module()
    parts = [torch.tensor([256.0]), torch.tensor([1.0]), torch.tensor([-256.0])]
    modeled = ref.bf16_ordered_sum(parts)
    reordered = ref.bf16_ordered_sum([parts[0], parts[2], parts[1]])
    assert modeled.item() == 0 and reordered.item() == 1
    ref.assert_dgrad_reference(modeled, modeled, torch.tensor([1.0]))
    with pytest.raises(AssertionError, match='TP_DGRAD_BF16_MODEL'):
        ref.assert_dgrad_reference(reordered, modeled, torch.tensor([1.0]))


def test_norm_reference_uses_full_reference_and_owner_ranges():
    from types import SimpleNamespace

    ref = reference_module()
    model = torch.nn.Module()
    model.lm_head = torch.nn.Module()
    model.lm_head.linear = torch.nn.Linear(1, 2, bias=False)
    model.embed_tokens = torch.nn.Embedding(2, 1)
    projection, replica = model.lm_head.linear.weight, model.embed_tokens.weight
    main_projection = torch.nn.Parameter(torch.tensor([-999.0]))
    main_replica = torch.nn.Parameter(torch.tensor([-999.0, -999.0]))
    main_projection.grad = torch.tensor([-999.0])
    main_replica.grad = torch.tensor([-999.0, -999.0])
    optimizer = SimpleNamespace(
        model_float16_groups=[[projection, replica]],
        shard_fp32_from_float16_groups=[[main_projection, main_replica]],
        model_fp32_groups=[],
        shard_fp32_groups=[],
        get_parameters=lambda: [main_projection, main_replica],
        _get_model_param_range_map=lambda p: {
            'param': SimpleNamespace(start=0, end=1 if p is projection else 2)
        },
    )
    complete = {
        'lm_head.weight': torch.tensor([[3.0], [4.0], [6.0], [8.0]]),
        'embed_tokens.weight': torch.tensor([[5.0], [7.0]]),
    }
    tensors, layout = ref.norm_reference_inputs(
        optimizer, model, complete, rank=1, world=2
    )
    assert len(tensors) == 1 and torch.equal(
        tensors[0], torch.tensor([6.0])
    ), 'TP_NORM_REFERENCE_FROM_SERIAL'
    assert layout[0]['name'] == 'lm_head.linear.weight'

"""CPU checks for the narrowly approved EP proxy validation contract."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / 'smoke/workflows/training')
)


def test_nonexpert_comparison_remains_bitwise():
    from qwen38_ep_probe import compare_tensor

    with pytest.raises(AssertionError, match='EP_NONEXPERT_BITWISE'):
        compare_tensor(
            'lm_head.weight', torch.tensor([1e-12]), torch.zeros(1), 'gradient'
        )


@pytest.mark.parametrize(
    'delta,reference,accepted',
    [(1e-8, 1e-3, True), (3e-8, 1e-3, False), (1e-8, 1e-6, False)],
)
def test_expert_gradient_requires_both_bounds(delta, reference, accepted):
    from qwen38_ep_probe import compare_tensor

    ref = torch.full((4, 4), reference)
    if accepted:
        compare_tensor('layers.0.mlp.experts.fc1.weight0', ref + delta, ref, 'gradient')
    else:
        with pytest.raises(AssertionError, match='EP_EXPERT_GRADIENT_BOUND'):
            compare_tensor(
                'layers.0.mlp.experts.fc1.weight0', ref + delta, ref, 'gradient'
            )


@pytest.mark.parametrize(
    'count,value,accepted', [(3, 2**-14, True), (4, 2**-14, False), (1, 2**-13, False)]
)
def test_expert_parameter_requires_magnitude_and_coordinate_count(
    count, value, accepted
):
    from qwen38_ep_probe import compare_tensor

    ref = torch.zeros(4, 4, dtype=torch.bfloat16)
    actual = ref.clone()
    actual.flatten()[:count] = value
    if accepted:
        compare_tensor('layers.0.mlp.experts.fc1.weight0', actual, ref, 'parameter')
    else:
        with pytest.raises(AssertionError, match='EP_EXPERT_PARAMETER_BOUND'):
            compare_tensor('layers.0.mlp.experts.fc1.weight0', actual, ref, 'parameter')


def _scope():
    from dataclasses import asdict

    from megatron.lite.model.qwen3_8_flash_next.protocol import build_model_config
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.contracts import OptimizerConfig
    from test_qwen38_training import tiny_training_config

    return dict(
        model_config=asdict(build_model_config(tiny_training_config())),
        runtime_config=asdict(
            MegatronLiteConfig(
                model_name='qwen3_8_flash_next',
                load_hf_weights=False,
                optimizer=OptimizerConfig(lr=0.003),
                impl_cfg={
                    'ngram_primes': (
                        17,
                        19,
                        23,
                        29,
                        31,
                        37,
                        41,
                        43,
                        47,
                        53,
                        59,
                        61,
                        67,
                        71,
                        73,
                        79,
                    )
                },
            )
        ),
        documents=[
            [i % 32 for i in range(64)],
            [(i * 3 + 47) % 127 for i in range(64)],
        ],
        seed=1234,
        steps=3,
        world=1,
        dimensions={
            'tp': 1,
            'etp': 1,
            'cp': 1,
            'pp': 1,
            'ep': 1,
            'dp': 1,
            'expert_dp': 1,
        },
        env={
            'torch': '2.12.0a0+5aff3928d8.nv26.05',
            'cuda': '13.2',
            'te': '2.15.0+42b84005',
            'cudnn': 92200,
            'gpu': 'NVIDIA H100 80GB HBM3',
            'capability': [9, 0],
            'fla': False,
            'tf32_matmul': True,
            'tf32_cudnn': True,
            'deterministic': False,
        },
    )


def test_scope_rejects_every_changed_model_and_runtime_field():
    from copy import deepcopy

    from qwen38_ep_probe import validate_scope

    valid = _scope()
    validate_scope(**valid)
    for group in ('model_config', 'runtime_config'):
        for field in valid[group]:
            if field == 'hf_path':
                continue
            changed = deepcopy(valid)
            changed[group][field] = 'out-of-approved-scope'
            with pytest.raises(AssertionError, match='EP_SCOPE'):
                validate_scope(**changed)


@pytest.mark.parametrize(
    'field,value',
    [
        ('seed', 1235),
        ('steps', 4),
        ('world', 4),
        ('documents', [[1] * 64, [2] * 64]),
        ('dimensions', {'ep': 2}),
        ('env', {'gpu': 'other'}),
    ],
)
def test_scope_rejects_training_data_plan_and_environment(field, value):
    from qwen38_ep_probe import validate_scope

    scope = _scope()
    scope[field] = value
    with pytest.raises(AssertionError, match='EP_SCOPE'):
        validate_scope(**scope)


def test_zero_reference_gradient_never_uses_an_epsilon():
    from qwen38_ep_probe import compare_tensor

    with pytest.raises(AssertionError, match='EP_EXPERT_GRADIENT_BOUND'):
        compare_tensor(
            'layers.0.mlp.experts.fc1.weight0',
            torch.tensor([1e-12]),
            torch.zeros(1),
            'gradient',
        )
    compare_tensor(
        'layers.0.mlp.experts.fc1.weight0', torch.zeros(1), torch.zeros(1), 'gradient'
    )

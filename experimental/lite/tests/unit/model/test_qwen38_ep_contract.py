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


def _oracle_fixture(rank=0, world=1):
    """Compute expected wgrad by independent per-token outer products."""
    reference = {'wgrad_calls': {}, 'gradients': {}}
    for layer in range(2):
        for fc in ('fc1', 'fc2'):
            key = f'{layer}.{fc}'
            calls = []
            for document in range(2):
                splits = [expert + document + 1 for expert in range(4)]
                size = sum(splits)
                x = (
                    (torch.arange(size * 3).reshape(size, 3) % 13 + 1) / 1024
                ).bfloat16()
                dy = (
                    (torch.arange(size * 2).reshape(size, 2) % 7 - 3) / 1024
                ).bfloat16()
                calls.append({'x': x, 'dy': dy, 'splits': splits})
            reference['wgrad_calls'][key] = calls
            for expert in range(4):
                expected = torch.zeros(2, 3, dtype=torch.float64)
                for call in calls:
                    start = sum(call['splits'][:expert])
                    for token in range(start, start + call['splits'][expert]):
                        expected += torch.outer(
                            call['dy'][token].double(), call['x'][token].double()
                        )
                reference['gradients'][
                    f'layers.{layer}.mlp.experts.{fc}.weight{expert}'
                ] = expected.float()
    if world == 1:
        return reference['wgrad_calls'], reference['gradients'], reference
    local_calls, local_gradients = {}, {}
    for key, calls in reference['wgrad_calls'].items():
        layer, fc = key.split('.')
        xs, dys, splits = [], [], []
        for local in range(2):
            expert = rank * 2 + local
            x = torch.cat([c['x'].split(c['splits'])[expert] for c in calls])
            dy = torch.cat([c['dy'].split(c['splits'])[expert] for c in calls])
            xs.append(x)
            dys.append(dy * 2)
            splits.append(len(x))
            local_gradients[f'layers.{layer}.mlp.experts.{fc}.weight{local}'] = (
                reference['gradients'][
                    f'layers.{layer}.mlp.experts.{fc}.weight{expert}'
                ].clone()
            )
        local_calls[key] = [
            {'x': torch.cat(xs), 'dy': torch.cat(dys), 'splits': splits}
        ]
    return local_calls, local_gradients, reference


@pytest.mark.parametrize('rank,world', [(0, 1), (0, 2), (1, 2)])
def test_fp64_oracle_covers_all_branches_and_expert_owners(rank, world):
    from qwen38_ep_probe import check_wgrad_oracle

    calls, gradients, reference = _oracle_fixture(rank, world)
    errors = check_wgrad_oracle(calls, gradients, reference, rank, world)
    assert len(errors) == 16 // world
    assert max(errors.values()) == 0


@pytest.mark.parametrize(
    'mutation,tag',
    [
        ('branch', 'EP_ORACLE_BRANCHES'),
        ('microbatch', 'EP_ORACLE_MICROBATCHES'),
        ('x', 'EP_ORACLE_INPUT'),
        ('dy', 'EP_ORACLE_DY'),
        ('reference_gradient', 'EP_REFERENCE_ORACLE_ERROR'),
        ('gradient', 'EP_FP32_ORACLE_ERROR'),
    ],
)
def test_fp64_oracle_rejects_corrupted_actual_signals(mutation, tag):
    from qwen38_ep_probe import check_wgrad_oracle

    calls, gradients, reference = _oracle_fixture(0, 2)
    name = 'layers.0.mlp.experts.fc1.weight0'
    if mutation == 'branch':
        del calls['1.fc2']
    elif mutation == 'microbatch':
        calls['0.fc1'].append(calls['0.fc1'][0])
    elif mutation in ('x', 'dy'):
        calls['0.fc1'][0][mutation][0, 0] += 1 / 1024
    elif mutation == 'reference_gradient':
        reference['gradients'][name][0, 0] += 1e-7
    else:
        gradients[name][0, 0] += 1e-7
    with pytest.raises(AssertionError, match=tag):
        check_wgrad_oracle(calls, gradients, reference, 0, 2)


def test_environment_survives_restricted_reference_loading(monkeypatch, tmp_path):
    import os
    from types import ModuleType

    from qwen38_ep_probe import environment

    te = ModuleType('transformer_engine')
    te.__version__ = '2.15.0+42b84005'
    gdn = ModuleType('megatron.lite.primitive.modules.gated_delta_net')
    gdn._HAS_FLA = False
    monkeypatch.setitem(sys.modules, 'transformer_engine', te)
    monkeypatch.setitem(sys.modules, gdn.__name__, gdn)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'NVIDIA H100 80GB HBM3')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device=None: (9, 0))
    for key in os.environ:
        if key.startswith(('NVTE_', 'MEGATRON_LITE_', 'MLITE_', 'FLA_', 'CUBLAS_')):
            monkeypatch.delenv(key)
    for key, value in {
        'CUBLAS_VERSION': '13.4.1.1',
        'NVTE_FLASH_ATTN': '1',
        'NVTE_FUSED_ATTN': '0',
        'NVTE_UNFUSED_ATTN': '0',
    }.items():
        monkeypatch.setenv(key, value)
    observed = environment()
    artifact = tmp_path / 'reference.pt'
    torch.save({'scope': {'environment': observed}}, artifact)
    restored = torch.load(artifact, weights_only=True)
    assert restored['scope']['environment'] == observed

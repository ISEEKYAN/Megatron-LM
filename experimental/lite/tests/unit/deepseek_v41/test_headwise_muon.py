# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent multi-head update and live attention optimizer routing checks."""

from copy import deepcopy

import pytest
import torch
from megatron.lite.primitive.optimizers.headwise_muon import HeadwiseMuon


def reference_direction(matrix):
    # Independent FP64 polynomial expansion: no production helper/constants.
    # Quintic schedule from Emerging Optimizers' published Muon recipe.
    coefficients = (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    )
    x = matrix.double()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    # Fixtures are nonzero and well above either backend normalization epsilon.
    x = x / torch.linalg.vector_norm(x)
    for a, b, c in coefficients:
        cubic = x @ x.T @ x
        quintic = cubic @ x.T @ x
        x = a * x + b * cubic + c * quintic
    x = x.T if transposed else x
    return 0.18 * x / x.square().mean().sqrt()


def check_independent_head_updates(rows, columns, restore):
    # Different directions AND scales; the heads are not scalar multiples.
    grid = torch.arange(1, 2 * rows * columns + 1).reshape(2, rows, columns)
    gradients = (
        (grid.float().sin() + grid.float().cos() / 3),
        (grid.float().cos() * 2 - grid.float().sin() / 5),
        (grid.float().sin() * -3 + grid.float().cos()),
    )
    p = torch.nn.Parameter((grid.float() / 17).reshape(2 * rows, columns))
    expected = p.detach().double().reshape(2, rows, columns).clone()
    momentum = torch.zeros_like(expected)

    def build():
        return HeadwiseMuon(
            [{'params': [p], 'matrix_shape': (2, rows, columns)}],
            lr=0.03,
            weight_decay=0.1,
            momentum=0.95,
            update_rms=0.18,
            ns_steps=5,
            coefficient_type='quintic',
        )

    optimizer = build()
    for step, gradient in enumerate(gradients):
        p.grad = gradient.reshape_as(p).clone()
        for head in range(2):
            momentum[head] = 0.95 * momentum[head] + 0.05 * gradient[head].double()
            nesterov = 0.95 * momentum[head] + 0.05 * gradient[head].double()
            expected[head] = expected[head] * (
                1 - 0.03 * 0.1
            ) - 0.03 * reference_direction(nesterov)
        assert optimizer.step(), 'HEADWISE_STEP_MUST_COMMIT'
        for head in range(2):
            torch.testing.assert_close(
                p.detach().reshape_as(expected)[head].double(),
                expected[head],
                atol=2e-6,
                rtol=2e-6,
                msg=f'HEADWISE_INDEPENDENT_UPDATE head={head} step={step}',
            )
        torch.testing.assert_close(
            optimizer.state[p]['momentum_buffer'].reshape_as(momentum).double(),
            momentum,
            atol=2e-7,
            rtol=2e-6,
            msg='HEADWISE_MOMENTUM',
        )
        if restore and step == 0:
            saved = deepcopy(optimizer.state_dict())
            optimizer = build()
            optimizer.load_state_dict(saved)


@pytest.mark.parametrize('rows,columns', [(3, 4), (5, 2)])
@pytest.mark.parametrize('restore', [False, True])
def test_main_wq_b_independent_head_updates(rows, columns, restore):
    check_independent_head_updates(rows, columns, restore)


def always_vanilla(prepare_step):
    """Mutation: ignore logical heads and orthogonalize each whole owner."""

    def wrong(self):
        saved = [
            (g, g['matrix_shape'], g.get('matrix_partitions'))
            for g in self.param_groups
        ]
        try:
            for group, _, _ in saved:
                group['matrix_shape'] = tuple(group['params'][0].shape)
                group['matrix_partitions'] = None
            return prepare_step(self)
        finally:
            for group, shape, partitions in saved:
                group['matrix_shape'] = shape
                group['matrix_partitions'] = partitions

    return wrong


@pytest.mark.parametrize('rows,columns', [(3, 4), (5, 2)])
def test_always_vanilla_is_rejected_by_independent_head_assertion(
    monkeypatch, rows, columns
):
    monkeypatch.setattr(
        HeadwiseMuon, 'prepare_step', always_vanilla(HeadwiseMuon.prepare_step)
    )
    with pytest.raises(
        AssertionError, match='HEADWISE_INDEPENDENT_UPDATE head=0 step=0'
    ):
        check_independent_head_updates(rows, columns, restore=False)


@pytest.mark.parametrize('role', ['wq_b', 'wq_a', 'wkv'])
def test_live_main_attention_groups_keep_shared_matrices(
    build_bundle, model_config, role
):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig

    release = model_config.to_hf_dict()
    # Keep all 64 query heads and 40 layers; shrink matrix widths only.
    release['text_config'].update(num_attention_heads=64, q_lora_rank=64)
    bundle = build_bundle(
        DeepseekV41Config(release),
        text_only=True,
        optimizer='muon',
        optimizer_config=OptimizerConfig(0.03, 5, 'quintic'),
    )
    model = bundle.chunks[0]
    backend = next(
        o for o in bundle.optimizer.optimizers if isinstance(o, HeadwiseMuon)
    )
    groups = {id(p): g for g in backend.param_groups for p in g['params']}
    expected_shape = {'wq_b': (64, 32, 64), 'wq_a': (64, 32), 'wkv': (32, 32)}[role]
    for block in model.layers:
        p = getattr(block.attn, role).weight
        group = groups[id(p)]
        assert group['matrix_shape'] == expected_shape, f'{role}: LOGICAL_MATRIX_LAYOUT'
        assert group['matrix_partitions'] is None, f'{role}: NO_EXTRA_HEAD_PARTITIONS'
    p = getattr(model.layers[0].attn, role).weight
    p.main_grad = torch.linspace(-1, 2, p.numel()).reshape_as(p)
    calls = []
    orthogonalize = backend._orthogonalize

    def observe(matrix, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return orthogonalize(matrix, *args, **kwargs)

    backend._orthogonalize = observe
    assert backend.prepare_step(), 'LIVE_MUON_PREPARE'
    count = 64 if role == 'wq_b' else 1
    assert calls == [expected_shape[-2:]] * count, f'{role}: INDEPENDENT_MATRIX_CALLS'
    assert len(backend.candidates()) == 1, 'ONLY_THE_OWNER_WITH_A_GRADIENT'
    backend.discard_step()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Published optimizer state survives owner residency changes exactly."""
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer


def build(device):
    model = torch.nn.ParameterDict(
        {
            'matrix': torch.nn.Parameter(
                torch.linspace(-0.3, 0.4, 16, device=device).reshape(4, 4)
            ),
            'rows': torch.nn.Parameter(
                torch.linspace(-0.2, 0.5, 12, device=device).reshape(3, 4)
            ),
            'norm': torch.nn.Parameter(torch.linspace(0.1, 0.4, 4, device=device)),
        }
    )
    config = SimpleNamespace(
        lr=1e-3, clip_grad=1.0, ns_steps=2, coefficient_type='quintic'
    )

    def groups():
        return [
            dict(
                params=[model[name]],
                algorithm=algorithm,
                owner_key=name,
                matrix_shape=tuple(model[name].shape),
                weight_decay=0.1,
            )
            for name, algorithm in [
                ('matrix', 'muon'),
                ('rows', 'sinkhorn'),
                ('norm', 'adamw'),
            ]
        ]

    optimizer = MixedOptimizer(
        model,
        config,
        group_builder=groups,
        owners=lambda: ([], [], [], None),
        stats_factory=SimpleNamespace,
    )
    return model, optimizer


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def step(model, optimizer):
    optimizer.zero_grad()
    sum(
        (index + 1) * parameter.square().sum()
        for index, parameter in enumerate(model.parameters())
    ).backward()
    result = optimizer.step()
    assert result[0] and torch.isfinite(torch.tensor(result[1]))


def roundtrip(device):
    model, optimizer = build(device)
    reference, resident = build(device)
    optimizer.offload_state_to_cpu()
    optimizer.load_state_to_device()
    assert all(not backend.state for backend in optimizer.optimizers)
    for _ in range(2):
        step(model, optimizer)
        step(reference, resident)
        equal(optimizer.state_dict(), resident.state_dict())
        gradients = [p.grad.clone() for p in model.parameters()]
        optimizer.offload_state_to_cpu()
        for p, grad in zip(model.parameters(), gradients, strict=True):
            assert torch.equal(p.grad, grad) and p.grad.device == grad.device
        assert all(
            value.device.type == 'cpu'
            for backend in optimizer.optimizers
            for state in backend.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
        equal(optimizer.state_dict(), resident.state_dict())
        optimizer.zero_grad()
        parameter_ids = [id(p) for p in model.parameters()]
        model.to('cpu')
        model.to(device)
        assert [id(p) for p in model.parameters()] == parameter_ids
        optimizer.load_state_to_device()
        for backend in optimizer.optimizers:
            for p, state in backend.state.items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        expected = torch.device('cpu') if key == 'step' else p.device
                        assert value.device == expected
        equal(optimizer.state_dict(), resident.state_dict())
        for a, b in zip(model.parameters(), reference.parameters(), strict=True):
            assert torch.equal(a, b)


def test_cpu_two_real_updates_and_empty_state_roundtrip():
    roundtrip('cpu')


@pytest.mark.parametrize('algorithm', ['muon', 'sinkhorn'])
@pytest.mark.parametrize('method', ['offload_state_to_cpu', 'load_state_to_device'])
def test_prepared_transaction_rejects_residency_change(algorithm, method):
    model, optimizer = build('cpu')
    step(model, optimizer)
    snapshot = optimizer.state_dict()
    backend = next(
        o for o in optimizer.optimizers if o.param_groups[0]['algorithm'] == algorithm
    )
    assert backend.prepare_step()
    prepared = backend._prepared
    with pytest.raises(RuntimeError, match='prepared optimizer step'):
        getattr(optimizer, method)()
    assert backend._prepared is prepared
    backend.discard_step()
    equal(snapshot, optimizer.state_dict())


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='actual CUDA residency required'
)
def test_cuda_two_updates_equal_resident_optimizer_after_cpu_roundtrip():
    roundtrip('cuda')

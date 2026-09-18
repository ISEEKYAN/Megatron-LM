# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.primitive.modules.paired_payload import PairedPayload


def test_payload_roundtrip_preserves_gradient_owners():
    h = torch.randn(1, 3, 2, 4, requires_grad=True)
    p = torch.randn(1, 3, 2, requires_grad=True)
    kv = torch.randn(3, 4, requires_grad=True)
    positions = torch.arange(3)
    payload = PairedPayload(h, p, kv=kv, positions=positions)
    restored = PairedPayload.from_tensors(payload.tensors())
    assert restored.h is h and restored.p is p and restored.kv is kv
    assert restored.positions is positions
    assert set(restored.differentiable()) == {'h', 'p', 'kv'}
    loss = (restored.h * restored.p.unsqueeze(-1)).sum() + restored.kv.sum()
    gradients = torch.autograd.grad(loss, (h, p, kv))
    for actual, expected in zip(
        gradients, (p.unsqueeze(-1).expand_as(h), h.sum(-1), torch.ones_like(kv))
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    'damage', ['pair', 'shape', 'bytes_grad', 'positions', 'arity']
)
def test_payload_rejects_invalid_transport(damage):
    h, p = torch.zeros(1, 3, 2, 4), torch.zeros(1, 3, 2)
    changes = {
        'pair': {'ced_h': h},
        'shape': {'p': p[..., :1]},
        'bytes_grad': {'kv_values': torch.ones(3, 4, requires_grad=True)},
        'positions': {'positions': torch.ones(3)},
    }
    with pytest.raises(ValueError):
        if damage == 'arity':
            PairedPayload.from_tensors((h, p))
        else:
            PairedPayload(**({'h': h, 'p': p} | changes[damage]))

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU controls for slot arithmetic/score derivatives, not native MegaMoE parity."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.moe import combine
from megatron.lite.model.deepseek_v41.vllm.routing import _Route, route


def test_shared_first_slot_order_has_an_observable_negative_control():
    shared = torch.tensor([[1.0]], requires_grad=True)
    slots = torch.tensor([[[1e8], [-1e8], [7.0]]], requires_grad=True)
    ids = torch.tensor([[0, 1, -1]])
    result = combine(slots, ids, shared)
    # shared + slot0 + slot1 rounds to zero; shared added last is one.
    assert result.item() == 0
    assert (slots[:, 0].float() + slots[:, 1].float() + shared).item() == 1
    result.sum().backward()
    assert shared.grad.item() == 1
    assert slots.grad.flatten().tolist() == [1, 1, 0]


@pytest.mark.parametrize("rows", [0, 3])
def test_fixed_slot_score_vjp_matches_normalized_sqrtsoftplus(rows):
    logits = torch.randn(rows, 8, requires_grad=True)
    ids = torch.tensor([[6, 1, 3]]).expand(rows, -1).int()
    scores = torch.nn.functional.softplus(logits)
    selected = scores.sqrt().gather(1, ids.long())
    target = selected / selected.sum(-1, keepdim=True) * 1.7
    grad = torch.randn_like(target)
    (expected,) = torch.autograd.grad(target, logits, grad)
    (actual, *_) = _Route.backward(
        SimpleNamespace(saved_tensors=(logits, ids), factor=1.7), grad, None
    )
    torch.testing.assert_close(actual, expected)


def test_hash_routing_requires_real_ids_before_native_kernel_import():
    router = SimpleNamespace(tid2eid=torch.zeros(2, 1), is_hash_layer=True)
    with pytest.raises(ValueError, match='actual packed'):
        route(torch.zeros(1, 2), router, None)


@pytest.mark.parametrize("rows", [0, 3])
def test_activation_decode_accepts_empty_chunks(rows):
    """MegaMoE runs the same chunk count on every EP rank, so a rank's last chunk may be empty."""
    from megatron.lite.model.deepseek_v41.vllm.experts import dequantize_activation

    scale = torch.exp2(torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) - 3)
    codes = torch.linspace(-2, 2, rows * 128).reshape(rows, 128).to(torch.float8_e4m3fn)
    out = dequantize_activation(codes, scale)
    assert out.shape == (rows, 128)
    expected = codes.float().reshape(rows, 4, 32) * scale.unsqueeze(-1)
    assert torch.equal(out, expected.reshape(rows, 128))


@pytest.mark.parametrize("rows", [0, 2])
def test_l2_requant_accepts_empty_chunks(rows):
    from megatron.lite.model.deepseek_v41.vllm.megamoe import _requant_l2

    h = torch.linspace(-600, 600, rows * 64).reshape(rows, 64)
    out = _requant_l2(h)
    assert out.shape == h.shape
    if rows:
        assert out.abs().max() <= 600 and torch.isfinite(out).all()

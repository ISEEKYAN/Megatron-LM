# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native layout and replay wiring controls; no claim of CUDA kernel parity."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm import fused_lora


def reference_layout(codes, scale, first):
    # Independent element coordinates from the native MegaMoE loader contract.
    e, n, k = codes.shape
    sf = scale.contiguous().view(torch.int32)
    a, b = torch.empty_like(codes), torch.empty_like(sf)
    for expert in range(e):
        for row in range(n):
            if first:
                half, r = divmod(row, n // 2)
                a[expert, (r // 8) * 16 + 8 * half + r % 8] = codes[expert, row]
                block, r = divmod(r, 64)
                j, r = divmod(r, 16)
                q, m = divmod(r, 8)
                target = block * 128 + 64 * q + 32 * half + 4 * m + j
            else:
                a[expert, row] = codes[expert, row]
                block, r = divmod(row, 128)
                j, i = divmod(r, 32)
                target = block * 128 + 4 * i + j
            b[expert, target] = sf[expert, row]
    return a, b


@pytest.mark.parametrize('first', [False, True])
@pytest.mark.parametrize('shape', [(1, 128, 128), (3, 256, 256), (2, 512, 384)])
def test_native_layout_independent_coordinates(first, shape):
    e, n, k = shape
    gen = torch.Generator().manual_seed(1837)
    codes = torch.randint(-128, 128, (e, n, k), dtype=torch.int8, generator=gen)
    scale = torch.randint(0, 255, (e, n, k // 16), dtype=torch.uint8, generator=gen)
    before = codes.clone(), scale.clone()
    got = fused_lora._layout(codes, scale, first=first)
    expected = reference_layout(codes, scale, first)
    assert all(torch.equal(a, b) for a, b in zip(got, expected))
    assert got[0].is_contiguous() and got[1].stride(1) == 1
    assert torch.equal(codes, before[0]) and torch.equal(scale, before[1])
    # This transpose is numerically harmless for scales of all ones but wrong
    # for real release exponents: retain a nonuniform negative control.
    assert not torch.equal(got[1], scale.contiguous().view(torch.int32))


@pytest.mark.parametrize('rank', [0, 7, 65, 72])
def test_native_hook_rank_rejected_before_kernel_import(monkeypatch, rank):
    gu = SimpleNamespace(rank=rank)
    slots = [SimpleNamespace(lora_a=torch.empty(rank, 8)) for _ in range(3)]
    owner = SimpleNamespace(experts=object(), shared_experts=object())
    values = iter([gu, *slots])
    monkeypatch.setattr(fused_lora, 'adapter', lambda *_: next(values))
    with pytest.raises(ValueError, match='ranks'):
        fused_lora.factors(owner)


@pytest.mark.parametrize('requires_x', [False, True])
def test_hook_backward_is_explicit_replay_not_native_or_additive_surrogate(monkeypatch, requires_x):
    parameter = torch.nn.Parameter(torch.tensor(3.0))
    group = SimpleNamespace(size=lambda: 1)
    owner = SimpleNamespace(
        ep_group=group,
        num_local_experts=1,
        experts=SimpleNamespace(linear_fc1=SimpleNamespace(weight0=torch.empty(256, 128))),
        _lora_rows=lambda x, scores, ids: x * parameter + scores.sum(-1, keepdim=True),
    )
    monkeypatch.setattr(fused_lora, 'factors', lambda _: (SimpleNamespace(rank=8),))
    monkeypatch.setattr(fused_lora.experts, 'symm_buffer', lambda *a, **kw: object())
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda *a, **kw: None)
    monkeypatch.setattr(fused_lora, 'frozen_weights', lambda _: ())
    monkeypatch.setattr(fused_lora, '_launch', lambda o, x, *a: x * 100 + parameter * 1000)
    x = torch.tensor([[2.0, 4.0]], requires_grad=requires_x)
    scores = torch.tensor([[0.25, 0.75]], requires_grad=True)
    ids = torch.tensor([[0, 0]], dtype=torch.int64)
    result = fused_lora._Hooks.apply(x, scores, ids, owner, parameter)
    assert torch.equal(result.detach(), x.detach() * 100 + parameter.detach() * 1000)
    result.sum().backward()
    assert parameter.grad.item() == 6
    assert torch.equal(scores.grad, torch.full_like(scores, 2.0))
    assert torch.equal(x.grad, torch.full_like(x, 3.0)) if requires_x else x.grad is None


def test_frozen_payloads_remain_native_and_per_layer_not_a_decoded_cache():
    from megatron.lite.primitive.quantization.frozen import FrozenWeight

    def weight(n, k, kind):
        divisor = 2 if kind == 'mxfp4' else 1
        dtype = torch.int8 if divisor == 2 else torch.float8_e4m3fn
        codes = torch.arange(n * k // divisor, dtype=torch.int64).to(torch.uint8)
        codes = codes.view(dtype).reshape(n, k // divisor)
        sf_shape = (n, k // 32) if divisor == 2 else (n // 32, k // 32)
        sf = (torch.arange(sf_shape[0] * sf_shape[1]) % 240).to(torch.uint8)
        return FrozenWeight(codes, sf.view(torch.float8_e8m0fnu).reshape(sf_shape), (n, k), kind)

    a = [weight(512, 256, 'mxfp4') for _ in range(2)]
    b = [weight(256, 256, 'mxfp4') for _ in range(2)]
    s1, s2 = weight(512, 256, 'fp8_block'), weight(256, 256, 'fp8_block')
    owner = SimpleNamespace(
        num_local_experts=2,
        experts=SimpleNamespace(
            linear_fc1=SimpleNamespace(**{f'weight{i}': w for i, w in enumerate(a)}),
            linear_fc2=SimpleNamespace(**{f'weight{i}': w for i, w in enumerate(b)}),
        ),
        shared_experts=SimpleNamespace(
            linear_fc1=SimpleNamespace(weight=s1), linear_fc2=SimpleNamespace(weight=s2)
        ),
    )
    got = fused_lora.frozen_weights(owner)
    for index, (ws, first) in enumerate(((a, True), (b, False), ([s1], True), ([s2], False))):
        codes = torch.stack([w.codes for w in ws])
        sf = torch.stack([w.scale.view(torch.uint8) for w in ws])
        if index >= 2:
            sf = sf.repeat_interleave(32, dim=1)
        c, s = reference_layout(codes.view(torch.uint8), sf, first)
        if index >= 2:
            c, s = c[0], s[0]
        assert torch.equal(got[index][0].view(torch.uint8), c)
        assert torch.equal(got[index][1], s)
    # No full-model packed/decoded cache gets attached to the owner.
    assert set(vars(owner)) == {'num_local_experts', 'experts', 'shared_experts'}
    assert all(not w.requires_grad and isinstance(w, FrozenWeight) for w in (*a, *b, s1, s2))


@pytest.mark.parametrize('rows', [0, 7])
def test_native_gate_up_shrink_is_one_call_with_three_slices(monkeypatch, rows):
    import sys
    from types import ModuleType

    calls = []
    ops = ModuleType('vllm.lora.ops.triton_ops')

    def native(x, weights, out, *args):
        calls.append((len(weights), x.shape, args))
        assert args == ('metadata', 1.0)
        for index, w in enumerate(weights):
            assert not w.requires_grad
            out[index].copy_(x.float() @ w[0, 0].float().T)

    ops.lora_shrink = native
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setattr(fused_lora, 'metadata', lambda _: ('metadata',))
    x = torch.randn(rows, 128, dtype=torch.bfloat16)
    weights = [torch.randn(8, 128, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    got = fused_lora.shrink_slices(x, weights)
    assert got.shape == (3, rows, 8)
    assert len(calls) == (1 if rows else 0)
    if rows:
        assert torch.equal(got, torch.stack([x.float() @ w.float().T for w in weights]))

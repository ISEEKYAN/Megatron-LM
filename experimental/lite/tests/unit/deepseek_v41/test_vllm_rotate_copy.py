# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Copy-layout adaptation only; no new rotation arithmetic or kernels."""

import pytest
import torch

from megatron.lite.model.deepseek_v41.vllm.attention import _rotate


def reference(x, positions, cache, inverse=False):
    width = cache.shape[-1]
    angles = cache[positions].float()
    cos, sin = angles.chunk(2, -1)
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if inverse:
        sin = -sin
    tail = x[..., -width:].float().unflatten(-1, (-1, 2))
    a, b = tail.unbind(-1)
    rotated = torch.stack((a * cos - b * sin, b * cos + a * sin), -1).flatten(-2)
    return torch.cat((x[..., :-width], rotated.to(x.dtype)), -1)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('rows', [0, 1, 7])
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_copy_rotation_raw_bits_and_input_ownership(dtype, rows, inverse, noncontiguous):
    torch.manual_seed(432)
    x = torch.randn(4, rows, 512, dtype=dtype).transpose(0, 1)
    if not noncontiguous:
        x = x.contiguous()
    if rows:
        x[0, 0, 0] = 0.0
        x[0, 0, 1] = -0.0
    pos = torch.arange(rows)
    cache = torch.randn(max(rows, 1), 64)
    original = x.clone()
    expected = reference(x, pos, cache, inverse)
    actual = _rotate(x, pos, cache, inverse)
    bitdtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.view(bitdtype), expected.view(bitdtype))
    assert torch.equal(x.view(bitdtype), original.view(bitdtype))
    assert actual.is_contiguous()
    owned = x.clone()
    same = _rotate(owned, pos, cache, inverse, True)
    assert same is owned
    assert torch.equal(same.view(bitdtype), expected.view(bitdtype))


@pytest.mark.parametrize('inverse', [False, True])
def test_native_adapter_reuses_inverse_op_without_mutating_cache(monkeypatch, inverse):
    import sys
    from types import ModuleType

    from megatron.lite.model.deepseek_v41.vllm.attention import _native_rotate

    called = []

    def native(x, positions, cache, **kw):
        called.append(kw)
        return reference(x, positions, cache, True).flatten(1).unsqueeze(1), torch.empty(0)

    mod = ModuleType('vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant')
    mod.fused_inv_rope_fp8_quant = native
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    x = torch.randn(7, 4, 512).bfloat16()
    pos = torch.tensor([7, 2, 2, 0, 5, 1, 4])
    cache = torch.randn(8, 64)
    old = cache.clone()
    out = _native_rotate(x, pos, cache, inverse)
    assert torch.equal(out.view(torch.int16), reference(x, pos, cache, inverse).view(torch.int16))
    assert torch.equal(cache, old)
    assert len(called) == 1 and called[0]['quantize'] is False and called[0]['n_groups'] == 1

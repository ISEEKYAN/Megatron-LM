# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sleep discards CUDA permutation-cache contents without removing dictionary keys."""

import pytest
import torch
from test_receiver_staging import adapter


@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('projection', ['w13', 'w2'])
def test_sleep_refit_rebuilds_shape_cache_bytes(v41_core_te, scale, projection):
    from native_reload import mxfp4_method, permute_builders

    consumer = adapter(v41_core_te)
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    method = model.experts.quant_method = mxfp4_method()
    builders = permute_builders()
    builder = (
        builders.get_w2_permute_indices_with_cache
        if projection == 'w2'
        else builders._maybe_get_cached_w3_w1_permute_indices
    )
    raw = torch.arange(128 * 8).remainder(239).byte().reshape(128, 8)
    if scale:
        raw[0] = 0
        raw[1] = 127
    cold_indices = builder({}, raw, 128, 32 if scale else None, True).clone()
    assert not torch.equal(cold_indices, torch.arange(128))
    for generation in (1, 2):
        value = raw + generation
        cold = value[cold_indices]
        builder(method._cache_permute_indices, value, 128, 32 if scale else None, True)
        for cached in method._cache_permute_indices.values():
            cached.zero_()  # wake retains keys, loses values.
        stale = builder(
            method._cache_permute_indices, value, 128, 32 if scale else None, True
        )
        assert not torch.equal(value[stale], cold)
        consumer.reset_mxfp4_permute_caches(model)
        assert not method._cache_permute_indices
        rebuilt = builder(
            method._cache_permute_indices, value, 128, 32 if scale else None, True
        )
        assert len(method._cache_permute_indices) == 1
        assert torch.equal(rebuilt, cold_indices)
        assert torch.equal(value[rebuilt], cold)


def test_reset_does_not_touch_other_quant_methods_or_kernel_storage(v41_core_te):
    from types import SimpleNamespace

    from test_receiver_staging import Mxfp4MoEMethod

    consumer = adapter(v41_core_te)
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    method = model.experts.quant_method = Mxfp4MoEMethod()
    method._cache_permute_indices = {'shape': torch.arange(4)}
    model.experts.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    pointer = model.experts.weight.data_ptr()
    model.other = torch.nn.Module()
    unrelated = model.other.quant_method = SimpleNamespace(
        _cache_permute_indices={'keep': torch.arange(4)}
    )
    getattr(consumer, 'reset_mxfp4_permute_caches', lambda model: None)(model)
    assert method._cache_permute_indices == {}
    assert 'keep' in unrelated._cache_permute_indices
    assert model.experts.weight.data_ptr() == pointer
    assert torch.equal(model.experts.weight, torch.ones(4))

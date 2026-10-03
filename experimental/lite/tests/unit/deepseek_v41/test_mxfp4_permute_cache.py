# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sleep discards CUDA permutation-cache contents without removing dictionary keys."""

import pytest
import torch
from test_receiver_staging import adapter


@pytest.mark.parametrize('generation', [1, 2])
@pytest.mark.parametrize('scale', [False, True])
def test_sleep_refit_rebuilds_shape_cache_bytes(v41_core_te, generation, scale):
    from test_receiver_staging import Mxfp4MoEMethod

    consumer = adapter(v41_core_te)
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    method = model.experts.quant_method = Mxfp4MoEMethod()
    key = ('w2', torch.Size([8, 2]), 128, 16 if scale else None, None)
    permutation = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
    method._cache_permute_indices = {key: permutation.clone()}
    raw = torch.arange(16, dtype=torch.uint8).reshape(8, 2) + generation
    if scale:
        # UE8M0 is compared as bytes, including zero padding and exponent 127.
        raw[0] = 0
        raw[1] = 127
    cold = raw[permutation]
    method._cache_permute_indices[key].zero_()  # wake retains keys, loses values.
    assert not torch.equal(raw[method._cache_permute_indices[key]], cold)
    getattr(consumer, 'reset_mxfp4_permute_caches', lambda model: None)(model)
    if key not in method._cache_permute_indices:
        method._cache_permute_indices[key] = permutation.clone()
    refit = raw[method._cache_permute_indices[key]]
    assert torch.equal(refit.view(torch.uint8), cold.view(torch.uint8))


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

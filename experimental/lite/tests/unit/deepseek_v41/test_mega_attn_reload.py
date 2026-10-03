# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Checkpoint bytes and cold-load ordering, without CUDA/vLLM imports."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from native_reload import fused_layout
from test_receiver_staging import adapter


class Pack:
    def process_weights_after_loading(self, layer):
        # Kernel layouts differ from checkpoint row/column layouts.
        layer.weight.data = layer.weight.data.t().contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()


class DeepseekV4MegaAttnAttention(torch.nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.n_local_heads, self.n_local_groups = 64 // tp, 8 // tp
        self._fused_layouts_ready = True
        for name, shape in (
            ('wq_b', (self.n_local_heads * 512, 1280)),
            ('wo_a', (self.n_local_groups * 1024, 4096)),
        ):
            projection = torch.nn.Module()
            for key, dims in (
                ('weight', shape),
                ('weight_scale', (shape[0], shape[1] // 32)),
            ):
                raw = (torch.arange(dims[0] * dims[1]) % 239).byte().reshape(dims)
                projection.register_parameter(
                    key, torch.nn.Parameter(raw, requires_grad=False)
                )
            projection.quant_method = Pack()
            setattr(self, name, projection)

    def finalize_loaded_weights(self):
        if not self._fused_layouts_ready:
            fused_layout().permute_wq_b_(
                self.wq_b.weight, self.wq_b.weight_scale, self.n_local_heads
            )
            fused_layout().permute_wo_a_(self.wo_a.weight, self.wo_a.weight_scale, 8)
            self._fused_layouts_ready = True


@pytest.fixture
def native_permutations(monkeypatch):
    name = 'vllm.models.deepseek_v41.common.ops.fused_layout'
    module = fused_layout()
    for count in range(1, len(name.split('.'))):
        parent_name = '.'.join(name.split('.')[:count])
        package = ModuleType(parent_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, parent_name, package)
    sys.modules[name.rsplit('.', 1)[0]].fused_layout = module
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize('tp', [4, 8])
@pytest.mark.parametrize('order', [('wq_b', 'wo_a'), ('wo_a', 'wq_b')])
def test_refit_matches_cold_bytes_for_two_generations(
    v41_core_te, native_permutations, tp, order
):
    consumer = adapter(v41_core_te)
    model = DeepseekV4MegaAttnAttention(tp)
    for generation in (1, 2):
        cold = DeepseekV4MegaAttnAttention(tp)
        cold._fused_layouts_ready = False
        for name in order:
            for key in ('weight', 'weight_scale'):
                raw = getattr(getattr(cold, name), key).data
                raw.copy_((raw.int() + generation).remainder(239).byte())
                getattr(getattr(model, name), key).data = raw.clone()
        cold.finalize_loaded_weights()
        for name in order:
            projection = getattr(cold, name)
            projection.quant_method.process_weights_after_loading(projection)
        # Against the old adapter this is precisely its defer/pack/finalize order.
        model._fused_layouts_ready = False
        hook = (
            consumer.MegaAttnReload(model)
            if hasattr(consumer, 'MegaAttnReload')
            else None
        )
        for name in order:
            projection = getattr(model, name)
            projection.quant_method.process_weights_after_loading(projection)
        if hook:
            hook.finish()
            hook.restore()
        model.finalize_loaded_weights()
        model.finalize_loaded_weights()  # no double permutation
        for name in order:
            for key in ('weight', 'weight_scale'):
                actual, expected = getattr(getattr(model, name), key), getattr(
                    getattr(cold, name), key
                )
                assert actual.shape == expected.shape
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert (
                'process_weights_after_loading'
                not in getattr(model, name).quant_method.__dict__
            )


def test_incomplete_projection_fails_and_restores(v41_core_te, native_permutations):
    model = DeepseekV4MegaAttnAttention(8)
    hook = adapter(v41_core_te).MegaAttnReload(model)
    model.wq_b.quant_method.process_weights_after_loading(model.wq_b)
    with pytest.raises(ValueError, match='incomplete MegaAttn'):
        hook.finish()
    assert not model._fused_layouts_ready
    hook.restore()
    assert all(
        'process_weights_after_loading' not in p.quant_method.__dict__
        for p in (model.wq_b, model.wo_a)
    )


@pytest.mark.parametrize('tp', [4, 8])
def test_wq_b_tile_codec_tp_slice_and_load_export_bytes(v41_core_te, tp):

    from megatron.lite.model.deepseek_v41.lite.checkpoint import _SPEC
    from megatron.lite.primitive.ckpt import hf_weights

    name = 'layers.0.attn.wq_b.weight'
    # Keep actual 512-element heads and TP shard axes; smaller q-rank bounds CPU work.
    shape = (64 * 512, 64)
    matrix = (
        torch.arange(shape[0] * shape[1]).reshape(shape).remainder(31) - 15
    ).bfloat16()
    weight, scale = hf_weights.encode_matrix(
        _SPEC, name, matrix, 'F8_E4M3', 5 * matrix.numel()
    )
    whole = _SPEC.encode(name, matrix, 'F8_E4M3')
    for actual, expected in zip((weight, scale), whole):
        assert actual.shape == expected.shape
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    reader = SimpleNamespace(
        _get_raw_tensor=lambda key, device: weight if key == name else scale
    )
    decoded = hf_weights.load_bound_weight(reader, name, _SPEC)
    encoded = _SPEC.encode(name, decoded, 'F8_E4M3')
    for actual, expected in zip(encoded, (weight, scale)):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for rank in range(tp):
        lo, hi = rank * shape[0] // tp, (rank + 1) * shape[0] // tp
        local = _SPEC.encode(name, matrix[lo:hi], 'F8_E4M3')
        assert local[0].shape == (64 // tp * 512, 64)
        for actual, full in zip(local, (weight, scale)):
            start, stop = rank * full.shape[0] // tp, (rank + 1) * full.shape[0] // tp
            assert torch.equal(
                actual.view(torch.uint8), full[start:stop].view(torch.uint8)
            )


def test_duplicate_projection_is_rejected(v41_core_te, native_permutations):
    model = DeepseekV4MegaAttnAttention(8)
    hook = adapter(v41_core_te).MegaAttnReload(model)
    model.wq_b.quant_method.process_weights_after_loading(model.wq_b)
    with pytest.raises(RuntimeError, match='processed twice'):
        model.wq_b.quant_method.process_weights_after_loading(model.wq_b)
    hook.restore()

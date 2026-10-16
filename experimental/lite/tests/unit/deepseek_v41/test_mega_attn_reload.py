# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unsupported MegaAttn fails during construction, before reload or packing."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_receiver_staging import adapter


@pytest.mark.parametrize('phase', ['config', 'model'])
def test_mega_backend_rejected_before_reload(v41_core_te, monkeypatch, phase):
    consumer = adapter(v41_core_te)
    model = type('DeepseekV4MegaAttnAttention', (torch.nn.Module,), {})()
    if phase == 'config':

        class BaseModelLoader:
            def create_model(self, *args):
                pytest.fail('MegaAttn must fail before model allocation')

        for name in (
            'vllm',
            'vllm.model_executor',
            'vllm.model_executor.model_loader',
            'vllm.model_executor.model_loader.reload',
            'vllm.model_executor.model_loader.base_loader',
        ):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        sys.modules['vllm.model_executor.model_loader.reload'].layerwise = object()
        sys.modules['vllm.model_executor.model_loader.base_loader'].BaseModelLoader = (
            BaseModelLoader
        )
        monkeypatch.setattr(consumer, 'install_mxfp4_load_numel_compat', lambda _: None)
        consumer.install_reload_metadata_hook()
    with pytest.raises(
        NotImplementedError,
        match=r'MegaAttn rollout backend unsupported in v1\.4 '
        r'\(planned v1\.5\); use FLASHMLA_SPARSE_DSV41',
    ):
        if phase == 'config':
            BaseModelLoader().create_model(
                SimpleNamespace(
                    attention_config=SimpleNamespace(backend='FLASHMLA_SPARSE_MEGA')
                ),
                SimpleNamespace(hf_config=SimpleNamespace(model_type='deepseek_v41')),
            )
        else:
            consumer.reject_mega_attn(model=model)


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

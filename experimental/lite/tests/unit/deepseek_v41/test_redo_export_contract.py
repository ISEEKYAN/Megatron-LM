# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deployment wire contract, independent of MLite's checkpoint reader.

Reference: vllm-project/vllm@751f6807d9cb3de50c27a5f27188c4fb04fe0e2b:
  models/deepseek_v41/quant_config.py:186 (MXFP8 linear / MXFP4 experts)
  model_executor/layers/quantization/modelopt.py:2200 (32x32 -> 1x32 scales)
  model_executor/layers/quantization/mxfp4.py:576 (packed K/2, scales K/32)
  models/deepseek_v41/common/engram.py:695 (FP8 rows, scales K/32)
Official HF dba1be0a40aa45a94ad051997016db3960a90277 shard headers 17/47
independently declare F8_E4M3/I8 weights and F8_E8M0 scale siblings.
"""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_redo_parity import release_config


@pytest.mark.parametrize('trainable_engram', [False, True])
def test_hf_export_obeys_external_quantized_storage(
    v41_core_te, tmp_path, trainable_engram
):
    from megatron.lite.model.deepseek_v41.lite import checkpoint, protocol
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    model = protocol.build_model(
        release_config(),
        impl_cfg=protocol.ImplConfig(
            device='cpu',
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(64)),
            trainable_engram=trainable_engram,
        ),
    ).chunks[0]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.3)
        if not trainable_engram:
            # Independent E4M3/E8M0 bytes: frozen rows must survive unchanged.
            table = model.layers[1].engram.embed
            table.weight.view(torch.uint8).fill_(0x7A)
            table.scale.view(torch.uint8).fill_(119)
    # VL's upstream mapper drops mtp.*. Preserve opaque inactive bytes here;
    # the assertions below concern the externally consumed active backbone.
    archive = tmp_path / 'archive'
    archive.mkdir()
    save_file(
        {n: torch.zeros(1, dtype=torch.uint8) for n in model.archival_bindings},
        str(archive / 'model.safetensors'),
    )
    model.archival_store = SafeTensorReader(str(archive))
    model.archival_keys = sorted(model.archival_bindings)
    destination = tmp_path / 'saved'
    checkpoint.save_model(model, destination, buffer_max_size_bytes=65536)
    config = json.loads((destination / 'config.json').read_text())
    assert config['quantization_config'] == dict(
        quant_method='fp8',
        activation_scheme='dynamic',
        weight_block_size=[32, 32],
        scale_fmt='ue8m0',
        expert_dtype='fp4',
    )
    # Read the external HF index only, never the MLite loader or resume state.
    index = json.loads((destination / 'model.safetensors.index.json').read_text())
    weights = {}
    for filename in set(index['weight_map'].values()):
        weights.update(load_file(str(destination / filename)))
    pairs = {
        'layers.14.attn.wq_a': (torch.float8_e4m3fn, (32, 32), (1, 1)),
        'layers.14.attn.wq_b': (torch.float8_e4m3fn, (32, 32), (1, 1)),
        'layers.14.attn.indexer.wq_b': (torch.float8_e4m3fn, (32, 32), (1, 1)),
        'layers.14.ffn.shared_experts.w1': (torch.float8_e4m3fn, (32, 32), (1, 1)),
        'layers.14.ffn.experts.0.w1': (torch.int8, (32, 16), (32, 1)),
        'layers.14.ffn.experts.0.w2': (torch.int8, (32, 16), (32, 1)),
        'layers.14.ffn.experts.0.w3': (torch.int8, (32, 16), (32, 1)),
        'layers.1.engram.wkv': (torch.float8_e4m3fn, (96, 64), (3, 2)),
        'layers.1.engram.embed': (torch.float8_e4m3fn, (18, 32), (18, 1)),
    }
    for name, (dtype, shape, scale_shape) in pairs.items():
        weight, scale = weights[name + '.weight'], weights.get(name + '.scale')
        assert weight.dtype == dtype, name
        assert tuple(weight.shape) == shape, name
        assert scale is not None, name
        assert scale.dtype == torch.float8_e8m0fnu, name
        assert tuple(scale.shape) == scale_shape, name
        # Independent wire bytes for 1.3: FP8 320 * 2^-8 = 1.25;
        # E2M1 6 * 2^-2 = 1.5. No production quantizer builds the expected data.
        byte, exponent = (0x77, 125) if dtype == torch.int8 else (0x7A, 119)
        assert torch.equal(
            weight.view(torch.uint8), torch.full(shape, byte, dtype=torch.uint8)
        )
        assert torch.equal(
            scale.view(torch.uint8),
            torch.full(scale_shape, exponent, dtype=torch.uint8),
        )
    for name in (
        'embed.weight',
        'head.weight',
        'layers.14.attn.compressor.wkv.weight',
        'layers.14.attn.indexer.wk.weight',
        'vision.patch_embed.proj.weight',
    ):
        assert weights[name].dtype == torch.float32, name
        assert name[:-6] + 'scale' not in weights, name
        assert torch.equal(weights[name], torch.full_like(weights[name], 1.3))

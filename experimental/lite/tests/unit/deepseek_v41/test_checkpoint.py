# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.model.deepseek_v41.lite.checkpoint import (
    CheckpointTensorStore,
    load_weight,
)
from safetensors.torch import save_file


@pytest.mark.parametrize('kind', ['fp4', 'fp8', 'engram'])
def test_decode_archive_and_reload(tmp_path, kind):
    name = 'layers.1.' + (
        'engram.embed.weight' if kind == 'engram' else 'attn.wq_a.weight'
    )
    if kind == 'fp4':
        weight = torch.tensor(
            [list(bytes.fromhex('10 32 54 76 98 ba dc fe') * 2)] * 2, dtype=torch.uint8
        ).view(torch.int8)
        scale = torch.tensor([[127], [128]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        row = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6] * 2
        expected = torch.tensor([row, [2 * x for x in row]])
    else:
        shape = (3, 32) if kind == 'engram' else (64, 64)
        weight = torch.ones(shape).to(torch.float8_e4m3fn)
        scales = [[126], [127], [128]] if kind == 'engram' else [[126, 127], [128, 129]]
        scale = torch.tensor(scales, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        expected = (
            scale.float()
            .repeat_interleave(1 if kind == 'engram' else 32, 0)
            .repeat_interleave(32, 1)
        )
    tensors = {name: weight, name[:-6] + 'scale': scale}
    source, archive, export = [tmp_path / n for n in ('source', 'archive', 'export')]
    save_file(tensors, source)
    store = CheckpointTensorStore.load([source], expected_keys=tensors)
    torch.testing.assert_close(
        load_weight(store, name, output_dtype=torch.float32), expected, atol=0, rtol=0
    )
    store.save(archive)
    restored = CheckpointTensorStore.load([archive], expected_keys=tensors)
    assert all(store.read(key) == restored.read(key) for key in tensors)
    save_file({name: expected.bfloat16()}, export)
    reloaded = CheckpointTensorStore.load([export], expected_keys=[name])
    assert torch.equal(load_weight(reloaded, name), expected.bfloat16())


@pytest.mark.parametrize(
    'damage,message',
    [
        ('missing', 'missing scale'),
        ('shape', 'scale shape'),
        ('dtype', 'release scales must be E8M0'),
        ('nan', 'nonfinite release scale'),
        ('stale', 'unexpected scale'),
    ],
)
def test_decode_rejects_scale_damage(tmp_path, damage, message):
    name = 'layers.0.ffn.experts.0.w1.weight'
    tensors = {name: torch.zeros(2, 16, dtype=torch.int8)}
    if damage != 'missing':
        scale = torch.ones(2, 1).to(torch.float8_e8m0fnu)
        if damage == 'shape':
            scale = scale[:1]
        elif damage == 'dtype':
            scale = scale.float()
        elif damage == 'nan':
            scale.view(torch.uint8)[0, 0] = 255
        elif damage == 'stale':
            tensors[name] = tensors[name].float()
        tensors[name[:-6] + 'scale'] = scale
    path = tmp_path / 'bad'
    save_file(tensors, path)
    store = CheckpointTensorStore.load([path], expected_keys=tensors)
    with pytest.raises((ValueError, TypeError), match=message):
        load_weight(store, name)

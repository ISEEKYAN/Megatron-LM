# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import io
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import block as hc
from megatron.lite.model.deepseek_v41.lite import image_data as data
from megatron.lite.model.deepseek_v41.lite import vision
from PIL import Image


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_pinned_vision_nondivisible_forward_and_vjp(official, dtype):
    source = official('vision.py')
    args = SimpleNamespace(
        vision_patch_size=14,
        vision_dim=16,
        vision_n_heads=2,
        vision_inter_dim=32,
        vision_n_layers=1,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        dim=12,
    )
    actual = torch.nn.Sequential(vision.ViT(args), vision.Aligner(args)).to(dtype)
    reference = torch.nn.Sequential(source.ViT(args), source.Aligner(args)).to(dtype)
    reference.load_state_dict(actual.state_dict())
    for height, width in ((3, 6), (4, 5), (1, 2)):
        x = torch.randn(height * width, 3, 14, 14, dtype=dtype, requires_grad=True)
        y = actual[1](actual[0](x, height, width), height, width)
        expected = reference[1](reference[0](x, height, width), height, width)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        probe = torch.randn_like(y)
        ga = torch.autograd.grad((y * probe).sum(), (x, *actual.parameters()))
        refs = dict(reference.named_parameters())
        gb = torch.autograd.grad(
            (expected * probe).sum(),
            (x, *(refs[n] for n, _ in actual.named_parameters())),
        )
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('size,ratio', [((29, 43), None), ((101, 19), 2)])
def test_pinned_processor_and_multimage_spans(official, size, ratio):
    source = official('image_processor.py')
    cfg = data.ImageConfig(max_wh_ratio=ratio)
    args = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=1024,
        vision_min_pixels=295936,
        vision_max_wh_ratio=ratio,
    )
    image = Image.frombytes(
        'RGB', size, bytes(i % 251 for i in range(size[0] * size[1] * 3))
    )
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    expected = source.load_image({'data': buffer.getvalue()}, args)
    actual = data.preprocess_image(image, cfg)
    assert actual[1:] == expected[1:]
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    ids, types, images = data.prepare_image_inputs(
        [1, 99, 2, 99, 3], [image, image], 99, cfg
    )
    layout = source.image_token_types(*actual[-2:]).tolist()
    assert ids == [1] + [99] * len(layout) + [2] + [99] * len(layout) + [3]
    assert types == [-1] + layout + [-1] + layout + [-1]
    assert [i.start for i in images] == [1, len(layout) + 2]


def test_multimage_hc_copy_and_delimiter_gradients():
    images = [
        [
            data.ImageInput(
                n, torch.zeros(2, 3, 2, 2), 1, 2, data.image_token_types(1, 2)
            )
            for n in (1, 7)
        ]
    ]
    text = torch.randn(1, 13, 4, requires_grad=True)
    features = [torch.randn(2, 4, requires_grad=True) for _ in range(2)]
    delimiters = [torch.randn(4, requires_grad=True) for _ in range(3)]
    merged = data.merge_image_embeddings(text, images, [features], *delimiters)
    expanded, _ = hc.expand_hc(merged, 3)
    expected = text.clone()
    for start, feature in zip((1, 7), features):
        expected[:, start : start + 5] = torch.stack(
            (delimiters[0], feature[0], feature[1], delimiters[2], delimiters[1])
        )
    probe = torch.arange(expanded.numel()).reshape_as(expanded).float()
    for i in range(3):
        torch.testing.assert_close(expanded[:, :, i], expected)
    args = (text, *features, *delimiters)
    ga = torch.autograd.grad((expanded * probe).sum(), args)
    gb = torch.autograd.grad((expected * probe.sum(2)).sum(), args)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    newline = torch.nn.Parameter(torch.zeros(2))
    image = data.ImageInput(
        0, torch.zeros(2, 3, 1, 1), 2, 1, data.image_token_types(2, 1)
    )
    output = data.merge_image_embeddings(
        torch.zeros(1, 6, 2, dtype=torch.bfloat16),
        [[image]],
        [[torch.zeros(2, 2)]],
        torch.zeros(2),
        torch.zeros(2),
        newline,
    )
    probe = torch.zeros_like(output)
    probe[0, 2], probe[0, 4] = 1, 2**-8
    output.backward(probe)
    assert torch.equal(newline.grad, torch.full((2,), 1 + 2**-8))

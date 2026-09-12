# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from types import SimpleNamespace
from megatron.lite.model.deepseek_v41.lite import block as hc


def _check_vision_official(monkeypatch, dtype):
    import hashlib
    import importlib.util
    from pathlib import Path

    from megatron.lite.model.deepseek_v41.lite import vision

    import os


    path = (
        Path(os.environ.get('DS41_REFERENCE_DIR', '/tmp/ds41-fixture-reference'))
        / 'vision.py'
    )
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == '5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c'
    )
    spec = importlib.util.spec_from_file_location('official_vision', path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    args = SimpleNamespace(
        vision_patch_size=14,
        vision_dim=64,
        vision_n_heads=4,
        vision_inter_dim=128,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        dim=12,
    )
    torch.manual_seed(19)
    actual = torch.nn.Sequential(vision.ViT(args), vision.Aligner(args)).to(dtype)
    reference = torch.nn.Sequential(official.ViT(args), official.Aligner(args)).to(
        dtype
    )
    reference.load_state_dict(actual.state_dict())
    for height, width in [(3, 6), (4, 5), (1, 2)]:
        x = torch.randn(height * width, 3, 14, 14, dtype=dtype, requires_grad=True)
        y = actual[1](actual[0](x, height, width), height, width)
        expected = reference[1](reference[0](x, height, width), height, width)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        probe = torch.randn_like(y)
        ga = torch.autograd.grad((y * probe).sum(), (x, *actual.parameters()))
        gb = torch.autograd.grad(
            (expected * probe).sum(),
            (
                x,
                *[
                    dict(reference.named_parameters())[name]
                    for name, _ in actual.named_parameters()
                ],
            ),
        )
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=0, atol=0)



def test_v41_multimage_copy_gradient():
    from megatron.lite.model.deepseek_v41.lite import image_data

    types = image_data.image_token_types(1, 2)
    images = [
        [
            image_data.ImageInput(1, torch.zeros(2, 3, 2, 2), 1, 2, types),
            image_data.ImageInput(7, torch.zeros(2, 3, 2, 2), 1, 2, types),
        ]
    ]
    text = torch.randn(1, 13, 4, requires_grad=True)
    features = [torch.randn(2, 4, requires_grad=True) for _ in range(2)]
    delimiters = [torch.randn(4, requires_grad=True) for _ in range(3)]
    merged = image_data.merge_image_embeddings(text, images, [features], *delimiters)
    expanded, _ = hc.expand_hc(merged, 3)
    probe = torch.arange(expanded.numel()).reshape_as(expanded).float()
    gradients = torch.autograd.grad(
        (expanded * probe).sum(), (text, *features, *delimiters)
    )
    expected = text.clone()
    for start, feature in zip((1, 7), features):
        expected[:, start : start + 5] = torch.stack(
            (delimiters[0], feature[0], feature[1], delimiters[2], delimiters[1])
        )
    for copy in range(3):
        torch.testing.assert_close(expanded[:, :, copy], expected)
    summed = probe.sum(2)
    mask = torch.ones(13, dtype=torch.bool)
    mask[1:6] = False
    mask[7:12] = False
    torch.testing.assert_close(gradients[0], summed * mask[None, :, None])
    for grad, start in zip(gradients[1:3], (2, 8)):
        torch.testing.assert_close(grad, summed[0, start : start + 2])
    for grad, offsets in zip(gradients[3:], ((1, 7), (5, 11), (4, 10))):
        torch.testing.assert_close(grad, summed[0, list(offsets)].sum(0))



@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v41_vision_nondivisible_official(monkeypatch, dtype):
    _check_vision_official(monkeypatch, dtype)



def test_v41_delimiter_reduction_preserves_native_fp32():
    from megatron.lite.model.deepseek_v41.lite.image_data import (
        ImageInput,
        image_token_types,
        merge_image_embeddings,
    )

    layout = image_token_types(2, 1)
    image = ImageInput(0, torch.zeros(2, 3, 1, 1), 2, 1, layout)
    newline = torch.nn.Parameter(torch.zeros(2))
    output = merge_image_embeddings(
        torch.zeros(1, 6, 2, dtype=torch.bfloat16),
        [[image]],
        [[torch.zeros(2, 2)]],
        torch.zeros(2),
        torch.zeros(2),
        newline,
    )
    grad = torch.zeros_like(output)
    grad[0, 2], grad[0, 4] = 1, 2**-8
    output.backward(grad)
    assert torch.equal(
        newline.grad, torch.full((2,), 1 + 2**-8)
    ), 'Delimiter row reduction rounded in BF16 before FP32 accumulation'

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.primitive.modules.image_data import ImageInput, image_token_types
from megatron.lite.primitive.modules.vision import Aligner, ViT
from megatron.lite.primitive.modules.vision_training import (
    VisionSchedule,
    VisionTrainability,
)


def owner():
    args = SimpleNamespace(
        vision_patch_size=2,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=16,
        vision_n_layers=1,
        vision_rope_theta=10000,
        vision_downsample_ratio=1,
        dim=4,
    )
    model = torch.nn.Module()
    model.vision, model.aligner = ViT(args), Aligner(args)
    model.embed = torch.nn.Embedding(8, 4)
    for key in ('image_start', 'image_end', 'image_newline'):
        model.register_parameter(key, torch.nn.Parameter(torch.randn(4)))
    model.register_buffer('_vision_trainability', torch.zeros(4, dtype=torch.bool))
    return model


@pytest.mark.parametrize('encoder', [False, True])
def test_external_vision_matches_owner_backward_and_restart(encoder):
    model = owner()
    mask = VisionTrainability(encoder, True, True, False)
    mask.apply(model)
    reference = deepcopy(model)
    schedule = VisionSchedule(model, 'cpu')
    model.vision_schedule = schedule
    image = ImageInput(0, torch.randn(2, 3, 2, 2), 1, 2, image_token_types(1, 2))
    expected = reference.aligner(reference.vision(image.patches, 1, 2), 1, 2)
    actual = schedule.forward([[image]])[0][0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match='pending vision backward'):
        mask.apply(model)
    with pytest.raises(RuntimeError, match='completed vision backward'):
        schedule.state_dict()
    schedule.backward(actual.square().sum())
    expected.square().sum().backward()
    for (_, parameter), (_, original) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert (parameter.grad is None) == (original.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, original.grad, atol=0, rtol=0)
    assert schedule.stage == 'idle' and not schedule.pending and not schedule.weights
    state = schedule.state_dict()
    schedule.load_state_dict(state)
    assert schedule.state_dict() == state
    assert model.vision_trainability == mask
    schedule.abort()


def test_external_vision_abort_releases_failed_forward():
    model = owner()
    VisionTrainability(True, True, True, True).apply(model)
    schedule = VisionSchedule(model, 'cpu')
    bad = ImageInput(0, torch.zeros(1, 3, 2, 2), 1, 2, image_token_types(1, 2))
    with pytest.raises(ValueError, match='Patch count'):
        schedule.forward([[bad]])
    assert schedule.stage == 'idle' and not schedule.pending and not schedule.weights

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch
from megatron.lite.model.deepseek_v41.lite.optimizer_groups import parameter_groups, VisionOptimizerConfig


@pytest.mark.parametrize('invalid', [None, 'indexer', 'unknown_visual', 'alias'])
def test_actual_optimizer_owners(moe, model_config, invalid):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model
    model = DeepseekV41Model(model_config, token_map=list(range(256)), quantized=False)
    if invalid == 'indexer':
        model.layers[2].attn.indexer.requires_grad_(True)
    elif invalid == 'unknown_visual':
        model.vision.extra = torch.nn.Parameter(torch.ones(3))
        model._bind('vision.extra', model.vision, 'extra', 'vision')
    elif invalid == 'alias':
        model.vision.extra = model.vision.norm.weight
    policy = VisionOptimizerConfig(0.5, 1.0, 0.0)
    if invalid:
        message = {'indexer': 'indexer must remain frozen',
                   'unknown_visual': 'Unknown parameter owner', 'alias': 'alias'}[invalid]
        with pytest.raises(ValueError, match=message):
            parameter_groups(model, lr=0.001, vision_policy=policy)
        return
    groups = parameter_groups(model, lr=0.001, vision_policy=policy)
    routed = {id(p): group['algorithm'] for group in groups for p in group['params']}
    assert set(routed) == {id(p) for p in model.parameters() if p.requires_grad}
    assert routed[id(model.aligner.w1.weight)] == 'muon'
    assert routed[id(model.vision.norm.weight)] == routed[id(model.image_start)] == 'adamw'
    assert all(id(p) not in routed for p in model.layers[2].attn.indexer.parameters())

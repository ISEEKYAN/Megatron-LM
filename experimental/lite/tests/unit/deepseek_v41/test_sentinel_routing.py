# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sentinel IDs take bias_vl only when the checkpoint has a vision tower."""
import torch


def test_sentinel_routing_follows_vision_tower():
    from megatron.lite.model.deepseek_v41.lite.model import sentinel_routing_mask

    ids = torch.tensor([[1, 129263, 129264, 129266, 129268, 129269]])
    assert torch.equal(
        sentinel_routing_mask(ids, True),
        torch.tensor([[False, False, True, True, True, False]]),
    )
    assert sentinel_routing_mask(ids, False) is None

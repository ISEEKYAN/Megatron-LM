# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import asdict

import pytest
from megatron.lite.primitive.optimizers.vision_config import (
    OptimizerConfig,
    VisionOptimizerConfig,
)


@pytest.mark.parametrize('field', range(3))
@pytest.mark.parametrize('value', [-1.0, float('inf'), float('nan')])
def test_visual_policy_rejects_invalid_numeric_values(field, value):
    values = [0.5, 1.0, 0.0]
    values[field] = value
    with pytest.raises(ValueError, match='finite nonnegative'):
        VisionOptimizerConfig(*values)


def test_visual_optimizer_policy_roundtrip():
    config = OptimizerConfig(
        0.01, 5, 'quintic', vision_policy=VisionOptimizerConfig(0, 1, 0)
    )
    serialized = asdict(config)
    serialized['vision_policy'] = VisionOptimizerConfig(**serialized['vision_policy'])
    assert OptimizerConfig(**serialized) == config

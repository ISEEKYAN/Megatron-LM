# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit optimizer configuration for visual and mixed parameter groups."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionOptimizerConfig:
    """Caller-selected post-training LR/decay, not pretraining defaults.

    All numeric policy is explicit; algorithm and parameter ownership
    are selected by the caller.
    """

    encoder_lr_multiplier: float
    image_vector_lr_multiplier: float
    image_vector_weight_decay: float

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in vars(self).values()):
            raise ValueError(
                'Visual optimizer policy requires finite nonnegative values'
            )


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float
    ns_steps: int
    coefficient_type: str
    clip_grad: float = 1.0
    vision_policy: VisionOptimizerConfig | None = None

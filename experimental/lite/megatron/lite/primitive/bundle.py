# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ModelBundle — return type of protocol.build_model()."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from megatron.lite.primitive.parallel.state import ParallelState


@dataclass
class ModelBundle:
    """Everything runtime needs to run a training loop.

    Returned by protocol.build_model(). Model owns the construction
    of all fields — runtime just consumes them.

    An optimizer may declare ``owns_param_group_policy = True``: its initial
    per-group LR ratios (relative to the configured base LR) and weight decay
    belong to the model. A connector scheduler must preserve those ratios and
    fixed decay, regardless of optimizer name. The base LR must be positive;
    nonconstant weight-decay scheduling is unsupported for this policy.
    Missing/False retains the connector's legacy scheduling policy.
    """

    chunks: list[nn.Module]
    parallel_state: ParallelState
    optimizer: Any | None = None
    finalize_grads: Callable[[], None] | None = None
    forward_step: Callable[..., dict] | None = None
    # extra metadata (expert_classifier, model_cfg, etc.)
    extras: dict[str, Any] = field(default_factory=dict)

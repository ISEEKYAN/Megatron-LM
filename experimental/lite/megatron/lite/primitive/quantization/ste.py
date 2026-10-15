# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Version-independent reference VJPs for deployment-visible arithmetic."""

import torch


def clamp_with_inclusive_vjp(x, *, min=None, max=None):
    """Keep the historical unit input derivative at both clamp endpoints."""
    clipped = x.clamp(min=min, max=max)
    inside = torch.ones_like(x, dtype=torch.bool)
    if min is not None:
        inside = inside & (x >= min)
    if max is not None:
        inside = inside & (x <= max)
    return torch.where(inside, x, clipped.detach())

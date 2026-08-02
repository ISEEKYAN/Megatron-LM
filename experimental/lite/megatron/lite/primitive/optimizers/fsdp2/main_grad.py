# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MLite-owned gradient accessors for the FSDP2 optimizer primitive."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

_MAIN_GRAD_STATE_ATTR = "_mlite_fsdp2_main_grad_state"


def get_param_grad(param: nn.Parameter) -> torch.Tensor | None:
    """Return the active MLite main gradient, otherwise PyTorch's gradient."""

    state = getattr(param, _MAIN_GRAD_STATE_ATTR, None)
    if state is None:
        return param.grad
    if not bool(getattr(state, "active", False)):
        return None
    main_grad = getattr(param, "main_grad", None)
    if not isinstance(main_grad, torch.Tensor):
        raise RuntimeError("FSDP2 main_grad state is active but its view is missing.")
    if main_grad.dtype is not torch.float32:
        raise RuntimeError(f"FSDP2 main_grad must be float32, got {main_grad.dtype}.")
    return main_grad


def zero_main_grads(params: Iterable[nn.Parameter]) -> None:
    """Zero each unique flat main-gradient buffer exactly once."""

    states: dict[int, Any] = {}
    for param in params:
        state = getattr(param, _MAIN_GRAD_STATE_ATTR, None)
        if state is not None:
            states[id(state)] = state
    for state in states.values():
        state.zero()


__all__ = ["get_param_grad", "zero_main_grads"]

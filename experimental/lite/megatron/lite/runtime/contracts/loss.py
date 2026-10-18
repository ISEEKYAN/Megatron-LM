# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Loss-side runtime context, kept separate from batch data contracts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class LossContext:
    """Per-microbatch loss/output policy for model-owned loss helpers."""

    temperature: float = 1.0
    calculate_entropy: bool = False
    return_log_probs: bool = True
    loss_scale: float = 1.0
    source_batch: Any | None = None
    normalization_denominator: float | None = None
    # Set by the runtime when an external loss_fn owns normalization, so a
    # model's native loss (and its denominator) is not consumed.
    external_normalization: bool = False


_CURRENT_LOSS_CONTEXT: ContextVar[LossContext | None] = ContextVar(
    "megatron_lite_loss_context", default=None
)


def get_loss_context() -> LossContext | None:
    return _CURRENT_LOSS_CONTEXT.get()


@contextmanager
def use_loss_context(loss_context: LossContext | None) -> Iterator[None]:
    token = _CURRENT_LOSS_CONTEXT.set(loss_context)
    try:
        yield
    finally:
        _CURRENT_LOSS_CONTEXT.reset(token)


def split_loss_context(item):
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and (item[1] is None or isinstance(item[1], LossContext))
    ):
        return item
    return item, None


def mark_external_normalization(data_iter):
    """Mark microbatch contexts whose loss an external loss_fn computes.

    Items without a LossContext are passed through unchanged, keeping the
    two-argument loss_fn call for callers that never supplied one.
    """
    for item in data_iter:
        batch, context = split_loss_context(item)
        yield (
            item
            if context is None
            else (batch, replace(context, external_normalization=True))
        )


__all__ = [
    "LossContext",
    "get_loss_context",
    "mark_external_normalization",
    "split_loss_context",
    "use_loss_context",
]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deterministic adjacent-microbatch ordering for expert-parallel operators."""

from __future__ import annotations

import threading
from contextlib import contextmanager

_CONDITION = threading.Condition()
_ACTIVE = False
_NEXT = "backward"


@contextmanager
def adjacent_microbatch_ep_overlap():
    """Activate backward/forward EP turns for one adjacent microbatch pair."""
    global _ACTIVE, _NEXT
    with _CONDITION:
        if _ACTIVE:
            raise RuntimeError("Adjacent microbatch EP overlap cannot be nested.")
        _ACTIVE = True
        _NEXT = "backward"
        _CONDITION.notify_all()
    try:
        yield
    finally:
        with _CONDITION:
            _ACTIVE = False
            _NEXT = "backward"
            _CONDITION.notify_all()


@contextmanager
def ep_overlap_turn(direction: str):
    """Keep EP calls ordered identically while adjacent F/B execute concurrently."""
    global _NEXT
    if direction not in ("forward", "backward"):
        raise ValueError(f"Unknown EP overlap direction: {direction}")
    acquired = False
    with _CONDITION:
        while _ACTIVE and _NEXT != direction:
            if not _CONDITION.wait(timeout=300):
                raise RuntimeError(f"Timed out waiting for adjacent EP {direction} turn.")
        acquired = _ACTIVE
    try:
        yield
    finally:
        if acquired:
            with _CONDITION:
                _NEXT = "forward" if direction == "backward" else "backward"
                _CONDITION.notify_all()


__all__ = ["adjacent_microbatch_ep_overlap", "ep_overlap_turn"]

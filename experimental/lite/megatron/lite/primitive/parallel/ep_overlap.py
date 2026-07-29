# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Deterministic DeepEP submission order for adjacent forward/backward."""

from __future__ import annotations

import threading
from contextlib import contextmanager

_condition = threading.Condition()
_active = False
_next = "backward"
_finished: set[str] = set()
_role = threading.local()


@contextmanager
def adjacent_ep_overlap():
    """Allow one backward and one forward to share DeepEP in a fixed order."""
    global _active, _next
    with _condition:
        if _active:
            raise RuntimeError("Adjacent EP overlap cannot be nested.")
        _active = True
        _next = "backward"
        _finished.clear()
    try:
        yield
    finally:
        with _condition:
            _active = False
            _next = "backward"
            _finished.clear()
            _condition.notify_all()


@contextmanager
def ep_overlap_role(role: str):
    """Mark all DeepEP submissions made by this thread as forward or backward."""
    if role not in ("forward", "backward"):
        raise ValueError(f"Unknown EP overlap role: {role}")
    previous = getattr(_role, "value", None)
    _role.value = role
    try:
        yield
    finally:
        with _condition:
            if _active:
                _finished.add(role)
                _condition.notify_all()
        _role.value = previous


@contextmanager
def ordered_ep_submission():
    """Serialize only DeepEP submissions; computation remains concurrent."""
    global _next
    role = getattr(_role, "value", None)
    acquired = False
    with _condition:
        while _active and role is not None and _next != role and _next not in _finished:
            if not _condition.wait(timeout=300):
                raise RuntimeError(f"Timed out waiting for adjacent EP {role} submission.")
        acquired = _active and role is not None
    try:
        yield
    finally:
        if acquired:
            with _condition:
                other = "forward" if role == "backward" else "backward"
                _next = role if other in _finished else other
                _condition.notify_all()


__all__ = ["adjacent_ep_overlap", "ep_overlap_role", "ordered_ep_submission"]

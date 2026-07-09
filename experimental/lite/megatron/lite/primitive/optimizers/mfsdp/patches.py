# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small parameter annotations consumed by the local M-FSDP implementation.

The PR #26 implementation installed these decisions by monkey-patching MCore.
The self-contained implementation owns the call sites, so the annotations stay
explicit and no external namespace is mutated.
"""

from __future__ import annotations

from typing import Any


SKIP_TP_DUPLICATE_SYNC_ATTR = "_mlite_mfsdp_skip_tp_duplicate_sync"
PARAM_NAME_ATTR = "_mlite_mfsdp_param_name"


def mark_skip_tp_duplicate_sync(param: Any) -> None:
    setattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR, True)


def clear_skip_tp_duplicate_sync(param: Any) -> None:
    if hasattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR):
        delattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR)


def set_mfsdp_param_name(param: Any, name: str) -> None:
    setattr(param, PARAM_NAME_ATTR, name)


__all__ = [
    "PARAM_NAME_ATTR",
    "SKIP_TP_DUPLICATE_SYNC_ATTR",
    "clear_skip_tp_duplicate_sync",
    "mark_skip_tp_duplicate_sync",
    "set_mfsdp_param_name",
]

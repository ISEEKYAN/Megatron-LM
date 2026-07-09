# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parameter metadata used by the native M-FSDP hot path."""

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


def should_skip_tp_duplicate_sync(param: Any) -> bool:
    if not bool(getattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR, False)):
        return False
    return getattr(param, "_tensor_parallel_mode", None) not in ("column", "row")


def install_mfsdp_tp_duplicate_sync_patch() -> None:
    """Compatibility no-op: the native path owns parameter sharding directly."""


def install_mfsdp_start_param_sync_patch() -> None:
    """Compatibility no-op: :class:`MFSdpModule` owns param synchronization."""


def install_mfsdp_param_sync_debug_patch(_param_and_grad_buffer: Any) -> None:
    """Compatibility no-op retained for callers of the former patch surface."""


__all__ = [
    "PARAM_NAME_ATTR",
    "SKIP_TP_DUPLICATE_SYNC_ATTR",
    "clear_skip_tp_duplicate_sync",
    "install_mfsdp_param_sync_debug_patch",
    "install_mfsdp_start_param_sync_patch",
    "install_mfsdp_tp_duplicate_sync_patch",
    "mark_skip_tp_duplicate_sync",
    "set_mfsdp_param_name",
    "should_skip_tp_duplicate_sync",
]

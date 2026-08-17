# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared Qwen3MoE model helpers."""

from __future__ import annotations


def decode_bank_surface(name: str) -> str | None:
    """Decode one model-owned multi-LoRA bank key to its native surface."""
    marker = "bank_"
    if marker not in name:
        return None
    encoded_factor = name.rsplit(".", 1)[-1]
    if not encoded_factor.startswith(marker):
        return None
    try:
        encoded, factor = encoded_factor[len(marker) :].rsplit("_", 1)
    except ValueError:
        return None
    if factor not in {"a", "b"}:
        return None
    try:
        return bytes.fromhex(encoded).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return None


def is_expert_param(name: str) -> bool:
    # Model-owned LoRA banks are shared adapter state.  Their optimizer owner
    # group is encoded on the Parameter, while checkpoints always use dense
    # replica semantics regardless of the native FC surface.
    if decode_bank_surface(name) is not None:
        return False
    return "experts" in name and "router" not in name


__all__ = ["decode_bank_surface", "is_expert_param"]

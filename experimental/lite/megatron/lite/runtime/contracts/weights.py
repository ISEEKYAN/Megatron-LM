# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contracts for model-to-rollout weight streams."""

from __future__ import annotations

from enum import Enum


class ResyncFormat(str, Enum):
    BF16 = "bf16"
    VLLM_CHECKPOINT = "vllm_checkpoint"

    @classmethod
    def parse(cls, value: str | ResyncFormat) -> ResyncFormat:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unsupported resync_format={value!r}; expected one of: {supported}"
            ) from exc


__all__ = ["ResyncFormat"]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native DeepSeek V4 (ds4flash) lite implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model

__all__ = ["DeepseekV4Model"]


def __getattr__(name: str):
    if name == "DeepseekV4Model":
        from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model

        return DeepseekV4Model
    raise AttributeError(name)

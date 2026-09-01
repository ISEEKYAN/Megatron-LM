# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native DeepSeek V4 (ds4flash) lite implementation."""

__all__ = ["DeepseekV4Model"]


def __getattr__(name: str):
    """Keep protocol-only CPU checks independent of Transformer Engine."""
    if name == "DeepseekV4Model":
        from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model

        return DeepseekV4Model
    raise AttributeError(name)

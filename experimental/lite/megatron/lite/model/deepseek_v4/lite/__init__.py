# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native DeepSeek V4 (ds4flash) lite implementation."""


def __getattr__(name):
    if name == "DeepseekV4Model":
        from .model import DeepseekV4Model

        return DeepseekV4Model
    raise AttributeError(name)


__all__ = ["DeepseekV4Model"]

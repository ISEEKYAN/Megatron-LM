# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense Qwen2 model package.

The lite implementation is the local exact-route slice for the paper target
family: TP=1 runtime, HF checkpoint mapping, PEFT LoRA adapter lifecycle, and
OLoRA-tail initialization.
"""

from megatron.lite.model.qwen2.config import Qwen2Config

__all__ = ["Qwen2Config"]

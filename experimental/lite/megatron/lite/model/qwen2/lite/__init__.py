# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense Qwen2 lite runtime package."""

from megatron.lite.model.qwen2.lite.model import Qwen2ForCausalLM, Qwen2Model

__all__ = ["Qwen2ForCausalLM", "Qwen2Model"]

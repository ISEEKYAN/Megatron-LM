# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Declarative LoRA targets for Qwen3.5-specific modules."""

from megatron.lite.primitive.modules.lora_apply import LoraTargetRule

LORA_TARGETS = (
    LoraTargetRule("SharedExpert", "gate_up", "linear_fc1"),
    LoraTargetRule("SharedExpert", "down", "linear_fc2"),
    LoraTargetRule("GatedDeltaNet", "in_proj", "linear_qkv"),
    LoraTargetRule("GatedDeltaNet", "o_proj", "linear_proj"),
)

__all__ = ["LORA_TARGETS"]

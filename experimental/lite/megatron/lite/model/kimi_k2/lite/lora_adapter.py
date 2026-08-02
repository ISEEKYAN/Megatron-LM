# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Declarative LoRA targets for Kimi K2-specific modules."""

from megatron.lite.primitive.modules.lora_apply import LoraTargetRule

LORA_TARGETS = (
    LoraTargetRule("DenseMLP", "gate_up", "linear_fc1"),
    LoraTargetRule("DenseMLP", "down", "linear_fc2"),
    LoraTargetRule("SharedExpert", "gate_up", "linear_fc1"),
    LoraTargetRule("SharedExpert", "down", "linear_fc2"),
    LoraTargetRule("MultiLatentAttention", "linear_q_down_proj", "linear_qkv"),
    LoraTargetRule("MultiLatentAttention", "linear_q_up_proj", "linear_qkv"),
    LoraTargetRule("MultiLatentAttention", "linear_kv_down_proj", "linear_qkv"),
    LoraTargetRule("MultiLatentAttention", "linear_kv_up_proj", "linear_qkv"),
    LoraTargetRule("MultiLatentAttention", "linear_proj", "linear_proj"),
)

__all__ = ["LORA_TARGETS"]

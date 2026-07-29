# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Declarative LoRA targets for GLM-5-specific modules."""

from megatron.lite.primitive.modules.lora_apply import LoraTargetRule

LORA_TARGETS = (
    LoraTargetRule("DenseMLP", "gate_up", "linear_fc1"),
    LoraTargetRule("DenseMLP", "down", "linear_fc2"),
    LoraTargetRule("SharedExpert", "gate_up", "linear_fc1"),
    LoraTargetRule("SharedExpert", "down", "linear_fc2"),
    LoraTargetRule("DynamicSparseAttention", "q_a_proj", "linear_qkv"),
    LoraTargetRule("DynamicSparseAttention", "q_b_proj", "linear_qkv"),
    LoraTargetRule("DynamicSparseAttention", "kv_a_proj_with_mqa", "linear_qkv"),
    LoraTargetRule("DynamicSparseAttention", "kv_b_proj", "linear_qkv"),
    LoraTargetRule("DynamicSparseAttention", "o_proj", "linear_proj"),
)

__all__ = ["LORA_TARGETS"]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Declarative LoRA targets for DeepSeek V4-specific modules."""

from megatron.lite.primitive.modules.lora_apply import LoraTargetRule

LORA_TARGETS = (
    LoraTargetRule("CompressedSparseAttention", "wq_a", "linear_qkv"),
    LoraTargetRule("CompressedSparseAttention", "wq_b", "linear_qkv"),
    LoraTargetRule("CompressedSparseAttention", "wkv", "linear_qkv"),
    LoraTargetRule("CompressedSparseAttention", "wo_a", "linear_proj"),
    LoraTargetRule("CompressedSparseAttention", "wo_b", "linear_proj"),
)

__all__ = ["LORA_TARGETS"]

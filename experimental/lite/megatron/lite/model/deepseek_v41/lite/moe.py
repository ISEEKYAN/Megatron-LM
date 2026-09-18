# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model assembly uses shared experts and the core bias update operator."""
from megatron.lite.primitive.modules.modality_moe import ModalityLoad, ModalityRouter
from megatron.lite.primitive.modules.modality_moe import RoutedExperts as DeepseekV41MoE
from megatron.lite.primitive.modules.modality_moe import (
    SwiGLUExpert,
    reduce_modality_load,
)

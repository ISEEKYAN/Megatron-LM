# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional kernel shims used by MLite primitives."""

from __future__ import annotations

from .vllm_ds4 import (
    DS4KVInsertAdapter,
    DS4TopKAdapter,
    DeepEPAdapter,
    DeepEPMode,
    FlashMLAAdapter,
    FusedExpertsAdapter,
    FusedQKVRMSNormAdapter,
    GateLinearAdapter,
    GroupedDeepGemmExpertsAdapter,
    GroupedFP8ExpertWeights,
    HashRouteAdapter,
    KVCacheLayout,
    MHCKernel,
    MHCTileLangAdapter,
    SharedExpertsAdapter,
)

__all__ = [
    "DS4KVInsertAdapter",
    "DS4TopKAdapter",
    "DeepEPAdapter",
    "DeepEPMode",
    "FlashMLAAdapter",
    "FusedExpertsAdapter",
    "FusedQKVRMSNormAdapter",
    "GateLinearAdapter",
    "GroupedDeepGemmExpertsAdapter",
    "GroupedFP8ExpertWeights",
    "HashRouteAdapter",
    "KVCacheLayout",
    "MHCKernel",
    "MHCTileLangAdapter",
    "SharedExpertsAdapter",
]

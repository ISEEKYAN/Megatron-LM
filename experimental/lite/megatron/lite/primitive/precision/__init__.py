# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Public entrypoints for the closed MLite precision primitive."""

from megatron.lite.primitive.precision.contract import (
    AuthoritativeSource,
    MasterOwner,
    ParameterContract,
    PrecisionDType,
    PrecisionImplementation,
    PrimitiveCapability,
    SemanticSite,
    WeightStorage,
)
from megatron.lite.primitive.precision.coverage import (
    CoverageEntry,
    CoverageManifest,
    PrecisionCoverage,
)
from megatron.lite.primitive.precision.hopper_blockwise import (
    FP8_WEIGHT_SUPPORTED_OPTIMIZERS,
    PRECISION_NAMES,
    PrecisionPhase,
    active_precision,
    build_hopper_blockwise_recipe,
    precision_forward_context,
    precision_model_init_context,
    precision_site_forward_context,
    require_optimizer_supports_precision,
    resolve_precision,
    validate_hopper_environment,
)

__all__ = [
    "AuthoritativeSource",
    "CoverageEntry",
    "CoverageManifest",
    "FP8_WEIGHT_SUPPORTED_OPTIMIZERS",
    "MasterOwner",
    "PRECISION_NAMES",
    "ParameterContract",
    "PrecisionCoverage",
    "PrecisionDType",
    "PrecisionImplementation",
    "PrecisionPhase",
    "PrimitiveCapability",
    "SemanticSite",
    "WeightStorage",
    "active_precision",
    "build_hopper_blockwise_recipe",
    "precision_forward_context",
    "precision_model_init_context",
    "precision_site_forward_context",
    "require_optimizer_supports_precision",
    "resolve_precision",
    "validate_hopper_environment",
]

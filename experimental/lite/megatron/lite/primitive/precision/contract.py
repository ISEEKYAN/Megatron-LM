# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Closed precision records shared by MLite runtime and primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SemanticSite(str, Enum):
    """Typed model-composition sites relevant to the closed FP8 profiles."""

    ATTENTION_PROJECTION = "attention_projection"
    DENSE_MLP = "dense_mlp"
    MOE_EXPERT = "moe_expert"
    ATTENTION_CORE = "attention_core"
    ROUTER = "router"
    NORM = "norm"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"


class PrimitiveCapability(str, Enum):
    """TE primitive capabilities that may satisfy an FP8 coverage requirement."""

    TE_LINEAR = "te_linear"
    TE_LAYERNORM_LINEAR = "te_layernorm_linear"
    TE_GROUPED_LINEAR = "te_grouped_linear"


class WeightStorage(str, Enum):
    """Storage used by selected GEMM compute parameters."""

    BF16 = "bf16"
    FP8_BLOCKWISE_E4M3 = "fp8_blockwise_e4m3"


class PrecisionDType(str, Enum):
    """Dtypes at the parameter, gradient, optimizer, and communication boundaries."""

    BF16 = "bf16"
    FP32 = "fp32"


class AuthoritativeSource(str, Enum):
    """Source from which compute and master parameters must be initialized."""

    HIGH_PRECISION = "high_precision_source"


class MasterOwner(str, Enum):
    """The single owner of the authoritative FP32 optimizer master."""

    MLITE_OPTIMIZER = "mlite_optimizer"


@dataclass(frozen=True, slots=True)
class ParameterContract:
    """Precision and ownership boundaries for one closed profile."""

    compute_weight: WeightStorage
    authoritative_load_source: AuthoritativeSource
    master_parameter: PrecisionDType
    main_gradient: PrecisionDType
    optimizer_state: PrecisionDType
    parameter_all_gather: PrecisionDType
    master_owner: MasterOwner


@dataclass(frozen=True, slots=True)
class PrecisionImplementation:
    """One immutable, closed precision profile implementation."""

    name: str
    recipe_factory: Callable[[], Any]
    fp8_sites: frozenset[SemanticSite]
    bf16_sites: frozenset[SemanticSite]
    parameter_contract: ParameterContract

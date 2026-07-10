# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The two closed Hopper blockwise FP8 profiles and their runtime gates."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Iterator

from megatron.lite.primitive.precision.contract import (
    AuthoritativeSource,
    MasterOwner,
    ParameterContract,
    PrecisionDType,
    PrecisionImplementation,
    SemanticSite,
    WeightStorage,
)

FROZEN_TRANSFORMER_ENGINE_VERSION = "2.18.0.dev0"
FROZEN_TRANSFORMER_ENGINE_COMMIT = "8b9968255eb879e6e390f427836906b29aad64d2"
_FROZEN_TRANSFORMER_ENGINE_SHORT_COMMIT = FROZEN_TRANSFORMER_ENGINE_COMMIT[:7]

PRECISION_NAMES = (
    "bf16",
    "hopper_blockwise_bf16_weight",
    "hopper_blockwise_fp8_weight",
)

_FP8_SITES = frozenset(
    {
        SemanticSite.ATTENTION_PROJECTION,
        SemanticSite.DENSE_MLP,
        SemanticSite.MOE_EXPERT,
    }
)
_BF16_SITES = frozenset(
    {
        SemanticSite.ATTENTION_CORE,
        SemanticSite.ROUTER,
        SemanticSite.NORM,
        SemanticSite.EMBEDDING,
        SemanticSite.LM_HEAD,
    }
)


def _parameter_contract(compute_weight: WeightStorage) -> ParameterContract:
    return ParameterContract(
        compute_weight=compute_weight,
        authoritative_load_source=AuthoritativeSource.HIGH_PRECISION,
        master_parameter=PrecisionDType.FP32,
        main_gradient=PrecisionDType.FP32,
        optimizer_state=PrecisionDType.FP32,
        parameter_all_gather=PrecisionDType.BF16,
        master_owner=MasterOwner.MLITE_OPTIMIZER,
    )


def _validate_recipe_environment() -> None:
    scale_override = os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES")
    if scale_override not in (None, "0"):
        raise RuntimeError(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES must be unset or 0 for Hopper blockwise FP8."
        )
    if os.getenv("NVTE_BACKWARD_OVERRIDE") is not None:
        raise RuntimeError(
            "NVTE_BACKWARD_OVERRIDE must be unset for Hopper blockwise FP8."
        )


@lru_cache(maxsize=1)
def _build_hopper_blockwise_recipe() -> Any:
    _validate_recipe_environment()
    from transformer_engine.common.recipe import Float8BlockScaling, Format, MMParams

    recipe = Float8BlockScaling(
        use_f32_scales=False,
        fp8_format=Format.E4M3,
        x_block_scaling_dim=1,
        w_block_scaling_dim=2,
        grad_block_scaling_dim=1,
        fp8_gemm_fprop=MMParams(use_split_accumulator=True),
        fp8_gemm_dgrad=MMParams(use_split_accumulator=True),
        fp8_gemm_wgrad=MMParams(use_split_accumulator=True),
        fp8_dpa=False,
        fp8_mha=False,
        backward_override=None,
    )
    return recipe


def _validate_hopper_blockwise_recipe(recipe: Any) -> None:
    from transformer_engine.common.recipe import Format

    quantizers = (
        recipe.fp8_quant_fwd_inp,
        recipe.fp8_quant_fwd_weight,
        recipe.fp8_quant_bwd_grad,
    )
    matches_frozen_recipe = (
        recipe.use_f32_scales is False
        and recipe.fp8_format is Format.E4M3
        and recipe.x_block_scaling_dim == 1
        and recipe.w_block_scaling_dim == 2
        and recipe.grad_block_scaling_dim == 1
        and recipe.fp8_gemm_fprop.use_split_accumulator is True
        and recipe.fp8_gemm_dgrad.use_split_accumulator is True
        and recipe.fp8_gemm_wgrad.use_split_accumulator is True
        and recipe.fp8_dpa is False
        and recipe.fp8_mha is False
        and recipe.backward_override is None
        and all(quantizer.power_2_scale for quantizer in quantizers)
    )
    if not matches_frozen_recipe:
        raise RuntimeError("Hopper blockwise FP8 recipe no longer matches the frozen contract.")


def build_hopper_blockwise_recipe() -> Any:
    """Return the single frozen Transformer Engine blockwise recipe."""

    recipe = _build_hopper_blockwise_recipe()
    _validate_hopper_blockwise_recipe(recipe)
    return recipe


HOPPER_BLOCKWISE_BF16_WEIGHT = PrecisionImplementation(
    name="hopper_blockwise_bf16_weight",
    recipe_factory=build_hopper_blockwise_recipe,
    fp8_sites=_FP8_SITES,
    bf16_sites=_BF16_SITES,
    parameter_contract=_parameter_contract(WeightStorage.BF16),
)
HOPPER_BLOCKWISE_FP8_WEIGHT = PrecisionImplementation(
    name="hopper_blockwise_fp8_weight",
    recipe_factory=build_hopper_blockwise_recipe,
    fp8_sites=_FP8_SITES,
    bf16_sites=_BF16_SITES,
    parameter_contract=_parameter_contract(WeightStorage.FP8_BLOCKWISE_E4M3),
)


def resolve_precision(name: str) -> PrecisionImplementation | None:
    """Resolve one of the three accepted public precision names."""

    if not isinstance(name, str):
        raise TypeError("precision must be a string")
    if name == "bf16":
        return None
    if name == HOPPER_BLOCKWISE_BF16_WEIGHT.name:
        return HOPPER_BLOCKWISE_BF16_WEIGHT
    if name == HOPPER_BLOCKWISE_FP8_WEIGHT.name:
        return HOPPER_BLOCKWISE_FP8_WEIGHT
    accepted = ", ".join(PRECISION_NAMES)
    raise ValueError(f"precision must be one of: {accepted}; got {name!r}")


@dataclass(frozen=True, slots=True)
class _HopperEnvironment:
    compute_capability: tuple[int, int]
    transformer_engine_version: str
    cuda_version: tuple[int, int]
    cublas_version: int
    block_scaling_supported: bool
    unsupported_reason: str


def _parse_cuda_version(value: str | None) -> tuple[int, int]:
    if not value:
        return (0, 0)
    parts = value.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse CUDA version {value!r}.") from exc


def _probe_hopper_environment() -> _HopperEnvironment:
    import torch
    import transformer_engine
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.quantization import check_fp8_block_scaling_support

    supported, reason = check_fp8_block_scaling_support()
    return _HopperEnvironment(
        compute_capability=tuple(torch.cuda.get_device_capability()),
        transformer_engine_version=str(transformer_engine.__version__),
        cuda_version=_parse_cuda_version(torch.version.cuda),
        cublas_version=int(tex.get_cublasLt_version()),
        block_scaling_supported=bool(supported),
        unsupported_reason=str(reason),
    )


def validate_hopper_environment() -> _HopperEnvironment:
    """Fail before model allocation unless the frozen Hopper environment is present."""

    _validate_recipe_environment()
    environment = _probe_hopper_environment()
    if environment.compute_capability != (9, 0):
        raise RuntimeError(
            "Hopper blockwise FP8 requires exactly SM90; found compute capability "
            f"{environment.compute_capability}."
        )

    version, separator, local = environment.transformer_engine_version.partition("+")
    if version != FROZEN_TRANSFORMER_ENGINE_VERSION:
        raise RuntimeError(
            "Hopper blockwise FP8 requires Transformer Engine "
            f"{FROZEN_TRANSFORMER_ENGINE_VERSION}; found {environment.transformer_engine_version}."
        )
    if separator and (
        len(local) < len(_FROZEN_TRANSFORMER_ENGINE_SHORT_COMMIT)
        or not FROZEN_TRANSFORMER_ENGINE_COMMIT.startswith(local)
    ):
        raise RuntimeError(
            "Hopper blockwise FP8 requires Transformer Engine commit "
            f"{_FROZEN_TRANSFORMER_ENGINE_SHORT_COMMIT}; found local version {local!r}."
        )
    if environment.cuda_version < (12, 9):
        raise RuntimeError(
            "Hopper blockwise FP8 requires CUDA 12.9 or newer; found "
            f"{environment.cuda_version[0]}.{environment.cuda_version[1]}."
        )
    if environment.cublas_version < 130400:
        raise RuntimeError(
            "Hopper blockwise FP8 MoE requires cuBLAS 13.4 or newer; found encoded version "
            f"{environment.cublas_version}."
        )
    if not environment.block_scaling_supported:
        reason = (
            environment.unsupported_reason
            or "Transformer Engine rejected block scaling"
        )
        raise RuntimeError(f"Hopper blockwise FP8 is unavailable: {reason}")
    return environment


class PrecisionPhase(str, Enum):
    """Runtime phase in which precision-aware primitives may operate."""

    MODEL_INIT = "model-init"
    FORWARD = "forward"


_ACTIVE_PRECISION: ContextVar[tuple[PrecisionPhase, PrecisionImplementation] | None] = (
    ContextVar("mlite_active_precision", default=None)
)


@contextmanager
def _precision_context(
    phase: PrecisionPhase, implementation: PrecisionImplementation
) -> Iterator[PrecisionImplementation]:
    if not isinstance(implementation, PrecisionImplementation):
        raise TypeError("precision context requires a PrecisionImplementation")
    token = _ACTIVE_PRECISION.set((phase, implementation))
    try:
        yield implementation
    finally:
        _ACTIVE_PRECISION.reset(token)


def precision_model_init_context(
    implementation: PrecisionImplementation,
) -> AbstractContextManager[PrecisionImplementation]:
    """Bind one implementation while model composition declares typed coverage."""

    return _precision_context(PrecisionPhase.MODEL_INIT, implementation)


def precision_forward_context(
    implementation: PrecisionImplementation,
) -> AbstractContextManager[PrecisionImplementation]:
    """Bind one implementation for correctly scoped primitive forward contexts."""

    return _precision_context(PrecisionPhase.FORWARD, implementation)


def precision_site_forward_context(
    implementation: PrecisionImplementation,
    site: SemanticSite,
) -> AbstractContextManager[None]:
    """Open TE FP8 autocast for one selected semantic GEMM site only."""

    if not isinstance(site, SemanticSite):
        raise TypeError("precision site must be a SemanticSite")
    active = active_precision(PrecisionPhase.FORWARD)
    if active is not implementation:
        raise RuntimeError(
            "precision primitive is bound to a different runtime forward context"
        )
    if site in implementation.bf16_sites:
        raise ValueError(f"fixed BF16 site {site.value} cannot enter FP8 autocast")
    if site not in implementation.fp8_sites:
        raise ValueError(
            f"site {site.value} is not selected by {implementation.name}"
        )

    import transformer_engine.pytorch as te

    return te.fp8_autocast(
        enabled=True,
        fp8_recipe=implementation.recipe_factory(),
        fp8_group=None,
    )


def active_precision(phase: PrecisionPhase) -> PrecisionImplementation:
    """Return the runtime-bound implementation for an exact lifecycle phase."""

    active = _ACTIVE_PRECISION.get()
    if active is None or active[0] is not phase:
        raise RuntimeError(
            f"precision primitive requires the runtime {phase.value} context"
        )
    return active[1]

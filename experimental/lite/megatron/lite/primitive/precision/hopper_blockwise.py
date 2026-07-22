# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The two closed Hopper blockwise FP8 profiles and their runtime gates."""

from __future__ import annotations

import os
import warnings
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

# The canonical training image ships a released Transformer Engine on which the
# parity evidence was recorded. Blockwise FP8 runs on the *same* image that runs
# BF16 -- no FP8-only overlay -- so this version is recorded for provenance, not
# used as a hard runtime gate. The only authoritative runtime gate is Transformer
# Engine's own capability probe (``check_fp8_block_scaling_support``); pinning an
# exact TE/CUDA/SM version would recreate a special-environment requirement and
# could reject FP8 on a newer or different accelerator where BF16 runs fine. A
# probed toolchain that differs from this canonical version emits a fail-loud
# provenance warning but is not blocked.
CANONICAL_TRANSFORMER_ENGINE_VERSION = "2.15.0"

# Minimum cuBLASLt version for the blockwise *GroupedLinear* (MoE expert) path.
# TE guards grouped GEMM with ``CUBLAS_GROUPED_GEMM_VERSION`` and requires cuBLAS
# >= 13.3, a requirement the general block-scaling capability probe does not
# cover: an environment can pass ``check_fp8_block_scaling_support`` (blockwise
# Linear/LayerNormLinear work) while grouped GEMM crashes for lack of the grouped
# kernel. A MoE FP8 profile therefore gates this requirement independently and
# fails loud before allocation instead of dying inside the grouped GEMM. Encoded
# like ``tex.get_cublasLt_version()`` (major*10000 + minor*100 + patch): 13.3 ->
# 130300; the canonical qualification image reports 130401 (cuBLAS 13.4.1). This
# is *not* a device/CUDA/TE version pin (that hard gate was removed): it is TE's
# own grouped-GEMM hard requirement, scoped to profiles that select MoE experts.
# Source: TransformerEngine v2.15 common/gemm/cublaslt_grouped_gemm.cu.
CUBLAS_GROUPED_GEMM_MIN_VERSION = 130300

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


# Optimizer backends whose parameter write-back can currently own FP8 *compute*
# weights. The distributed optimizer needs the upstream Megatron
# ``fp8_param_gather`` path to gather and re-quantize FP8 compute parameters
# during its sharded update; that path is deliberately out of scope for the
# closed Hopper profiles. A DistOpt run against an FP8-weight profile would
# silently update only the BF16-gathered parameters while the FP8 compute
# weights stay stale -- a run that looks FP8-trained but is not. Only FSDP2 keeps
# the FP8 compute weights live through its per-parameter update today.
FP8_WEIGHT_SUPPORTED_OPTIMIZERS = frozenset({"fsdp2"})


def require_optimizer_supports_precision(
    implementation: PrecisionImplementation, optimizer_backend: str
) -> None:
    """Fail loud on optimizer/precision combinations that cannot train correctly.

    BF16-weight profiles work under every optimizer backend. FP8 *compute*
    weights currently require the FSDP2 optimizer: the distributed-optimizer
    write-back for FP8 weights depends on the upstream Megatron
    ``fp8_param_gather`` path, which is out of scope for the closed Hopper
    profiles. Constructing a distributed-optimizer run against an FP8-weight
    profile is therefore rejected here instead of silently producing a run that
    looks FP8-trained but leaves the FP8 compute weights stale.
    """

    if implementation.parameter_contract.compute_weight is WeightStorage.BF16:
        return
    if optimizer_backend in FP8_WEIGHT_SUPPORTED_OPTIMIZERS:
        return
    supported = ", ".join(sorted(FP8_WEIGHT_SUPPORTED_OPTIMIZERS))
    raise ValueError(
        f"{implementation.name} stores FP8 compute weights, which the "
        f"{optimizer_backend!r} optimizer cannot train correctly: the "
        "distributed-optimizer FP8 write-back requires the upstream Megatron "
        "fp8_param_gather path, which is out of scope for the closed Hopper "
        f"profiles. Use a supported optimizer ({supported}) for FP8-weight "
        "training, or select the hopper_blockwise_bf16_weight profile."
    )


def resolve_precision(name: str) -> PrecisionImplementation | None:
    """Resolve one of the accepted public precision names.

    ``bf16`` disables managed precision. The two Hopper profiles share the
    frozen recipe and typed coverage but differ only in selected GEMM weight
    storage.
    """

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


def _warn_on_unvalidated_toolchain(environment: _HopperEnvironment) -> None:
    """Emit a non-blocking provenance warning when the toolchain is not canonical.

    Honours the environment rule "whatever runs BF16 must run FP8": a differing
    Transformer Engine version is *not* a hard gate (the capability probe already
    proved blockwise FP8 works), but it is surfaced loudly so a run on an
    unvalidated toolchain is never silent.
    """

    version, _separator, _local = environment.transformer_engine_version.partition("+")
    if version != CANONICAL_TRANSFORMER_ENGINE_VERSION:
        warnings.warn(
            "Blockwise FP8 is running on Transformer Engine "
            f"{environment.transformer_engine_version}, which differs from the "
            f"canonical validated version {CANONICAL_TRANSFORMER_ENGINE_VERSION}. "
            "The capability probe succeeded so this is allowed, but parity evidence "
            "was recorded against the canonical version.",
            RuntimeWarning,
            stacklevel=2,
        )


def validate_hopper_environment(
    implementation: PrecisionImplementation | None = None,
) -> _HopperEnvironment:
    """Fail loud before model allocation unless this environment supports blockwise FP8.

    The primary authoritative gate is Transformer Engine's own capability probe
    (``check_fp8_block_scaling_support`` inside ``_probe_hopper_environment``).
    There is deliberately no hard pin on an exact device class, TE version, or
    CUDA/cuBLAS threshold for the general blockwise path: blockwise FP8 must run
    on the same canonical image that runs BF16, and version pins would recreate a
    special-environment requirement and could reject FP8 on a newer or different
    accelerator where BF16 is fine. When the probe rejects the environment we
    fail loud with its concrete reason plus the recorded toolchain for diagnosis.

    When ``implementation`` selects MoE experts (blockwise ``GroupedLinear``),
    one additional TE-owned requirement is gated: the grouped GEMM needs cuBLAS
    >= ``CUBLAS_GROUPED_GEMM_MIN_VERSION``, which the block-scaling probe does not
    cover. This is not a reintroduced device/CUDA pin; it is TE's own grouped
    hard requirement, so a MoE FP8 run on an under-versioned cuBLAS fails loud
    here rather than crashing inside the grouped GEMM.
    """

    _validate_recipe_environment()
    environment = _probe_hopper_environment()
    if not environment.block_scaling_supported:
        reason = (
            environment.unsupported_reason
            or "Transformer Engine reported no blockwise FP8 support"
        )
        raise RuntimeError(
            "Blockwise FP8 is unavailable in this environment: "
            f"{reason} (device compute capability "
            f"{environment.compute_capability[0]}.{environment.compute_capability[1]}, "
            f"Transformer Engine {environment.transformer_engine_version}, "
            f"CUDA {environment.cuda_version[0]}.{environment.cuda_version[1]}, "
            f"cuBLASLt {environment.cublas_version})."
        )
    if (
        implementation is not None
        and SemanticSite.MOE_EXPERT in implementation.fp8_sites
        and environment.cublas_version < CUBLAS_GROUPED_GEMM_MIN_VERSION
    ):
        raise RuntimeError(
            "Blockwise FP8 MoE experts require Transformer Engine's grouped GEMM, "
            f"which needs cuBLAS >= {CUBLAS_GROUPED_GEMM_MIN_VERSION} "
            "(CUBLAS_GROUPED_GEMM_VERSION guard); this environment reports cuBLASLt "
            f"{environment.cublas_version}. The block-scaling probe can pass for "
            "blockwise Linear/LayerNormLinear while GroupedLinear crashes, so the "
            f"MoE FP8 profile {implementation.name!r} is gated here rather than left "
            "to fail inside the grouped GEMM."
        )
    _warn_on_unvalidated_toolchain(environment)
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

    if implementation.parameter_contract.compute_weight is WeightStorage.BF16:
        return _precision_context(PrecisionPhase.MODEL_INIT, implementation)
    return _quantized_precision_model_init_context(implementation)


@contextmanager
def _quantized_precision_model_init_context(
    implementation: PrecisionImplementation,
) -> Iterator[PrecisionImplementation]:
    """Construct TE FP8 parameters while retaining their FP32 source values.

    TE owns quantization and FSDP hooks for its Float8Tensor.  MLite only owns
    the subsequently-created FP32 optimizer master, so this deliberately does
    not request TE optimizer masters.
    """

    from transformer_engine.pytorch.quantization import quantized_model_init

    with quantized_model_init(
        enabled=True,
        recipe=implementation.recipe_factory(),
        preserve_high_precision_init_val=True,
    ):
        with _precision_context(PrecisionPhase.MODEL_INIT, implementation) as active:
            yield active


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

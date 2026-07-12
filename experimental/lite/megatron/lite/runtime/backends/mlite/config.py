# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron Lite backend configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from megatron.lite.primitive.cuda_graph import CudaGraphDebugMode
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig, pick_fields


@dataclass(slots=True)
class DebugConfig:
    """Megatron Lite backend debug flags. Not exposed to end users."""

    param_update: bool = False
    optimizer_state: bool = False
    grad_phases: bool = False
    router_summary: bool = False
    moe_io: bool = False
    attn_io: bool = False


@dataclass
class MegatronLiteConfig:
    """Config for MegatronLiteRuntime (Megatron Lite's default 5D parallel runtime).

    Megatron Lite-specific training features live in ``impl_cfg`` (a plain dict —
    each impl reads the keys it needs via its own typed ImplConfig).
    """

    # ── identity ──
    model_name: str = "auto"
    impl: str = "lite"
    hf_path: str = ""

    # ── parallelism and optimizer ──
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # ── common runtime/model fields ──
    attention_backend_override: str | None = "flash"
    router_aux_loss_coef: float | None = None
    load_hf_weights: bool = True

    # ── CUDA Graph diagnostic override (common runtime boundary) ──
    # The absence of an override means "apply the strongest CUDA Graph coverage
    # qualified for this exact model/runtime plan". ``OFF`` is a diagnostic escape
    # hatch only (eager correctness oracle / A-B baseline / debugging); it is NOT
    # a peer user-selectable feature and does not select coverage. Capture
    # granularity, backend, targets, pools, and optimizer-graph choice are
    # resolved by the runtime's explicit assembly of the controller, never by a
    # per-model field. ``CudaGraphProfile``/``CudaGraphTarget`` selection surfaces
    # are hard-walled: they must never appear in ``impl_cfg`` or any model
    # ``ImplConfig`` (see ``docs/cuda-graph-design.md`` §API Alternatives).
    cuda_graph_debug: CudaGraphDebugMode = CudaGraphDebugMode.AUTO

    # ── impl-specific (each impl reads its own keys) ──
    impl_cfg: dict[str, Any] = field(default_factory=dict)

    # ── debug ──
    debug: DebugConfig = field(default_factory=DebugConfig)

    # ── bench-only hook: mutate model_cfg after build (e.g. expert truncation) ──
    model_config_hook: Any = None

    @classmethod
    def from_dict(cls, hf_path: str, cfg: dict[str, Any]) -> MegatronLiteConfig:
        """Construct MegatronLiteConfig from a flat dict (legacy / OmegaConf path)."""
        if "num_microbatches" in cfg:
            raise ValueError(
                "MegatronLiteConfig no longer accepts `num_microbatches`; "
                "pass it to Runtime.forward_backward(..., num_microbatches=...) instead"
            )
        parallel = ParallelConfig(**pick_fields(ParallelConfig, cfg))

        opt_d = cfg.get("optimizer", {})
        optimizer = (
            OptimizerConfig(**pick_fields(OptimizerConfig, opt_d))
            if isinstance(opt_d, dict)
            else OptimizerConfig()
        )
        if isinstance(opt_d, dict) and isinstance(opt_d.get("override_optimizer_config"), dict):
            optimizer.override_optimizer_config = dict(opt_d["override_optimizer_config"])

        # impl_cfg: merge nested dict + top-level overrides
        impl_cfg: dict[str, Any] = {}
        nested = cfg.get("impl_cfg")
        if isinstance(nested, dict):
            impl_cfg.update(nested)
        for k in list(impl_cfg):
            if k in cfg:
                impl_cfg[k] = cfg[k]
        for k in ("recompute", "use_thd", "use_deepep", "precision_aware_opt"):
            if k in cfg and k not in impl_cfg:
                impl_cfg[k] = cfg[k]

        # Hard-wall the CG selection surface: profile/target/granularity must not
        # leak in via impl_cfg. Only the diagnostic override is honored, coerced
        # from a string ("auto"/"off") if given.
        for banned in ("cuda_graph_profile", "cuda_graph_target", "cuda_graph"):
            if banned in impl_cfg:
                raise ValueError(
                    f"impl_cfg must not carry CUDA Graph selection surface {banned!r}: "
                    "capture profile/target/granularity are resolved by the runtime's "
                    "explicit assembly, not a model field. Use `cuda_graph_debug` for "
                    "the diagnostic OFF override only."
                )
        cg_debug = cfg.get("cuda_graph_debug")
        if isinstance(cg_debug, str):
            cg_debug = CudaGraphDebugMode(cg_debug.lower())

        skip = {"parallel", "optimizer", "impl_cfg", "debug", "cuda_graph_debug"}
        return cls(
            **{k: v for k, v in pick_fields(cls, cfg).items() if k not in skip},
            hf_path=hf_path,
            parallel=parallel,
            optimizer=optimizer,
            impl_cfg=impl_cfg,
            **({"cuda_graph_debug": cg_debug} if cg_debug is not None else {}),
        )


__all__ = ["MegatronLiteConfig", "DebugConfig"]

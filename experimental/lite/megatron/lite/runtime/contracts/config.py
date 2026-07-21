# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared runtime configuration for Megatron Lite."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any


def pick_fields(cls, src: dict[str, Any]) -> dict[str, Any]:
    """Extract fields of dataclass *cls* that exist in *src*."""
    return {f.name: src[f.name] for f in dc_fields(cls) if f.name in src}


if TYPE_CHECKING:
    from megatron.lite.runtime.backends.bridge.config import BridgeConfig
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig


@dataclass
class ParallelConfig:
    """Parallel dimensions used by Megatron Lite."""

    tp: int = 1
    etp: int | None = None
    ep: int = 1
    pp: int = 1
    vpp: int = 1
    cp: int = 1
    # Optional explicit mcore pipeline layout for advanced users (custom mode),
    # e.g. "E|t*5|t*6|t,m,L" (MTP `m` must sit on the final/loss stage; standalone MTP
    # is not supported yet). None -> auto-balanced from (num_layers, pp, mtp).
    pp_layout: str | list | None = None


@dataclass
class OptimizerConfig:
    """Optimizer + LR scheduler config. Aligned with VERL McoreOptimizerConfig.

    Stable VERL fields use VERL default values.
    Compatibility aliases are lowered into backend-specific override dicts
    by the adapter layer before consumption.
    """

    # --- stable VERL fields ---
    # Name the algorithm explicitly: ``optimizer`` is the enclosing config
    # object throughout the runtime and ``optimizer.optimizer`` was ambiguous.
    optimizer_algorithm: str = "adam"
    lr: float = 1e-3
    min_lr: float = 0.0
    clip_grad: float = 1.0
    weight_decay: float = 0.01
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    lr_warmup_steps: int = -1
    lr_warmup_init: float = 0.0
    lr_decay_steps: int | None = None
    lr_decay_style: str = "linear"
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: int | None = None
    use_checkpoint_opt_param_scheduler: bool = False

    # --- Megatron native Muon fields (Megatron-LM d64ba4ccb) ---
    muon_momentum: float = 0.95
    muon_split_qkv: bool = True
    muon_nesterov: bool = False
    muon_scale_mode: str = "spectral"
    muon_fp32_matmul_prec: str = "medium"
    muon_coefficient_type: str = "quintic"
    muon_num_ns_steps: int = 5
    muon_tp_mode: str = "blockwise"
    muon_extra_scale_factor: float = 1.0
    muon_scalar_optimizer: str = "adam"

    # --- Megatron native LayerWise/DDP fields ---
    use_layer_wise_param_layout: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    overlap_param_gather_with_optimizer_step: bool = False

    # --- Megatron native optimizer-offload fields ---
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 0.0
    use_torch_optimizer_for_cpu_offload: bool = False
    overlap_cpu_optimizer_d2h_h2d: bool = False
    pin_cpu_grads: bool = True
    pin_cpu_params: bool = True
    offload_optimizer_states: bool = False

    # --- compatibility aliases ---
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    adam_eps: float | None = None
    offload_fraction: float | None = None
    use_precision_aware_optimizer: bool | None = None
    decoupled_weight_decay: bool | None = None

    def __post_init__(self) -> None:
        """Normalize legacy aliases and reject ambiguous native contracts."""

        if self.offload_fraction is not None:
            if self.optimizer_offload_fraction not in (0.0, self.offload_fraction):
                raise ValueError(
                    "offload_fraction compatibility alias conflicts with "
                    "optimizer_offload_fraction"
                )
            self.optimizer_offload_fraction = float(self.offload_fraction)
        if not 0.0 <= self.optimizer_offload_fraction <= 1.0:
            raise ValueError("optimizer_offload_fraction must be in [0, 1]")
        if self.muon_scalar_optimizer != "adam":
            raise ValueError("muon_scalar_optimizer currently supports only 'adam'")
        if (
            self.optimizer_algorithm.lower() == "muon"
            and self.overlap_param_gather_with_optimizer_step
        ):
            raise ValueError(
                "overlap_param_gather_with_optimizer_step is not supported with Muon"
            )


@dataclass
class RuntimeConfig:
    """Top-level runtime configuration.

    Attributes:
        backend: Runtime backend name. Use ``"mlite"`` for Megatron Lite or
            ``"bridge"`` for Megatron-Bridge.
        hf_path: Path to HuggingFace model directory. Required for real runs.
        backend_cfg: Backend config or a compatible dict.
    """

    backend: str = "mlite"
    hf_path: str = ""
    backend_cfg: MegatronLiteConfig | BridgeConfig | dict[str, Any] = field(default_factory=dict)


__all__ = ["OptimizerConfig", "ParallelConfig", "RuntimeConfig"]

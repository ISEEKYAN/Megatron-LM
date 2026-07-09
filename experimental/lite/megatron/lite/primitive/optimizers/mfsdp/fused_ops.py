# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional fused optimizer construction with a Torch reference fallback."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def build_optimizer(
    param_groups: list[dict[str, Any]],
    opt: Any,
) -> torch.optim.Optimizer:
    optimizer_name = str(getattr(opt, "optimizer", "adam"))
    lr = float(getattr(opt, "lr", 1.0e-4))
    weight_decay = float(getattr(opt, "weight_decay", 0.01))
    values = dict(getattr(opt, "override_optimizer_config", None) or {})
    use_fused = bool(values.get("use_fused_optimizer", True)) and _has_cuda_params(
        param_groups
    )
    if use_fused:
        fused = _build_apex_optimizer(
            optimizer_name,
            param_groups,
            opt,
            lr=lr,
            weight_decay=weight_decay,
        )
        if fused is not None:
            return fused

    if optimizer_name == "adam":
        beta1 = getattr(opt, "adam_beta1", None)
        beta2 = getattr(opt, "adam_beta2", None)
        eps = getattr(opt, "adam_eps", None)
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9 if beta1 is None else beta1, 0.999 if beta2 is None else beta2),
            eps=1.0e-8 if eps is None else eps,
            foreach=False,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(getattr(opt, "sgd_momentum", 0.9)),
        )
    raise ValueError(f"Unsupported M-FSDP optimizer: {optimizer_name!r}.")


def _build_apex_optimizer(
    optimizer_name: str,
    param_groups: list[dict[str, Any]],
    opt: Any,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer | None:
    try:
        apex_optimizers = importlib.import_module("apex.optimizers")
        if optimizer_name == "adam":
            beta1 = getattr(opt, "adam_beta1", None)
            beta2 = getattr(opt, "adam_beta2", None)
            eps = getattr(opt, "adam_eps", None)
            return apex_optimizers.FusedAdam(
                param_groups,
                lr=lr,
                betas=(
                    0.9 if beta1 is None else beta1,
                    0.999 if beta2 is None else beta2,
                ),
                eps=1.0e-8 if eps is None else eps,
                weight_decay=weight_decay,
                adam_w_mode=True,
            )
        if optimizer_name == "sgd":
            return apex_optimizers.FusedSGD(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                momentum=float(getattr(opt, "sgd_momentum", 0.9)),
            )
    except (ImportError, AttributeError, RuntimeError, TypeError) as error:
        logger.warning(
            "Fused optimizer unavailable (%s); using Torch optimizer.", error
        )
    return None


def _has_cuda_params(param_groups: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(param, torch.Tensor) and param.device.type == "cuda"
        for group in param_groups
        for param in group["params"]
    )


__all__ = ["build_optimizer"]

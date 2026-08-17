# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional fused optimizer construction with a Torch reference fallback."""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)

OptimizerFactory = Callable[[list[dict[str, Any]], Any], torch.optim.Optimizer]


def build_optimizer(
    param_groups: list[dict[str, Any]],
    opt: Any,
    *,
    optimizer_factory: OptimizerFactory | None = None,
    use_decoupled_grad: bool = False,
) -> torch.optim.Optimizer:
    """Build the optimizer algorithm independently from M-FSDP sharding.

    ``optimizer_factory`` receives already-sharded main parameter groups (FP32
    by default) plus the original optimizer config. Optional algorithms such as
    Muon can regroup them without changing the M-FSDP communication path.
    """
    if optimizer_factory is not None:
        return optimizer_factory(param_groups, opt)

    optimizer_name = str(getattr(opt, "optimizer", "adam"))
    lr = float(getattr(opt, "lr", 1.0e-4))
    weight_decay = float(getattr(opt, "weight_decay", 0.01))
    values = dict(getattr(opt, "override_optimizer_config", None) or {})
    use_fused = bool(values.get("use_fused_optimizer", True)) and _has_cuda_params(
        param_groups
    )
    if use_fused:
        fused = _build_fused_optimizer(
            optimizer_name,
            param_groups,
            opt,
            lr=lr,
            weight_decay=weight_decay,
            use_decoupled_grad=use_decoupled_grad,
        )
        if fused is not None:
            return fused
    if use_decoupled_grad:
        raise RuntimeError(
            "M-FSDP BF16 main gradients require TransformerEngine FusedAdam "
            "with use_decoupled_grad support."
        )

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


def _build_fused_optimizer(
    optimizer_name: str,
    param_groups: list[dict[str, Any]],
    opt: Any,
    *,
    lr: float,
    weight_decay: float,
    use_decoupled_grad: bool = False,
) -> torch.optim.Optimizer | None:
    errors = []
    for module_name in (
        "transformer_engine.pytorch.optimizers",
        "apex.optimizers",
    ):
        if use_decoupled_grad and module_name != "transformer_engine.pytorch.optimizers":
            continue
        try:
            optimizers = importlib.import_module(module_name)
            if optimizer_name == "adam":
                beta1 = getattr(opt, "adam_beta1", None)
                beta2 = getattr(opt, "adam_beta2", None)
                eps = getattr(opt, "adam_eps", None)
                kwargs = {
                    "lr": lr,
                    "betas": (
                        0.9 if beta1 is None else beta1,
                        0.999 if beta2 is None else beta2,
                    ),
                    "eps": 1.0e-8 if eps is None else eps,
                    "weight_decay": weight_decay,
                    "adam_w_mode": True,
                }
                if use_decoupled_grad:
                    signature = inspect.signature(optimizers.FusedAdam).parameters
                    if (
                        "use_decoupled_grad" not in signature
                        and not any(
                            parameter.kind is inspect.Parameter.VAR_KEYWORD
                            for parameter in signature.values()
                        )
                    ):
                        raise RuntimeError(
                            "TransformerEngine FusedAdam does not expose "
                            "use_decoupled_grad."
                        )
                    kwargs.update(
                        {
                            "use_decoupled_grad": True,
                            # M-FSDP already owns the FP32 main parameter shard.
                            "master_weights": False,
                        }
                    )
                return optimizers.FusedAdam(param_groups, **kwargs)
            if optimizer_name == "sgd":
                return optimizers.FusedSGD(
                    param_groups,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=float(getattr(opt, "sgd_momentum", 0.9)),
                )
        except (ImportError, AttributeError, RuntimeError, TypeError) as error:
            errors.append(f"{module_name}: {error}")
    logger.warning(
        "Fused optimizer unavailable (%s); using Torch optimizer.", "; ".join(errors)
    )
    return None


def _has_cuda_params(param_groups: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(param, torch.Tensor) and param.device.type == "cuda"
        for group in param_groups
        for param in group["params"]
    )


__all__ = ["OptimizerFactory", "build_optimizer"]

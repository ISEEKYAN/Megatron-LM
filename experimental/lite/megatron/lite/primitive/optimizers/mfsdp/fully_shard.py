# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Narrow ``optim_grads_params`` construction API for M-FSDP."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch.nn as nn
from megatron.lite.primitive.optimizers.mfsdp.config import (
    MFSDPConfig,
    MFSDPProcessGroups,
)
from megatron.lite.primitive.optimizers.mfsdp.wrapper import MegatronFSDP


def fully_shard_model(
    module: nn.Module,
    *,
    groups: MFSDPProcessGroups,
    config: MFSDPConfig,
    is_expert: Callable[[str], bool],
    unit_modules: Iterable[type[nn.Module] | str] | None,
    enable_fine_grained_param_gather_hook: bool = False,
    enable_fine_grained_param_gather_backward_hook: bool = False,
    fine_grained_recurse_module_types: Iterable[type[nn.Module]] | None = None,
) -> MegatronFSDP:
    if config.sharding_strategy != "optim_grads_params":
        raise ValueError(
            "MLite M-FSDP only implements the optim_grads_params strategy."
        )
    return MegatronFSDP(
        module,
        groups=groups,
        config=config,
        is_expert=is_expert,
        unit_modules=unit_modules,
        enable_fine_grained_param_gather_hook=(
            enable_fine_grained_param_gather_hook
        ),
        enable_fine_grained_param_gather_backward_hook=(
            enable_fine_grained_param_gather_backward_hook
        ),
        fine_grained_recurse_module_types=fine_grained_recurse_module_types,
    )


__all__ = ["fully_shard_model"]

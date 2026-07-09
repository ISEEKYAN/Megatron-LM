# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Topology and parameter metadata normalization for M-FSDP."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from megatron.lite.primitive.optimizers.mfsdp.patches import (
    clear_skip_tp_duplicate_sync,
    mark_skip_tp_duplicate_sync,
    set_mfsdp_param_name,
)


def annotate_parallel_parameters(
    module: nn.Module,
    is_expert: Callable[[str], bool],
    *,
    tp_size: int,
    etp_size: int,
) -> None:
    """Normalize ownership from topology and explicit parameter attributes.

    The primitive deliberately does not infer ownership from model or layer
    names.  Composition may set ``partition_dim``/``tensor_model_parallel``;
    otherwise matrix parameters use their active dense/expert TP domain.
    """
    sequence_parallel_ids = {id(param) for param in getattr(module, "sp_params", ())}
    for name, param in module.named_parameters():
        set_mfsdp_param_name(param, name)
        expert = bool(is_expert(name))
        param._mfsdp_is_expert = expert
        if not hasattr(param, "allreduce"):
            param.allreduce = not expert

        replicated = bool(
            id(param) in sequence_parallel_ids
            or getattr(param, "sequence_parallel", False)
            or getattr(param, "average_gradients_across_tp_domain", False)
            or getattr(param, "shared", False)
        )
        if id(param) in sequence_parallel_ids:
            param.sequence_parallel = True
        active_tp_size = max(int(etp_size if expert else tp_size), 1)
        explicitly_sharded = bool(getattr(param, "tensor_model_parallel", False))
        sharded = (
            not replicated
            and param.ndim > 1
            and (explicitly_sharded or active_tp_size > 1)
        )
        param.tensor_model_parallel = sharded
        if sharded:
            mark_skip_tp_duplicate_sync(param)
        else:
            clear_skip_tp_duplicate_sync(param)


__all__ = ["annotate_parallel_parameters"]

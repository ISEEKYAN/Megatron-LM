# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Self-contained Megatron-FSDP optimizer construction for Megatron Lite.

The model wrapper and communication pipelines are the vendored M-FSDP
implementation in :mod:`.impl`.  The only code shared with the FSDP2 backend
is its backend-neutral AdamW/gradient adapter; this module never enters the
FSDP2 ``fully_shard`` training path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2.optimizer import build_fsdp2_adamw
from megatron.lite.primitive.optimizers.mfsdp.checkpoint_keys import (
    attach_mfsdp_checkpoint_metadata,
)
from megatron.lite.primitive.optimizers.mfsdp.config import validate_mfsdp_config
from megatron.lite.primitive.optimizers.mfsdp.impl.fully_shard import (
    fully_shard_model,
    fully_shard_optimizer,
)
from megatron.lite.primitive.optimizers.mfsdp.metadata import ensure_mfsdp_tp_partition_attrs
from megatron.lite.primitive.protocols import ExpertClassifierFn, default_expert_classifier


def _override(opt: Any, name: str, default: Any) -> Any:
    values = dict(getattr(opt, "override_optimizer_config", None) or {})
    return values.get(name, getattr(opt, name, default))


def build_mfsdp_stack(
    model_chunks: list[nn.Module],
    *,
    model_cfg,
    engine_cfg,
    ps,
    is_expert: ExpertClassifierFn | None = None,
    proto=None,
    fsdp_unit_modules: tuple[type[nn.Module] | str, ...] | None = None,
    skip_fsdp_wrap: bool = False,
):
    """Wrap chunks with the local M-FSDP implementation and build AdamW."""
    del model_cfg, proto
    validate_mfsdp_config(engine_cfg)
    opt = engine_cfg.optimizer
    classifier = is_expert or default_expert_classifier

    if skip_fsdp_wrap:
        wrapped_chunks = list(model_chunks)
    else:
        wrapped_chunks = [
            fully_shard_model(
                chunk,
                fsdp_unit_modules=fsdp_unit_modules,
                zero_dp_strategy=_override(
                    opt, "mfsdp_sharding_strategy", "optim_grads_params"
                ),
                grad_reduce_in_fp32=bool(_override(opt, "grad_reduce_in_fp32", False)),
                preserve_fp32_weights=bool(_override(opt, "preserve_fp32_weights", True)),
                overlap_grad_reduce=bool(_override(opt, "overlap_grad_reduce", True)),
                overlap_param_gather=bool(_override(opt, "overlap_param_gather", True)),
                check_for_nan_in_grad=bool(_override(opt, "check_for_nan_in_grad", True)),
                average_in_collective=bool(_override(opt, "average_in_collective", False)),
                disable_bucketing=index > 0,
            )
            for index, chunk in enumerate(model_chunks)
        ]

    expert_params = []
    dense_params = []
    for chunk in wrapped_chunks:
        for name, param in chunk.named_parameters():
            (expert_params if classifier(name) else dense_params).append(param)

    optimizer = build_fsdp2_adamw(
        wrapped_chunks,
        opt,
        ps,
        expert_sharded_grad_params=expert_params,
        expert_sharded_grad_scale=(
            float(ps.expert_dp_size) / float(ps.dp_cp_size)
            if expert_params and ps.dp_cp_size
            else 1.0
        ),
        expert_sharded_grad_norm_group=ps.ep_dp_group if expert_params else None,
        grad_norm_accum_dtype=_override(opt, "grad_norm_accum_dtype", "float32"),
        adamw_foreach=False,
        # M-FSDP already exposes its distributed FP32 optimization weights;
        # creating a second FSDP2-style master copy would detach the optimizer
        # from the wrapper's parameter-install lifecycle.
        use_fp32_master=False,
    )
    if not skip_fsdp_wrap:
        fully_shard_optimizer(optimizer.optimizer)

    optimizer._mc_pg_collection = None
    optimizer.model_chunks = wrapped_chunks
    attach_mfsdp_checkpoint_metadata(optimizer, ps=ps, is_expert=classifier)
    return wrapped_chunks, optimizer


def build_mfsdp_training_optimizer(
    model_chunks: list[nn.Module],
    *,
    model_cfg,
    impl_cfg,
    ps,
    model_name: str,
    is_expert: ExpertClassifierFn | None = None,
    fsdp_unit_modules: tuple[type[nn.Module] | str, ...] | None = None,
    deterministic: bool | None = None,
):
    """Build the local M-FSDP stack from an ``ImplConfig``."""
    del deterministic
    opt = impl_cfg.optimizer_config
    if opt is None:
        opt = SimpleNamespace(
            optimizer="adam",
            lr=1e-4,
            min_lr=0.0,
            weight_decay=0.01,
            clip_grad=1.0,
            offload_fraction=None,
            adam_beta1=None,
            adam_beta2=None,
            adam_eps=None,
        )
    engine_cfg = SimpleNamespace(
        model_name=model_name,
        parallel=impl_cfg.parallel,
        optimizer=opt,
    )
    model_chunks[:], optimizer = build_mfsdp_stack(
        model_chunks,
        model_cfg=model_cfg,
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=is_expert,
        fsdp_unit_modules=fsdp_unit_modules,
    )

    def finalize_grads() -> None:
        finalize_mfsdp_grads(model_chunks, optimizer)

    return optimizer, finalize_grads


def finalize_mfsdp_grads(model_chunks: list[nn.Module], optimizer) -> None:
    """Finish M-FSDP reduce-scatter work before the optimizer update."""
    del optimizer
    for chunk in model_chunks:
        finish_grad_sync = getattr(chunk, "finish_grad_sync", None)
        if callable(finish_grad_sync):
            finish_grad_sync()


__all__ = [
    "build_mfsdp_stack",
    "build_mfsdp_training_optimizer",
    "finalize_mfsdp_grads",
]

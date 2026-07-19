# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Absorbing model-composition kernel.

Every native lite model repeated the same ``build_model`` body: build the (VPP)
chunks, wrap recompute / offload, wire the ``dist_opt`` or ``fsdp2`` optimizer,
attach the sharded state dict, and pack a :class:`ModelBundle`.  Only a handful
of per-model choices differed.

``assemble`` owns that shared body once.  A model declares its differences as a
:class:`ModelSpec` and calls ``assemble(spec, model_cfg, impl_cfg)``; it no
longer re-implements chunk construction, optimizer wiring, or bundle assembly.

The single point where a model tells the kernel how to walk its transformer
units is :attr:`ModelSpec.transformer_units` (issue #114): recompute, offload,
and any future per-unit pass go through that one enumeration, never through an
ad-hoc ``chunk.layers`` reach-in scattered across protocols.

This module lives in the ``model`` layer (not ``primitive``): the dist_opt path
registers training hooks via ``runtime.megatron_utils``, which the primitive
layer is forbidden to import but the model layer may use.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.recompute import (
    ModuleMap,
    apply_offload,
    apply_recompute,
    parse_recompute_spec,
)

# A model's build knobs live on its own ``ImplConfig`` dataclass; the kernel only
# reads the fields shared by every model, so it takes ``impl_cfg`` structurally.
ImplConfigLike = Any
ModelConfigLike = Any


@dataclass(frozen=True)
class ModelSpec:
    """Typed declaration of a model's composition deltas.

    The kernel drives the shared assembly; the model supplies only what is
    genuinely model-specific through these fields.  Every callable receives the
    already-built ``chunks`` / ``ps`` / ``impl_cfg`` so a model never re-derives
    parallel state or re-reads ``impl_cfg`` outside its own hooks.
    """

    #: Model name used for optimizer bucketing / diagnostics.
    name: str

    #: Build one chunk. ``vpp_chunk_id`` is ``None`` for a single (non-VPP)
    #: chunk, else the chunk index. The kernel handles ``.to(bf16).cuda()``.
    chunk_factory: Callable[[ModelConfigLike, ImplConfigLike, ParallelState, int | None], nn.Module]

    #: Enumerate the transformer units of a chunk that recompute / offload walk.
    #: This is the single #114 unit enumeration — no protocol reaches into
    #: ``chunk.layers`` directly.
    transformer_units: Callable[[nn.Module], Iterable[nn.Module]]

    #: ``{spec_name: accessor}`` for recompute / offload sub-module targeting.
    module_map: ModuleMap

    #: ``forward_step(model, batch) -> dict`` stored on the bundle.
    forward_step: Callable[..., dict]

    #: Classify a parameter name as an expert (MoE) param.
    expert_classifier: Callable[[str], bool]

    #: Placement function for the sharded state dict (dist_opt path).
    placement_fn: Callable[..., Any]

    #: ``fsdp2`` unit module classes wrapped as FSDP units.
    fsdp2_unit_modules: Callable[[], tuple[type[nn.Module], ...]]

    # --- optional per-model hooks (default no-op) --------------------------

    #: Mutate ``model_cfg`` / validate ``impl_cfg`` before any chunk is built
    #: (parallel-scope gates, MTP config, aux-loss coefs, ...).
    prepare: Callable[[ModelConfigLike, ImplConfigLike, ParallelState], None] | None = None

    #: Run after all chunks are built (e.g. cross-entropy fusion toggles,
    #: attention-backend configuration). Receives ``(chunks, impl_cfg)``.
    post_chunk_hook: Callable[[list[nn.Module], ImplConfigLike], None] | None = None

    #: ``pre_forward_hook(scale)`` stored on the bundle (aux-loss auto-scaler).
    pre_forward_hook_factory: Callable[[], Callable[[torch.Tensor], None]] | None = None

    #: Extra kwargs forwarded to the fsdp2 optimizer builder (default: none).
    fsdp2_extra_kwargs: dict[str, Any] = field(default_factory=dict)

    #: Normalize ``impl_cfg.optimizer`` to a backend name. Defaults to identity
    #: (models pass ``"dist_opt"`` / ``"fsdp2"`` / ``None`` directly). DS4 also
    #: accepts an ``OptimizerConfig`` / dict and maps it to ``"dist_opt"``.
    optimizer_backend_name: Callable[[Any], str | None] | None = None


def _build_chunks(
    spec: ModelSpec, model_cfg: ModelConfigLike, impl_cfg: ImplConfigLike, ps: ParallelState
) -> list[nn.Module]:
    p = impl_cfg.parallel
    vpp = None if p.vpp == 1 else p.vpp
    ids: list[int | None] = list(range(vpp)) if vpp is not None else [None]
    return [
        spec.chunk_factory(model_cfg, impl_cfg, ps, i).to(torch.bfloat16).cuda() for i in ids
    ]


def _apply_recompute_offload(
    spec: ModelSpec, chunks: list[nn.Module], impl_cfg: ImplConfigLike
) -> None:
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(spec.transformer_units(chunk), recompute_spec, spec.module_map)
    if impl_cfg.offload:
        for chunk in chunks:
            apply_offload(spec.transformer_units(chunk), impl_cfg.offload, spec.module_map)


def _wire_optimizer(
    spec: ModelSpec,
    chunks: list[nn.Module],
    model_cfg: ModelConfigLike,
    impl_cfg: ImplConfigLike,
    ps: ParallelState,
) -> tuple[Any | None, Callable[[], None] | None, Callable[[], dict] | None, str]:
    """Return ``(optimizer, finalize_grads, post_model_load_hook, backend)``."""
    optimizer = getattr(impl_cfg, "optimizer", None)
    if spec.optimizer_backend_name is not None:
        optimizer = spec.optimizer_backend_name(optimizer)
    if optimizer == "dist_opt":
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_training_optimizer,
        )
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        opt, finalize_grads = build_dist_opt_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=impl_cfg,
            ps=ps,
            model_name=spec.name,
            is_expert=spec.expert_classifier,
            deterministic=impl_cfg.deterministic,
        )
        attach_model_sharded_state_dict(
            chunks, ps, get_placements=spec.placement_fn, is_expert=spec.expert_classifier
        )
        register_training_hooks(chunks, opt)
        return opt, finalize_grads, None, "dist_opt"

    if optimizer == "fsdp2":

        def _post_model_load_hook() -> dict:
            from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=spec.fsdp2_unit_modules(),
                    expert_classifier=spec.expert_classifier,
                    deterministic=impl_cfg.deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    leaf_module_names=(),
                    **spec.fsdp2_extra_kwargs,
                )
            }

        return None, None, _post_model_load_hook, "fsdp2"

    if optimizer is None:
        return None, None, None, "none"

    raise ValueError(f"Unknown {spec.name} lite optimizer: {optimizer!r}.")


def assemble(
    spec: ModelSpec, model_cfg: ModelConfigLike, impl_cfg: ImplConfigLike
) -> ModelBundle:
    """Compose a model from its :class:`ModelSpec` and configs.

    Shared body: prepare -> parallel state -> chunks -> post-chunk hook ->
    recompute/offload -> optimizer wiring -> bundle.
    """
    ps = init_parallel(impl_cfg.parallel)

    if spec.prepare is not None:
        spec.prepare(model_cfg, impl_cfg, ps)

    chunks = _build_chunks(spec, model_cfg, impl_cfg, ps)

    if spec.post_chunk_hook is not None:
        spec.post_chunk_hook(chunks, impl_cfg)

    _apply_recompute_offload(spec, chunks, impl_cfg)

    optimizer, finalize_grads, post_model_load_hook, optimizer_backend = _wire_optimizer(
        spec, chunks, model_cfg, impl_cfg, ps
    )

    extras: dict[str, Any] = {
        "model_cfg": model_cfg,
        "optimizer_backend": optimizer_backend,
        "post_model_load_hook": post_model_load_hook,
    }
    if spec.pre_forward_hook_factory is not None:
        extras["pre_forward_hook"] = spec.pre_forward_hook_factory()

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=spec.forward_step,
        extras=extras,
    )


__all__ = ["ModelSpec", "assemble"]

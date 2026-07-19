# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Absorbing model-composition kernel: the shared ``build_model`` body.

``assemble`` owns the body every native lite model repeated (VPP chunk build,
recompute/offload, dist_opt/fsdp2 wiring, sharded-state-dict attach, ModelBundle
packing). A model declares its per-model deltas as inline callables passed
straight to ``assemble`` -- there is no per-model spec object to construct.
``transformer_units`` is the single #114 unit enumeration that recompute and
offload both walk. Lives in the ``model`` layer because dist_opt registers hooks
via ``runtime.megatron_utils`` (barred from ``primitive``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

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

# ``impl_cfg`` is a per-model ``ImplConfig``; the kernel reads only shared fields.
ImplConfigLike = Any
ModelConfigLike = Any


@runtime_checkable
class ModelSpec(Protocol):
    """Structural type of the delta callables a model passes to ``assemble``.

    Used only for conformance typing/checking -- no model instantiates it.
    """

    name: str
    chunk_factory: Callable[[ModelConfigLike, ImplConfigLike, ParallelState, int | None], nn.Module]
    transformer_units: Callable[[nn.Module], Iterable[nn.Module]]
    module_map: ModuleMap
    forward_step: Callable[..., dict]
    expert_classifier: Callable[[str], bool]
    placement_fn: Callable[..., Any]
    fsdp2_unit_modules: Callable[[], tuple[type[nn.Module], ...]]


def _build_chunks(
    chunk_factory: Callable[..., nn.Module],
    model_cfg: ModelConfigLike,
    impl_cfg: ImplConfigLike,
    ps: ParallelState,
) -> list[nn.Module]:
    p = impl_cfg.parallel
    vpp = None if p.vpp == 1 else p.vpp
    ids: list[int | None] = list(range(vpp)) if vpp is not None else [None]
    return [chunk_factory(model_cfg, impl_cfg, ps, i).to(torch.bfloat16).cuda() for i in ids]


def _apply_recompute_offload(
    chunks: list[nn.Module],
    transformer_units: Callable[[nn.Module], Iterable[nn.Module]],
    module_map: ModuleMap,
    impl_cfg: ImplConfigLike,
) -> None:
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(transformer_units(chunk), recompute_spec, module_map)
    if impl_cfg.offload:
        for chunk in chunks:
            apply_offload(transformer_units(chunk), impl_cfg.offload, module_map)


def _wire_optimizer(
    chunks: list[nn.Module],
    model_cfg: ModelConfigLike,
    impl_cfg: ImplConfigLike,
    ps: ParallelState,
    *,
    name: str,
    expert_classifier: Callable[[str], bool],
    placement_fn: Callable[..., Any],
    fsdp2_unit_modules: Callable[[], tuple[type[nn.Module], ...]],
    fsdp2_extra_kwargs: dict[str, Any],
    fsdp2_deterministic: bool,
    optimizer_backend_name: Callable[[Any], str | None] | None,
    register_hooks: bool,
) -> tuple[Any | None, Callable[[], None] | None, Callable[[], dict] | None, str]:
    """Return ``(optimizer, finalize_grads, post_model_load_hook, backend)``."""
    optimizer = getattr(impl_cfg, "optimizer", None)
    if optimizer_backend_name is not None:
        optimizer = optimizer_backend_name(optimizer)

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
            model_name=name,
            is_expert=expert_classifier,
            deterministic=impl_cfg.deterministic,
        )
        attach_model_sharded_state_dict(
            chunks, ps, get_placements=placement_fn, is_expert=expert_classifier
        )
        if register_hooks:
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
                    unit_modules=fsdp2_unit_modules(),
                    expert_classifier=expert_classifier,
                    deterministic=fsdp2_deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    leaf_module_names=(),
                    **fsdp2_extra_kwargs,
                )
            }

        return None, None, _post_model_load_hook, "fsdp2"

    if optimizer is None:
        return None, None, None, "none"

    raise ValueError(f"Unknown {name} lite optimizer: {optimizer!r}.")


def assemble(
    model_cfg: ModelConfigLike,
    impl_cfg: ImplConfigLike,
    *,
    name: str,
    chunk_factory: Callable[[ModelConfigLike, ImplConfigLike, ParallelState, int | None], nn.Module],
    transformer_units: Callable[[nn.Module], Iterable[nn.Module]],
    module_map: ModuleMap,
    forward_step: Callable[..., dict],
    expert_classifier: Callable[[str], bool],
    placement_fn: Callable[..., Any],
    fsdp2_unit_modules: Callable[[], tuple[type[nn.Module], ...]],
    prepare: Callable[[ModelConfigLike, ImplConfigLike, ParallelState], None] | None = None,
    post_chunk_hook: Callable[[list[nn.Module], ImplConfigLike], None] | None = None,
    pre_forward_hook_factory: Callable[[], Callable[[torch.Tensor], None]] | None = None,
    fsdp2_extra_kwargs: dict[str, Any] | None = None,
    optimizer_backend_name: Callable[[Any], str | None] | None = None,
    extra_extras: Callable[[list[nn.Module], ImplConfigLike], dict[str, Any]] | None = None,
    fsdp2_deterministic_fn: Callable[[ImplConfigLike], bool] | None = None,
    register_hooks: bool = True,
) -> ModelBundle:
    """Compose a model from its delta callables and configs.

    Shared body: prepare -> parallel state -> chunks -> post-chunk hook ->
    recompute/offload -> extra_extras -> optimizer wiring -> bundle. Per-model
    deltas (chunk_factory, transformer_units, hooks, ...) arrive as kwargs;
    ``transformer_units`` is the single #114 enumeration recompute/offload walk.
    """
    ps = init_parallel(impl_cfg.parallel)

    if prepare is not None:
        prepare(model_cfg, impl_cfg, ps)

    chunks = _build_chunks(chunk_factory, model_cfg, impl_cfg, ps)

    if post_chunk_hook is not None:
        post_chunk_hook(chunks, impl_cfg)

    _apply_recompute_offload(chunks, transformer_units, module_map, impl_cfg)

    extras: dict[str, Any] = {} if extra_extras is None else extra_extras(chunks, impl_cfg)

    optimizer, finalize_grads, post_model_load_hook, optimizer_backend = _wire_optimizer(
        chunks,
        model_cfg,
        impl_cfg,
        ps,
        name=name,
        expert_classifier=expert_classifier,
        placement_fn=placement_fn,
        fsdp2_unit_modules=fsdp2_unit_modules,
        fsdp2_extra_kwargs=fsdp2_extra_kwargs or {},
        fsdp2_deterministic=(
            fsdp2_deterministic_fn(impl_cfg)
            if fsdp2_deterministic_fn is not None
            else impl_cfg.deterministic
        ),
        optimizer_backend_name=optimizer_backend_name,
        register_hooks=register_hooks,
    )

    extras.update(
        model_cfg=model_cfg,
        optimizer_backend=optimizer_backend,
        post_model_load_hook=post_model_load_hook,
    )
    if pre_forward_hook_factory is not None:
        extras["pre_forward_hook"] = pre_forward_hook_factory()

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=forward_step,
        extras=extras,
    )


__all__ = ["ModelSpec", "assemble"]

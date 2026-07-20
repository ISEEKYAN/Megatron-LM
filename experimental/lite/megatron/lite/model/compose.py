# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared model-composition toolbox: small utils each ``build_model`` reuses.

This is a *toolbox*, not a framework. There is no single assembly entry, no
per-model spec object, and no absorbing kernel. Each model keeps an explicit,
top-to-bottom ``build_model`` and calls these helpers for the mechanics that were
genuinely copy-pasted across every native lite model:

* ``build_vpp_chunks`` — VPP chunk construction + ``.to(bfloat16).cuda()``.
* ``apply_recompute_offload`` — recompute/offload over the single #114
  ``transformer_units`` walk (recompute and offload always walk the *same* units).
* ``wire_dist_opt`` — dist_opt optimizer + sharded-state attach + grad hooks.
* ``make_fsdp2_post_load_hook`` — the fsdp2 post-model-load optimizer builder.

Lives in the ``model`` layer (not ``primitive``) because ``wire_dist_opt``
registers hooks via ``runtime.megatron_utils``, which ``primitive`` may not import.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import (
    ModuleMap,
    apply_offload,
    apply_recompute,
    parse_recompute_spec,
)

# ``impl_cfg`` is a per-model ``ImplConfig``; helpers read only the shared fields.
ImplConfigLike = Any
ModelConfigLike = Any


def build_vpp_chunks(
    chunk_factory: Callable[[ModelConfigLike, ImplConfigLike, ParallelState, int | None], nn.Module],
    model_cfg: ModelConfigLike,
    impl_cfg: ImplConfigLike,
    ps: ParallelState,
) -> list[nn.Module]:
    """Build one chunk per virtual-pipeline slot, in bf16 on the current CUDA device.

    ``chunk_factory(model_cfg, impl_cfg, ps, vpp_chunk_id)`` builds a single chunk;
    ``vpp_chunk_id`` is ``None`` when VPP is off (``vpp <= 1``).
    """
    vpp = None if impl_cfg.parallel.vpp == 1 else impl_cfg.parallel.vpp
    chunk_ids: list[int | None] = list(range(vpp)) if vpp is not None else [None]
    return [
        chunk_factory(model_cfg, impl_cfg, ps, cid).to(torch.bfloat16).cuda() for cid in chunk_ids
    ]


def apply_recompute_offload(
    chunks: list[nn.Module],
    transformer_units: Callable[[nn.Module], Iterable[nn.Module]],
    module_map: ModuleMap,
    *,
    recompute: str | list[str] | None,
    offload: list[str] | None,
) -> None:
    """Apply recompute then offload over the single #114 ``transformer_units`` walk.

    Both recompute and offload walk the *same* per-chunk unit enumeration, so the
    model passes one ``transformer_units`` callable here (the #114 invariant: one
    unit list, not two divergent walks).
    """
    recompute_spec = parse_recompute_spec(recompute)
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(transformer_units(chunk), recompute_spec, module_map)
    if offload:
        for chunk in chunks:
            apply_offload(transformer_units(chunk), offload, module_map)


def wire_dist_opt(
    chunks: list[nn.Module],
    model_cfg: ModelConfigLike,
    impl_cfg: ImplConfigLike,
    ps: ParallelState,
    *,
    name: str,
    is_expert: Callable[[str], bool],
    placement_fn: Callable[..., Any],
    deterministic: bool,
    register_hooks: bool = True,
) -> tuple[Any, Callable[[], None]]:
    """Build the dist_opt optimizer, attach sharded state, (optionally) hook grads.

    Returns ``(optimizer, finalize_grads)``. ``deterministic`` is passed through
    verbatim — the caller decides the effective value (e.g. Qwen3.5 forces it off
    on the THD GatedDeltaNet path for the chunk/fsdp2 wiring but keeps dist_opt on
    the raw impl flag, preserving pre-refactor behavior).
    """
    from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
    from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_training_optimizer

    optimizer, finalize_grads = build_dist_opt_training_optimizer(
        chunks,
        model_cfg=model_cfg,
        impl_cfg=impl_cfg,
        ps=ps,
        model_name=name,
        is_expert=is_expert,
        deterministic=deterministic,
    )
    attach_model_sharded_state_dict(chunks, ps, get_placements=placement_fn, is_expert=is_expert)
    if register_hooks:
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        register_training_hooks(chunks, optimizer)
    return optimizer, finalize_grads


def make_fsdp2_post_load_hook(
    chunks: list[nn.Module],
    impl_cfg: ImplConfigLike,
    ps: ParallelState,
    *,
    unit_modules: tuple[type[nn.Module], ...],
    expert_classifier: Callable[[str], bool],
    deterministic: bool,
    leaf_module_names: tuple[str, ...] = (),
    **extra_kwargs: Any,
) -> Callable[[], dict]:
    """Return the ``post_model_load_hook`` that builds the fsdp2 optimizer.

    fsdp2 must wrap after weights are loaded, so the model stashes this closure in
    ``ModelBundle.extras['post_model_load_hook']`` and the runtime calls it later.
    """

    def _post_model_load_hook() -> dict:
        from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

        return {
            "optimizer": build_fsdp2_training_optimizer(
                chunks,
                impl_cfg.optimizer_config,
                ps,
                unit_modules=unit_modules,
                expert_classifier=expert_classifier,
                deterministic=deterministic,
                vpp=impl_cfg.parallel.vpp,
                leaf_module_names=leaf_module_names,
                **extra_kwargs,
            )
        }

    return _post_model_load_hook


__all__ = [
    "apply_recompute_offload",
    "build_vpp_chunks",
    "make_fsdp2_post_load_hook",
    "wire_dist_opt",
]

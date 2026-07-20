# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3MoE lite impl — reference model protocol. Copy + adapt for new models.

Runtime calls ``build_model_config(source, **overrides)`` then
``build_model(model_cfg, *, impl_cfg)``. ``build_model`` is explicit and calls the
shared ``compose`` toolbox helpers (build_vpp_chunks / apply_recompute_offload /
wire_dist_opt / make_fsdp2_post_load_hook) for the mechanics common to all models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn
from megatron.lite.model.protocol_utils import (
    add_cross_entropy_fusion,
    add_loss_context_kwargs,
    pack_thd_forward_kwargs,
    set_cross_entropy_fusion,
    unpack_thd_forward_output,
)
from megatron.lite.model.qwen3_moe.common import is_expert_param
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite.checkpoint import EXPERT_CLASSIFIER, PLACEMENT_FN
from megatron.lite.model.qwen3_moe.lite.checkpoint import load_hf_weights as _load_hf_weights_impl
from megatron.lite.model.qwen3_moe.lite.model import MTPLossAutoScaler, Qwen3MoEModel
from megatron.lite.model.compose import (
    apply_recompute_offload,
    build_vpp_chunks,
    make_fsdp2_post_load_hook,
    wire_dist_opt,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    freeze_non_lora_params,
    normalize_lora_config,
    trainable_param_stats,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

__all__ = [
    "EXPERT_CLASSIFIER",
    "ImplConfig",
    "PLACEMENT_FN",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "load_hf_weights",
    "vocab_size",
]

# ---------------------------------------------------------------------------
# ImplConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImplConfig:
    """Lite impl knobs. Constructed by runtime from user config."""

    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_deepep: bool = False
    use_thd: bool = False
    cross_entropy_fusion: bool = False
    router_aux_loss_coef: float | None = None
    router_bias_rate: float = 0.0
    # User-level OptimizerConfig threaded through the runtime.
    optimizer_config: OptimizerConfig | None = None
    mtp_enable: bool = False
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool | None = None
    deterministic: bool = True
    lora: LoraConfig | dict | None = None


# ---------------------------------------------------------------------------
# Module map for recompute/offload
# ---------------------------------------------------------------------------

MODULE_MAP = {
    "core_attn": lambda layer: layer.attn.core_attn,
    "experts": lambda layer: layer.moe.experts,
    "moe": lambda layer: layer.moe,
    "router": lambda layer: layer.moe.router,
    "mlp_norm": lambda layer: layer.mlp_norm,
    "attn_proj": lambda layer: layer.attn.proj,
}


def build_model_config(source: str | Path | dict, **overrides) -> Qwen3MoEConfig:
    """Build Qwen3MoE architecture config from HF source."""
    if isinstance(source, dict):
        cfg = Qwen3MoEConfig._from_hf_dict(source)
    else:
        cfg = Qwen3MoEConfig.from_hf(str(source))
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    kwargs = pack_thd_forward_kwargs(model, batch)
    add_loss_context_kwargs(kwargs, include_return_log_probs=True)
    add_cross_entropy_fusion(kwargs, model)
    return model(**kwargs)


def _forward_step_bshd(model: nn.Module, batch: PackedBatch) -> dict:
    labels = batch.labels.reshape(1, -1) if batch.labels is not None else None
    return model(input_ids=batch.input_ids.reshape(1, -1), labels=labels, packed_seq_params=None)


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output) -> Any:
    return unpack_thd_forward_output(model, batch, output)


def _prepare(model_cfg: Qwen3MoEConfig, impl_cfg: ImplConfig) -> None:
    p = impl_cfg.parallel
    if impl_cfg.use_deepep and (p.etp is not None and p.etp > 1):
        raise ValueError("use_deepep and etp>1 are mutually exclusive")
    if impl_cfg.router_aux_loss_coef is not None:
        model_cfg.router_aux_loss_coef = impl_cfg.router_aux_loss_coef
    if impl_cfg.mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError("mtp_enable=True but HF config has no num_nextn_predict_layers.")
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
        if impl_cfg.mtp_use_repeated_layer is not None:
            model_cfg.mtp_use_repeated_layer = impl_cfg.mtp_use_repeated_layer
    else:
        model_cfg.num_nextn_predict_layers = 0


def _chunk_factory(
    model_cfg: Qwen3MoEConfig, impl_cfg: ImplConfig, ps: ParallelState, vpp_chunk_id: int | None
) -> nn.Module:
    mtp_enable = bool(impl_cfg.mtp_enable)
    vpp = None if impl_cfg.parallel.vpp == 1 else impl_cfg.parallel.vpp
    model_kwargs: dict[str, Any] = dict(
        use_deepep=impl_cfg.use_deepep, fp8=False,
        recompute_modules=parse_recompute_spec(impl_cfg.recompute),
        router_bias_rate=impl_cfg.router_bias_rate,
        use_thd=impl_cfg.use_thd,
        mtp_enable=mtp_enable,
        mtp_enable_train=mtp_enable and bool(impl_cfg.mtp_enable_train),
        mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
        lora_config=normalize_lora_config(impl_cfg.lora),
    )
    if vpp is None:
        return Qwen3MoEModel(model_cfg, ps, **model_kwargs)
    return Qwen3MoEModel(model_cfg, ps, vpp=vpp, vpp_chunk_id=vpp_chunk_id, **model_kwargs)


def _fsdp2_unit_modules() -> tuple[type[nn.Module], ...]:
    from megatron.lite.model.qwen3_moe.lite.model import TransformerLayer

    return (TransformerLayer,)


def _pre_forward_hook(loss_scale):
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
    MTPLossAutoScaler.set_loss_scale(loss_scale)


def build_model(model_cfg: Qwen3MoEConfig, *, impl_cfg: ImplConfig) -> ModelBundle:
    _prepare(model_cfg, impl_cfg)

    ps = init_parallel(impl_cfg.parallel)
    chunks = build_vpp_chunks(_chunk_factory, model_cfg, impl_cfg, ps)
    set_cross_entropy_fusion(chunks, impl_cfg.cross_entropy_fusion)

    # #114: recompute and offload walk the same transformer_units (chunk.layers).
    apply_recompute_offload(
        chunks,
        lambda chunk: chunk.layers,
        MODULE_MAP,
        recompute=impl_cfg.recompute,
        offload=impl_cfg.offload,
    )

    # LoRA freeze+stats must run before the optimizer sees params.
    lora_config = normalize_lora_config(impl_cfg.lora)
    lora_stats = None
    if lora_config.enabled:
        lora_stats = {"chunks": []}
        for chunk in chunks:
            freeze_stats = freeze_non_lora_params(chunk)
            trainable_stats = trainable_param_stats(chunk)
            lora_stats["chunks"].append({**freeze_stats, **trainable_stats})

    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    if impl_cfg.optimizer == "dist_opt":
        # register_hooks=False: qwen3_moe skips the megatron grad-sync hooks.
        optimizer, finalize_grads = wire_dist_opt(
            chunks,
            model_cfg,
            impl_cfg,
            ps,
            name="qwen3_moe",
            is_expert=is_expert_param,
            placement_fn=PLACEMENT_FN,
            deterministic=impl_cfg.deterministic,
            register_hooks=False,
        )
        optimizer_backend = "dist_opt"
    elif impl_cfg.optimizer == "fsdp2":
        post_model_load_hook = make_fsdp2_post_load_hook(
            chunks,
            impl_cfg,
            ps,
            unit_modules=_fsdp2_unit_modules(),
            expert_classifier=is_expert_param,
            deterministic=impl_cfg.deterministic,
        )
        optimizer_backend = "fsdp2"
    elif impl_cfg.optimizer is None:
        optimizer_backend = "none"
    else:
        raise ValueError(f"Unknown qwen3_moe lite optimizer: {impl_cfg.optimizer!r}.")

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step if impl_cfg.use_thd else _forward_step_bshd,
        extras={
            "model_cfg": model_cfg,
            "pre_forward_hook": _pre_forward_hook,
            "optimizer_backend": optimizer_backend,
            "post_model_load_hook": post_model_load_hook,
            "lora_config": lora_config,
            "lora_stats": lora_stats,
        },
    )


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: Qwen3MoEConfig, ps: ParallelState
) -> None:
    """Load HF pretrained weights into model chunk."""
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(
    chunks: list[nn.Module], model_cfg: Qwen3MoEConfig, ps: ParallelState, **kwargs
):
    """Export HF weights from model chunks."""
    from megatron.lite.model.qwen3_moe.lite.checkpoint import export_hf_weights as _export

    for chunk in chunks:
        yield from _export(chunk, model_cfg, ps, **kwargs)


def vocab_size(model_cfg: Qwen3MoEConfig) -> int | None:
    return getattr(model_cfg, "vocab_size", None)

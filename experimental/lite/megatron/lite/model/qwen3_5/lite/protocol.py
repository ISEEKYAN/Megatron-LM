# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 lite impl — native model protocol for Megatron Lite runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from megatron.lite.model.protocol_utils import (
    add_cross_entropy_fusion,
    add_loss_context_kwargs,
    pack_thd_forward_kwargs,
    set_cross_entropy_fusion,
    unpack_thd_forward_output,
)
from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_5.lite.checkpoint import EXPERT_CLASSIFIER, PLACEMENT_FN
from megatron.lite.model.qwen3_5.lite.checkpoint import export_hf_weights as _export_hf_weights_impl
from megatron.lite.model.qwen3_5.lite.checkpoint import load_hf_weights as _load_hf_weights_impl
from megatron.lite.model.qwen3_5.lite.checkpoint import save_hf_weights as _save_hf_weights_impl
from megatron.lite.model.compose import assemble
from megatron.lite.primitive.bundle import ModelBundle
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
    "save_hf_weights",
    "vocab_size",
]


def is_expert_param(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_deepep: bool = False
    use_thd: bool = False
    cross_entropy_fusion: bool = False
    hf_path: str = ""
    attention_backend_override: str | None = None
    router_aux_loss_coef: float | None = None
    router_bias_rate: float = 0.0
    deterministic: bool = True
    optimizer_config: OptimizerConfig | None = None
    mtp_enable: bool = False
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool | None = None
    mount_vision_model: bool = False
    # GatedDeltaNet context-parallel mode: "headwise" (default, head-parallel a2a;
    # bitwise-exact vs CP-off and memory-sharded), "replicated" (all-gather, exact but
    # full sequence replicated on every rank), or "chunkwise" (FLA ring; seq-shard +
    # all heads, best memory; bf16-floor vs CP-off, faithful packing-aware mirror of
    # upstream Megatron linear_cp_mode='chunkwise').
    gdn_cp_mode: str = "headwise"


def _full_attn_module(layer, name: str):
    full_attn = getattr(layer, "full_attn", None)
    return getattr(full_attn, name, None) if full_attn is not None else None


MODULE_MAP = {
    "core_attn": lambda layer: _full_attn_module(layer, "core_attn"),
    "experts": lambda layer: layer.moe.experts,
    "moe": lambda layer: layer.moe,
    "router": lambda layer: layer.moe.router,
    "mlp_norm": lambda layer: layer.mlp_norm,
    "attn_proj": lambda layer: _full_attn_module(layer, "proj"),
    "linear_attn": lambda layer: layer.linear_attn,
}


def build_model_config(source: str | Path | dict, **overrides) -> Qwen35Config:
    """Build Qwen3.5 architecture config from HF source."""
    if isinstance(source, dict):
        cfg = Qwen35Config._from_hf_dict(source)
    else:
        cfg = Qwen35Config.from_hf(str(source))
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    kwargs = pack_thd_forward_kwargs(model, batch)
    add_loss_context_kwargs(kwargs)
    add_cross_entropy_fusion(kwargs, model)
    return model(**kwargs)


def _forward_step_bshd(model: nn.Module, batch: PackedBatch) -> dict:
    """Dense [b=1, s] forward for a single packed sequence (no THD packing).

    Used for deterministic parity comparison vs a dense Megatron-Core reference:
    the THD GatedDeltaNet kernel is non-deterministic, whereas the dense path is
    deterministic. CP=1 only (single unpadded sequence => dense == THD tokens).
    """
    input_ids = batch.input_ids.reshape(1, -1)
    labels = batch.labels.reshape(1, -1) if batch.labels is not None else None
    kwargs: dict[str, Any] = {"input_ids": input_ids, "labels": labels, "packed_seq_params": None}
    add_cross_entropy_fusion(kwargs, model)
    return model(**kwargs)


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output) -> Any:
    return unpack_thd_forward_output(model, batch, output)


def _make_aux_loss_hook():
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    def hook(scale: torch.Tensor) -> None:
        MoEAuxLossAutoScaler.set_loss_scale(scale)
        MTPLossAutoScaler.set_loss_scale(scale)

    return hook


def _effective_deterministic(model_cfg: Qwen35Config, impl_cfg: ImplConfig) -> bool:
    # THD GatedDeltaNet kernel is non-deterministic; force off in that case.
    if impl_cfg.use_thd and impl_cfg.deterministic and "linear_attention" in model_cfg.layer_types:
        return False
    return impl_cfg.deterministic


def _prepare(model_cfg: Qwen35Config, impl_cfg: ImplConfig, ps: ParallelState) -> None:
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
    model_cfg: Qwen35Config, impl_cfg: ImplConfig, ps: ParallelState, vpp_chunk_id: int | None
) -> nn.Module:
    from megatron.lite.model.qwen3_5.lite.model import Qwen35Model

    mtp_enable = bool(impl_cfg.mtp_enable)
    vpp = None if impl_cfg.parallel.vpp == 1 else impl_cfg.parallel.vpp
    train_cfg = SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=vpp,
        use_deepep=impl_cfg.use_deepep, fp8=False,
        recompute_modules=parse_recompute_spec(impl_cfg.recompute),
        deterministic=_effective_deterministic(model_cfg, impl_cfg),
    )
    return Qwen35Model(
        model_cfg, train_cfg, ps, vpp_chunk_id=vpp_chunk_id,
        router_bias_rate=impl_cfg.router_bias_rate,
        use_thd=impl_cfg.use_thd,
        hf_path=impl_cfg.hf_path,
        attention_backend_override=impl_cfg.attention_backend_override,
        mtp_enable=mtp_enable,
        mtp_enable_train=mtp_enable and bool(impl_cfg.mtp_enable_train),
        mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
        mount_vision_model=impl_cfg.mount_vision_model,
        gdn_cp_mode=impl_cfg.gdn_cp_mode,
    )


def _fsdp2_unit_modules() -> tuple[type[nn.Module], ...]:
    from megatron.lite.model.qwen3_5.lite.model import Qwen35Layer

    return (Qwen35Layer,)


def build_model(model_cfg: Qwen35Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    return assemble(
        model_cfg,
        impl_cfg,
        name="qwen3_5",
        chunk_factory=_chunk_factory,
        transformer_units=lambda chunk: chunk.layers,
        module_map=MODULE_MAP,
        forward_step=_forward_step if impl_cfg.use_thd else _forward_step_bshd,
        expert_classifier=is_expert_param,
        placement_fn=PLACEMENT_FN,
        fsdp2_unit_modules=_fsdp2_unit_modules,
        prepare=_prepare,
        post_chunk_hook=lambda chunks, impl: set_cross_entropy_fusion(
            chunks, impl.cross_entropy_fusion
        ),
        pre_forward_hook_factory=_make_aux_loss_hook,
        # THD GatedDeltaNet kernel is non-deterministic; force off on that path.
        fsdp2_deterministic_fn=lambda impl: _effective_deterministic(model_cfg, impl),
    )


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: Qwen35Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(
    chunks: list[nn.Module], model_cfg: Qwen35Config, ps: ParallelState, **kwargs
):
    yield from _export_hf_weights_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(
    chunks: list[nn.Module], path: str, model_cfg: Qwen35Config, ps: ParallelState
) -> None:
    _save_hf_weights_impl(chunks, path, model_cfg, ps)


def vocab_size(model_cfg) -> int | None:
    cfg = getattr(model_cfg, "text_config", model_cfg)
    return getattr(cfg, "vocab_size", None)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kimi K2 lite impl - native model protocol for Megatron Lite runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from megatron.lite.model.kimi_k2.config import KimiK2Config
from megatron.lite.model.protocol_utils import (
    add_cross_entropy_fusion,
    add_loss_context_kwargs,
    pack_thd_forward_kwargs,
    set_cross_entropy_fusion,
    unpack_thd_forward_output,
)
from megatron.lite.model.compose import ModelSpec, assemble
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch


def EXPERT_CLASSIFIER(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


def PLACEMENT_FN(param_name: str) -> list:
    from megatron.lite.model.kimi_k2.lite.checkpoint import PLACEMENT_FN as placement_fn

    return placement_fn(param_name)


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


def _maybe(module_name: str):
    def getter(layer):
        module = getattr(layer, module_name, None)
        return module

    return getter


def _moe_module(name: str):
    def getter(layer):
        moe = getattr(layer, "moe", None)
        return getattr(moe, name, None) if moe is not None else None

    return getter


MODULE_MAP = {
    "core_attn": lambda layer: layer.self_attention.core_attn,
    "experts": _moe_module("experts"),
    "moe": _maybe("moe"),
    "router": _moe_module("router"),
    "mlp": _maybe("mlp"),
    "mlp_norm": lambda layer: layer.mlp_norm,
    "attn_proj": lambda layer: layer.self_attention.linear_proj,
    "mla": lambda layer: layer.self_attention,
}


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


def build_model_config(source: str | Path | dict, **overrides) -> KimiK2Config:
    if isinstance(source, dict):
        cfg = KimiK2Config._from_hf_dict(source)
    else:
        cfg = KimiK2Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    kwargs = pack_thd_forward_kwargs(model, batch)
    add_loss_context_kwargs(kwargs)
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


def _iter_transformer_units(chunk: nn.Module):
    return chunk.layers


def _prepare(model_cfg: KimiK2Config, impl_cfg: ImplConfig, ps: ParallelState) -> None:
    p = impl_cfg.parallel
    if impl_cfg.use_deepep and (p.etp is not None and p.etp > 1):
        raise ValueError("use_deepep and etp>1 are mutually exclusive")
    if impl_cfg.router_aux_loss_coef is not None:
        model_cfg.aux_loss_alpha = impl_cfg.router_aux_loss_coef
    if impl_cfg.mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError("mtp_enable=True but HF config has no num_nextn_predict_layers.")
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
        if impl_cfg.mtp_use_repeated_layer is not None:
            model_cfg.mtp_use_repeated_layer = impl_cfg.mtp_use_repeated_layer
    elif hasattr(model_cfg, "num_nextn_predict_layers"):
        model_cfg.num_nextn_predict_layers = 0


def _chunk_factory(
    model_cfg: KimiK2Config, impl_cfg: ImplConfig, ps: ParallelState, vpp_chunk_id: int | None
) -> nn.Module:
    from megatron.lite.model.kimi_k2.lite.model import KimiK2Model

    mtp_enable = bool(impl_cfg.mtp_enable)
    vpp = None if impl_cfg.parallel.vpp == 1 else impl_cfg.parallel.vpp
    train_cfg = SimpleNamespace(
        tp=ps.tp_size, ep=ps.ep_size, etp=ps.etp_size, pp=ps.pp_size, cp=ps.cp_size, vpp=vpp,
        use_deepep=impl_cfg.use_deepep, fp8=False,
        recompute_modules=parse_recompute_spec(impl_cfg.recompute),
        deterministic=impl_cfg.deterministic,
    )
    return KimiK2Model(
        model_cfg, train_cfg, ps, vpp_chunk_id=vpp_chunk_id,
        router_bias_rate=impl_cfg.router_bias_rate,
        use_thd=impl_cfg.use_thd,
        hf_path=impl_cfg.hf_path,
        attention_backend_override=impl_cfg.attention_backend_override,
        mtp_enable=mtp_enable,
        mtp_enable_train=mtp_enable and bool(impl_cfg.mtp_enable_train),
        mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
    )


def _fsdp2_unit_modules() -> tuple[type[nn.Module], ...]:
    from megatron.lite.model.kimi_k2.lite.model import KimiK2Layer

    return (KimiK2Layer,)


def build_model(model_cfg: KimiK2Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    spec = ModelSpec(
        name="kimi_k2",
        chunk_factory=_chunk_factory,
        transformer_units=_iter_transformer_units,
        module_map=MODULE_MAP,
        forward_step=_forward_step,
        expert_classifier=is_expert_param,
        placement_fn=PLACEMENT_FN,
        fsdp2_unit_modules=_fsdp2_unit_modules,
        prepare=_prepare,
        post_chunk_hook=lambda chunks, impl: set_cross_entropy_fusion(
            chunks, impl.cross_entropy_fusion
        ),
        pre_forward_hook_factory=_make_aux_loss_hook,
    )
    return assemble(spec, model_cfg, impl_cfg)


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: KimiK2Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    from megatron.lite.model.kimi_k2.lite.checkpoint import load_hf_weights as load_impl

    load_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(chunks, model_cfg: KimiK2Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.kimi_k2.lite.checkpoint import export_hf_weights as export_impl

    yield from export_impl(chunks, model_cfg, ps, **kwargs)


def vocab_size(model_cfg) -> int | None:
    cfg = getattr(model_cfg, "text_config", model_cfg)
    return getattr(cfg, "vocab_size", None)


__all__ = [
    "EXPERT_CLASSIFIER",
    "ImplConfig",
    "PLACEMENT_FN",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "is_expert_param",
    "load_hf_weights",
    "vocab_size",
]

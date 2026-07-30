# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron Lite model protocol for native Tencent Hy3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.hy3.config import Hy3Config
from megatron.lite.model.hy3.lite.checkpoint import (
    EXPERT_CLASSIFIER,
    PLACEMENT_FN,
    load_hf_weights as _load_hf_weights,
)
from megatron.lite.model.hy3.lite.model import Hy3Model, Hy3TransformerLayer
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.quantization import (
    QATSpec,
    apply_qat_to_chunks,
    normalize_qat_spec,
)
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "mc"
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_deepep: bool = False
    use_thd: bool = False
    optimizer_config: OptimizerConfig | None = None
    mtp_enable: bool = False
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1
    deterministic: bool = True
    qat: QATSpec | dict | None = None


MODULE_MAP = {
    "core_attn": lambda layer: layer.attn.core_attn,
    "mlp": lambda layer: layer.mlp,
    "experts": lambda layer: layer.moe.experts if layer.moe is not None else None,
    "moe": lambda layer: layer.moe,
    "router": lambda layer: layer.moe.router if layer.moe is not None else None,
    "mlp_norm": lambda layer: layer.mlp_norm,
    "attn_proj": lambda layer: layer.attn.proj,
}


def build_model_config(source: str | Path | dict, **overrides) -> Hy3Config:
    config = (
        Hy3Config._from_hf_dict(source, **overrides)
        if isinstance(source, dict)
        else Hy3Config.from_hf(str(source), **overrides)
    )
    return config


def _forward_step(model: nn.Module, batch: dict) -> dict:
    kwargs = {"input_ids": batch["input_ids"], "labels": batch["labels"]}
    for key in (
        "packed_seq_params",
        "position_ids",
        "loss_mask",
        "temperature",
        "use_fused_kernels",
        "calculate_entropy",
        "return_log_probs",
    ):
        if key in batch:
            kwargs[key] = batch[key]
    if kwargs["input_ids"].dim() == 1:
        kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
    return model(**kwargs)


def build_model(model_cfg: Hy3Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    parallel = impl_cfg.parallel
    if impl_cfg.use_deepep and parallel.etp is not None and parallel.etp > 1:
        raise ValueError("use_deepep and etp>1 are mutually exclusive")
    if impl_cfg.mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError("mtp_enable=True but the HF config has no MTP layer")
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
    else:
        model_cfg.num_nextn_predict_layers = 0

    ps = init_parallel(parallel)
    recompute = parse_recompute_spec(impl_cfg.recompute)
    model_kwargs: dict[str, Any] = {
        "use_deepep": impl_cfg.use_deepep,
        "fp8": False,
        "recompute_modules": recompute,
        "use_thd": impl_cfg.use_thd,
        "mtp_enable": impl_cfg.mtp_enable,
        "mtp_enable_train": impl_cfg.mtp_enable and impl_cfg.mtp_enable_train,
        "mtp_detach_encoder": impl_cfg.mtp_detach_encoder,
    }
    vpp = None if parallel.vpp == 1 else parallel.vpp
    chunks = [
        Hy3Model(
            model_cfg,
            ps,
            vpp=vpp,
            vpp_chunk_id=index if vpp is not None else None,
            **model_kwargs,
        )
        .to(torch.bfloat16)
        .cuda()
        for index in range(vpp or 1)
    ]
    if recompute:
        for chunk in chunks:
            apply_recompute(chunk.layers, recompute, MODULE_MAP)
    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        for chunk in chunks:
            apply_offload(chunk.layers, impl_cfg.offload, MODULE_MAP)

    # Parametrize before optimizer construction so it captures the BF16 master.
    apply_qat_to_chunks(chunks, normalize_qat_spec(impl_cfg.qat))

    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    if impl_cfg.optimizer == "mc":
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_mc_training_optimizer,
        )

        optimizer, finalize_grads = build_mc_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=impl_cfg,
            ps=ps,
            model_name="hy3",
            is_expert=EXPERT_CLASSIFIER,
            deterministic=impl_cfg.deterministic,
        )
        optimizer_backend = "mc"
    elif impl_cfg.optimizer == "fsdp2":
        optimizer_backend = "fsdp2"

        def build_optimizer_after_load():
            from megatron.lite.primitive.optimizers.fsdp2 import (
                build_fsdp2_training_optimizer,
            )

            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(Hy3TransformerLayer,),
                    expert_classifier=EXPERT_CLASSIFIER,
                    deterministic=impl_cfg.deterministic,
                    vpp=parallel.vpp,
                    leaf_module_names=(),
                )
            }

        post_model_load_hook = build_optimizer_after_load
    elif impl_cfg.optimizer is None:
        optimizer_backend = "none"
    else:
        raise ValueError(f"Unknown Hy3 lite optimizer: {impl_cfg.optimizer!r}")

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step,
        extras={
            "model_cfg": model_cfg,
            "pre_forward_hook": MTPLossAutoScaler.set_loss_scale,
            "optimizer_backend": optimizer_backend,
            "post_model_load_hook": post_model_load_hook,
        },
    )


def load_hf_weights(
    chunk: nn.Module,
    hf_path: str,
    model_cfg: Hy3Config,
    ps: ParallelState,
) -> None:
    if hf_path:
        _load_hf_weights(chunk, hf_path, model_cfg, ps)


def export_hf_weights(chunks, model_cfg: Hy3Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.hy3.lite.checkpoint import export_hf_weights as export

    for chunk in chunks:
        yield from export(chunk, model_cfg, ps, **kwargs)


def vocab_size(model_cfg: Hy3Config) -> int:
    return model_cfg.vocab_size


__all__ = [
    "ImplConfig",
    "MODULE_MAP",
    "PLACEMENT_FN",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "load_hf_weights",
    "vocab_size",
]

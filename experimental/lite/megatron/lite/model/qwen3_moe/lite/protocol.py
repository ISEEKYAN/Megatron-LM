# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# isort: skip_file
"""Qwen3MoE lite impl — model protocol for Megatron Lite runtime.

This file is the reference implementation of the Megatron Lite model protocol.
New model authors: copy this file and adapt.

Protocol convention (what runtime calls):
  Required:
    ImplConfig                                      — @dataclass, per-impl knobs
    build_model_config(source, **overrides)          → ModelConfig
    build_model(model_cfg, *, impl_cfg)              → ModelBundle
  Optional (in ModelBundle.extras or module-level):
    load_hf_weights(chunk, hf_path, model_cfg, ps)  — HF weight loading
    export_hf_weights(chunks, model_cfg, ps)         — HF weight export
    save_hf_weights(chunks, path, model_cfg, ps)     — HF checkpoint writing
    vocab_size(model_cfg) -> int                     — benchmark metadata
  Escape hatch:
    create_runtime(hf_path, cfg) -> Runtime          — fully override runtime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from megatron.lite.model.protocol_utils import (
    add_cross_entropy_fusion,
    add_loss_context_kwargs,
    pack_thd_forward_kwargs,
)
from megatron.lite.model.protocol_utils import (
    router_replay_roots as router_replay_roots,
)
from megatron.lite.model.protocol_utils import (
    set_cross_entropy_fusion,
    unpack_thd_forward_output,
)
from megatron.lite.model.qwen3_moe.common import is_expert_param
from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite.checkpoint import (
    EXPERT_CLASSIFIER,
    PLACEMENT_FN,
)
from megatron.lite.model.qwen3_moe.lite.checkpoint import (
    load_hf_weights as _load_hf_weights_impl,
)
from megatron.lite.model.qwen3_moe.lite.lora_adapter import LORA_TARGETS
from megatron.lite.model.qwen3_moe.lite.multi_lora import MoELoraSidecar
from megatron.lite.model.qwen3_moe.lite.model import MTPLossAutoScaler, Qwen3MoEModel
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.modules.lora import (
    LoraSpec,
    apply_olora_tail_init,
    normalize_lora_spec,
)
from megatron.lite.primitive.modules.multi_lora_bank import (
    DenseLoraBank,
    LoraBankPartition,
    MultiLoraSpec,
    MultiLoraTrainingState,
    NamedLoraBankRegistry,
    normalize_multi_lora_spec,
    validate_multi_lora_parallel_support,
)
from megatron.lite.primitive.modules.lora_apply import (
    apply_lora_to_chunks,
    validate_lora_parallel_support,
)
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.quantization import (
    QATSpec,
    apply_qat_to_chunks,
    normalize_qat_spec,
)
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

__all__ = [
    "EXPERT_CLASSIFIER",
    "ImplConfig",
    "PLACEMENT_FN",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "export_hf_lora_adapter",
    "load_hf_weights",
    "save_hf_weights",
    "vocab_size",
]

# ---------------------------------------------------------------------------
# ImplConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImplConfig:
    """Lite impl knobs. Constructed by runtime from user config."""

    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"  # None = no optimizer (inference)
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
    lora: LoraSpec | dict | None = None
    multi_lora: MultiLoraSpec | dict | None = None
    # Weight-only QAT: float fp8_e4m3 / mxfp4 or int8 / int4. Default None = disabled.
    qat: QATSpec | dict | None = None


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


# ---------------------------------------------------------------------------
# Required: build_model_config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Required: build_model
# ---------------------------------------------------------------------------


def _inject_multi_lora_sidecars(kwargs, batch: PackedBatch, multi_lora_state) -> None:
    if "multi_lora_sidecars" in batch.extras:
        raise ValueError(
            "multi_lora_sidecars is model-owned; pass multi_lora_slots instead."
        )
    slots = batch.extras.get("multi_lora_slots")
    if slots is None:
        if multi_lora_state is not None:
            raise ValueError(
                "enabled impl_cfg.multi_lora requires multi_lora_slots in batch.extras"
            )
        return
    if multi_lora_state is None:
        raise ValueError("multi_lora_slots requires enabled impl_cfg.multi_lora.")
    local_layers = set(multi_lora_state.local_layer_indices)
    missing_local_layers = local_layers.difference(slots)
    if missing_local_layers:
        raise ValueError(
            "multi_lora_slots is missing local pipeline layers: "
            f"{sorted(missing_local_layers)}"
        )

    def _sidecar(layer_idx, layer_slots):
        attention_banks = multi_lora_state.attention_banks_for_layer(layer_idx)
        return MoELoraSidecar(
            *multi_lora_state.banks_for_layer(layer_idx),
            lora_indices=layer_slots,
            scale=multi_lora_state.scale,
            # FC banks are EP-replicated while attention banks stay on their
            # TP carrier; this flag is consumed only by MoE FC sidecar calls.
            requires_explicit_ep_sync=True,
            qkv=None if attention_banks is None else attention_banks[0],
            proj=None if attention_banks is None else attention_banks[1],
        )

    kwargs["multi_lora_sidecars"] = {
        layer_idx: _sidecar(layer_idx, layer_slots)
        for layer_idx, layer_slots in slots.items()
        if layer_idx in local_layers
    }


def _forward_step(
    model: nn.Module, batch: PackedBatch, *, multi_lora_state=None
) -> dict:
    kwargs = pack_thd_forward_kwargs(model, batch)
    _inject_multi_lora_sidecars(kwargs, batch, multi_lora_state)
    add_loss_context_kwargs(kwargs, include_return_log_probs=True)
    add_cross_entropy_fusion(kwargs, model)
    return model(**kwargs)


def _forward_step_bshd(
    model: nn.Module, batch: PackedBatch, *, multi_lora_state=None
) -> dict:
    labels = batch.labels.reshape(1, -1) if batch.labels is not None else None
    kwargs = {
        "input_ids": batch.input_ids.reshape(1, -1),
        "labels": labels,
        "packed_seq_params": None,
    }
    _inject_multi_lora_sidecars(kwargs, batch, multi_lora_state)
    return model(**kwargs)


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output) -> Any:
    return unpack_thd_forward_output(model, batch, output)


def _build_multi_lora_training_state(
    chunks: list[nn.Module], model_cfg: Qwen3MoEConfig, spec: MultiLoraSpec
) -> MultiLoraTrainingState | None:
    """Create one model-owned, expert-shared native-surface registry."""
    if not spec.enabled:
        return None
    device = next(chunks[0].parameters()).device
    dtype = next(chunks[0].parameters()).dtype
    banks: dict[str, DenseLoraBank] = {}
    layer_surfaces: dict[int, tuple[str, str]] = {}
    attention_surfaces: dict[int, tuple[str, str]] = {}
    for chunk in chunks:
        for layer in chunk.layers:
            layer_idx = layer.layer_idx
            fc1 = DenseLoraBank(
                nn.Parameter(
                    torch.empty(
                        len(spec.names),
                        spec.rank,
                        model_cfg.hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                ),
                nn.Parameter(
                    torch.empty(
                        len(spec.names),
                        model_cfg.moe_intermediate_size * 2,
                        spec.rank,
                        device=device,
                        dtype=dtype,
                    )
                ),
            )
            fc2 = DenseLoraBank(
                nn.Parameter(
                    torch.empty(
                        len(spec.names),
                        spec.rank,
                        model_cfg.moe_intermediate_size,
                        device=device,
                        dtype=dtype,
                    )
                ),
                nn.Parameter(
                    torch.empty(
                        len(spec.names),
                        model_cfg.hidden_size,
                        spec.rank,
                        device=device,
                        dtype=dtype,
                    )
                ),
            )
            attn = layer.attn
            # TP1 test doubles and the TP1 base module need no distributed
            # attribute; real TP modules provide their authoritative size.
            tp_size = int(getattr(attn.qkv, "tp_size", 1))
            if spec.rank % tp_size:
                raise ValueError("multi-LoRA rank must be divisible by TP size.")
            # Preserve TE's fused LayerNorm+QKV GEMM and request the exact
            # normalized activation it already produces for the sidecar delta.
            attn.qkv.linear.return_layernorm_output = True
            attn.qkv.linear.return_layernorm_output_gathered = False
            qkv = DenseLoraBank(
                a_bank=nn.Parameter(
                    torch.empty(
                        len(spec.names),
                        spec.rank // tp_size,
                        model_cfg.hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                ),
                b_bank=nn.Parameter(
                    torch.empty(
                        len(spec.names),
                        attn.qkv.local_out,
                        spec.rank,
                        device=device,
                        dtype=dtype,
                    )
                ),
                partition=LoraBankPartition(
                    tp_size=tp_size, rank_partitioned_a=tp_size > 1
                ),
            )
            proj = DenseLoraBank(
                a_bank=nn.Parameter(
                    torch.empty(
                        len(spec.names),
                        spec.rank,
                        attn.proj.local_in,
                        device=device,
                        dtype=dtype,
                    )
                ),
                b_bank=nn.Parameter(
                    torch.empty(
                        len(spec.names),
                        model_cfg.hidden_size // tp_size,
                        spec.rank,
                        device=device,
                        dtype=dtype,
                    )
                ),
                partition=LoraBankPartition(
                    tp_size=tp_size, output_partitioned_b=tp_size > 1
                ),
            )
            nn.init.kaiming_uniform_(fc1.a_bank, a=5**0.5)
            nn.init.zeros_(fc1.b_bank)
            nn.init.kaiming_uniform_(fc2.a_bank, a=5**0.5)
            nn.init.zeros_(fc2.b_bank)
            nn.init.kaiming_uniform_(qkv.a_bank, a=5**0.5)
            nn.init.zeros_(qkv.b_bank)
            nn.init.kaiming_uniform_(proj.a_bank, a=5**0.5)
            nn.init.zeros_(proj.b_bank)
            for bank, is_expert_bank in (
                (fc1, True),
                (fc2, True),
                (qkv, False),
                (proj, False),
            ):
                tensor_model_parallel = (
                    bank.partition.rank_partitioned_a
                    or bank.partition.output_partitioned_b
                )
                for parameter in (bank.a_bank, bank.b_bank):
                    parameter.tensor_model_parallel = tensor_model_parallel
                    # FC banks follow expert-DP ownership plus the MoE EP
                    # reduction; attention banks stay in regular dense-DP.
                    parameter.allreduce = not is_expert_bank
            fc1_surface = f"layers.{layer_idx}.moe.experts._fc1_weight_0"
            fc2_surface = f"layers.{layer_idx}.moe.experts._fc2_weight_0"
            qkv_surface = f"layers.{layer_idx}.attn.qkv.linear.weight"
            proj_surface = f"layers.{layer_idx}.attn.proj.linear.weight"
            layer_surfaces[layer_idx] = (fc1_surface, fc2_surface)
            attention_surfaces[layer_idx] = (qkv_surface, proj_surface)
            # One canonical grouped native surface per layer.  The HF exporter
            # expands this shared bank across experts exactly once; registering
            # every expert here would make that exporter expand an E-sized map E
            # times and overwrite the same output keys.
            banks[fc1_surface] = fc1
            banks[fc2_surface] = fc2
            banks[qkv_surface] = qkv
            banks[proj_surface] = proj
    registry = NamedLoraBankRegistry(
        banks=banks,
        names={name: slot for slot, name in enumerate(spec.names)},
        rank=spec.rank,
        alpha=spec.alpha,
        base_model_identity={},
        lora_spec=LoraSpec(
            enabled=True, rank=spec.rank, alpha=spec.alpha, use_rslora=spec.use_rslora
        ),
    )
    state = MultiLoraTrainingState(registry, layer_surfaces, attention_surfaces)
    chunks[0].add_module("multi_lora_training_state", state)
    return state


def build_model(model_cfg: Qwen3MoEConfig, *, impl_cfg: ImplConfig) -> ModelBundle:
    """Build lite Qwen3MoE: model, parallel state, optimizer — everything.

    Model owns all construction. Runtime just consumes the ModelBundle.
    """
    p = impl_cfg.parallel
    lora_spec = normalize_lora_spec(impl_cfg.lora)
    multi_lora_spec = normalize_multi_lora_spec(impl_cfg.multi_lora)
    if lora_spec.enabled and multi_lora_spec.enabled:
        raise ValueError(
            "single LoRA and model-owned multi-LoRA cannot be enabled together."
        )
    if multi_lora_spec.enabled and impl_cfg.optimizer == "fsdp2":
        raise ValueError(
            "model-owned multi-LoRA currently supports dist_opt only; FSDP2 "
            "parameter replacement would invalidate the bank registry identity."
        )
    validate_lora_parallel_support(lora_spec, etp_size=p.etp)
    validate_multi_lora_parallel_support(
        multi_lora_spec, tp_size=p.tp, etp_size=p.etp, use_deepep=impl_cfg.use_deepep
    )

    # ── validation ──
    if impl_cfg.use_deepep and (p.etp is not None and p.etp > 1):
        raise ValueError("use_deepep and etp>1 are mutually exclusive")

    # ── override model config from impl_cfg ──
    if impl_cfg.router_aux_loss_coef is not None:
        model_cfg.router_aux_loss_coef = impl_cfg.router_aux_loss_coef
    mtp_enable = bool(impl_cfg.mtp_enable)
    mtp_enable_train = mtp_enable and bool(impl_cfg.mtp_enable_train)
    if mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError(
                "mtp_enable=True but HF config has no num_nextn_predict_layers."
            )
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
        if impl_cfg.mtp_use_repeated_layer is not None:
            model_cfg.mtp_use_repeated_layer = impl_cfg.mtp_use_repeated_layer
    else:
        model_cfg.num_nextn_predict_layers = 0

    # ── parallel state (model creates its own) ──
    ps = init_parallel(p)
    deterministic = impl_cfg.deterministic

    # ── build chunks ──
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    model_kwargs: dict[str, Any] = dict(
        use_deepep=impl_cfg.use_deepep,
        fp8=False,
        recompute_modules=recompute_spec,
        router_bias_rate=impl_cfg.router_bias_rate,
        use_thd=impl_cfg.use_thd,
        mtp_enable=mtp_enable,
        mtp_enable_train=mtp_enable_train,
        mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
    )

    vpp = None if p.vpp == 1 else p.vpp
    meta_init = impl_cfg.optimizer == "fsdp2"

    def build_chunk(**kwargs):
        with torch.device("meta") if meta_init else nullcontext():
            chunk = Qwen3MoEModel(model_cfg, ps, **kwargs, **model_kwargs).to(torch.bfloat16)
        if meta_init:
            _validate_meta_parameters(chunk)
        chunk._mlite_meta_init = meta_init
        return chunk if meta_init else chunk.cuda()

    chunks = [build_chunk()] if vpp is None else [
        build_chunk(vpp=vpp, vpp_chunk_id=i) for i in range(vpp)
    ]

    set_cross_entropy_fusion(chunks, impl_cfg.cross_entropy_fusion)

    # ── recompute ──
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(chunk.layers, recompute_spec, MODULE_MAP)

    # ── offload ──
    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        for chunk in chunks:
            apply_offload(chunk.layers, impl_cfg.offload, MODULE_MAP)

    multi_lora_state = _build_multi_lora_training_state(
        chunks, model_cfg, multi_lora_spec
    )

    # The HF loader resolves canonical parameter names from ``state_dict``.
    # LoRA wrappers add a ``.base.`` component, so attaching here would make
    # every wrapped base parameter invisible to the loader.  Keep disabled
    # LoRA inert, and attach enabled LoRA in the post-load/pre-wrap hook below.
    lora_stats = (
        None
        if lora_spec.enabled
        else apply_lora_to_chunks(chunks, lora_spec, ps=ps, model_targets=LORA_TARGETS)
    )

    def _attach_lora_after_load():
        nonlocal lora_stats
        if lora_stats is None:
            lora_stats = apply_lora_to_chunks(
                chunks, lora_spec, ps=ps, model_targets=LORA_TARGETS
            )
            if lora_spec.init == "olora_tail":
                for chunk in chunks:
                    apply_olora_tail_init(chunk)
        return lora_stats

    # Weight-only QAT (fake-quant/STE on the BF16 master, including MoE experts).
    # Must run before optimizer construction so dist_opt captures weight.original.
    apply_qat_to_chunks(chunks, normalize_qat_spec(impl_cfg.qat))

    # ── optimizer (model chooses which primitive) ──
    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    if impl_cfg.optimizer == "dist_opt":
        optimizer_backend = "dist_opt"

        def _build_dist_opt():
            from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
            from megatron.lite.primitive.optimizers.megatron_wrap import (
                build_dist_opt_training_optimizer,
            )

            built_optimizer, built_finalize_grads = build_dist_opt_training_optimizer(
                chunks,
                model_cfg=model_cfg,
                impl_cfg=impl_cfg,
                ps=ps,
                model_name="qwen3_moe",
                is_expert=is_expert_param,
                deterministic=deterministic,
            )
            attach_model_sharded_state_dict(
                chunks, ps, get_placements=PLACEMENT_FN, is_expert=is_expert_param
            )
            return built_optimizer, built_finalize_grads

        if lora_spec.enabled:

            def _post_model_load_hook():
                stats = _attach_lora_after_load()
                built_optimizer, built_finalize_grads = _build_dist_opt()
                return {
                    "optimizer": built_optimizer,
                    "finalize_grads": built_finalize_grads,
                    "extras": {"lora_stats": stats},
                }

            post_model_load_hook = _post_model_load_hook
        else:
            optimizer, finalize_grads = _build_dist_opt()
    elif impl_cfg.optimizer == "fsdp2":
        optimizer_backend = "fsdp2"

        def _post_model_load_hook():
            from megatron.lite.model.qwen3_moe.lite.model import TransformerLayer
            from megatron.lite.primitive.optimizers.fsdp2 import (
                build_fsdp2_training_optimizer,
            )

            stats = _attach_lora_after_load()
            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(TransformerLayer,),
                    expert_classifier=is_expert_param,
                    deterministic=deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    # Non-layer params stay under the root FSDP2 unit. The fused
                    # CE path reads head.col.linear.weight directly, and the
                    # embedding path is also driven from model.forward().
                    leaf_module_names=(),
                ),
                "extras": {"lora_stats": stats},
            }

        post_model_load_hook = _post_model_load_hook
    elif impl_cfg.optimizer is None:
        optimizer_backend = "none"
        if lora_spec.enabled:

            def _post_model_load_hook():
                return {"extras": {"lora_stats": _attach_lora_after_load()}}

            post_model_load_hook = _post_model_load_hook
    else:
        raise ValueError(f"Unknown qwen3_moe lite optimizer: {impl_cfg.optimizer!r}.")

    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    def _pre_forward_hook(loss_scale):
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        MTPLossAutoScaler.set_loss_scale(loss_scale)

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=partial(
            _forward_step if impl_cfg.use_thd else _forward_step_bshd,
            multi_lora_state=multi_lora_state,
        ),
        extras={
            "model_cfg": model_cfg,
            # Lite's router uses megatron.lite's MoEAuxLossAutoScaler; hand the
            # classmethod directly as the per-microbatch hook.
            "pre_forward_hook": _pre_forward_hook,
            "optimizer_backend": optimizer_backend,
            "post_model_load_hook": post_model_load_hook,
            "lora_spec": lora_spec,
            "lora_stats": lora_stats,
            "multi_lora_registry": (
                None if multi_lora_state is None else multi_lora_state.registry
            ),
            "multi_lora_training_state": multi_lora_state,
        },
    )


# ---------------------------------------------------------------------------
# Optional: load_hf_weights
# ---------------------------------------------------------------------------


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
    from megatron.lite.model.qwen3_moe.lite.checkpoint import (
        export_hf_weights as _export,
    )

    for chunk in chunks:
        yield from _export(chunk, model_cfg, ps, **kwargs)


def export_hf_lora_adapter(
    chunks: list[nn.Module], model_cfg: Qwen3MoEConfig, ps: ParallelState, **kwargs
):
    """Export LoRA factors in vLLM/PEFT naming (adapter-only rollout sync)."""
    from megatron.lite.model.qwen3_moe.lite.checkpoint import (
        export_hf_lora_adapter as _export_hf_lora_adapter_impl,
    )

    # A named dense bank is independent of a model chunk.  Export it once, not
    # once per PP/VPP chunk, through the same protocol entry runtime uses.
    if kwargs.get("multi_lora_registry") is not None:
        if not chunks:
            raise ValueError(
                "named multi-LoRA export requires at least one model chunk."
            )
        yield from _export_hf_lora_adapter_impl(chunks[0], model_cfg, ps, **kwargs)
        return
    for chunk in chunks:
        yield from _export_hf_lora_adapter_impl(chunk, model_cfg, ps, **kwargs)


def save_hf_weights(
    chunks: list[nn.Module], path: str, model_cfg: Qwen3MoEConfig, ps: ParallelState
) -> None:
    from megatron.lite.model.qwen3_moe.lite.checkpoint import save_hf_weights as _save

    _save(chunks, path, model_cfg, ps)


# ---------------------------------------------------------------------------
# Tooling metadata (benchmark / debug)
# ---------------------------------------------------------------------------


def vocab_size(model_cfg: Qwen3MoEConfig) -> int | None:
    return getattr(model_cfg, "vocab_size", None)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    EXPERT_CLASSIFIER,
    PLACEMENT_FN,
    export_hf_weights as _export_hf_weights_impl,
    load_hf_weights as _load_hf_weights_impl,
    save_hf_weights as _save_hf_weights_impl,
)
from megatron.lite.model.compose_kernel import ModelSpec, assemble
from megatron.lite.model.protocol_utils import (
    add_loss_context_kwargs,
    nested_from_packed,
    pack_r3_replay_mask as _pack_r3_replay_mask,
    pack_routed_experts as _pack_routed_experts,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.cp import (
    contiguous_position_ids_for_cp,
    contiguous_slice_for_cp,
    local_position_ids_for_cp,
    local_sequence_tensor_for_cp,
)
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.runtime.contracts import OptimizerConfig, PackedBatch, ParallelConfig


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    optimizer_config: OptimizerConfig | None = None
    hf_path: str = ""
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_thd: bool = False
    use_deepep: bool = False
    attention_backend_override: str | None = None
    deterministic: bool = True
    mtp_enable: bool = True
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_num_layers: int | None = None
    num_nextn_predict_layers: int | None = None
    mtp_loss_scaling_factor: float = 0.1


MODULE_MAP = {
    "attn": lambda layer: layer.self_attn,
    "core_attn": lambda layer: layer.self_attn,
    "moe": lambda layer: layer.mlp,
    "experts": lambda layer: layer.mlp.experts,
    "router": lambda layer: layer.mlp.gate,
    "attn_norm": lambda layer: layer.input_layernorm,
    "ffn_norm": lambda layer: layer.post_attention_layernorm,
}

# The Kimi-derived model has no ``attention_mask`` arg (CSA derives its causal /
# sliding-window masking from ``position_ids``, as the previous DS4 did with
# attention_mask=None); keep it out of the forward whitelist.
_MODEL_FORWARD_KEYS = (
    "input_ids",
    "position_ids",
    "labels",
    "loss_mask",
    "temperature",
    "calculate_entropy",
    "enable_mtp",
    "packed_seq_params",
)


def build_model_config(source: str | Path | dict, **overrides) -> DeepseekV4Config:
    if isinstance(source, dict):
        cfg = DeepseekV4Config._from_hf_dict(source)
    else:
        cfg = DeepseekV4Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _normalize_ds4_position_ids(position_ids):
    if position_ids is None:
        return None
    if position_ids.dim() == 3:
        if position_ids.size(0) == 3:
            position_ids = position_ids[0]
        elif position_ids.size(1) == 1:
            position_ids = position_ids.squeeze(1)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    return position_ids


def _as_batch_row(tensor):
    if tensor is not None and tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _infer_cp_local_seq_len(
    *,
    input_ids,
    position_ids,
    cp_size,
):
    seq_len = input_ids.size(1)
    if cp_size <= 1:
        return seq_len
    if position_ids is not None and position_ids.size(-1) in (seq_len, seq_len * cp_size):
        return seq_len
    return seq_len // cp_size if seq_len % cp_size == 0 else seq_len


# The 1-D-packed -> jagged-nested split is model-agnostic and lives in the shared
# protocol_utils layer. DS4 only forks the *CP layout* pair (contiguous DSA, see
# ``_prepare_packed_batch_kwargs`` below); this pre-CP primitive must not diverge,
# so alias the shared implementation instead of re-copying it. Keeping the local
# name preserves existing call sites and unit tests.
_nested_from_packed_tensor = nested_from_packed


def _prepare_packed_batch_kwargs(model, batch: PackedBatch) -> dict[str, Any]:
    ps = parallel_state_from_model(model) or ParallelState()
    seq_lens = batch.sizes().to(device=batch.input_ids.device)
    packed = pack_nested_thd(
        _nested_from_packed_tensor(batch.input_ids, seq_lens),
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group,
        split_cp=False,
        labels=_nested_from_packed_tensor(batch.labels, seq_lens),
        loss_mask=_nested_from_packed_tensor(batch.loss_mask, seq_lens),
        roll_labels=batch.labels is not None,
        roll_loss_mask=batch.loss_mask is not None,
    )
    kwargs: dict[str, Any] = {
        "input_ids": packed.input_ids,
        "labels": packed.labels,
        "loss_mask": packed.loss_mask,
        "position_ids": packed.position_ids,
        "packed_seq_params": packed.packed_seq_params,
        "enable_mtp": False,
    }
    add_loss_context_kwargs(kwargs)
    _prepare_packed_contiguous_cp_kwargs(model, kwargs)
    return {key: value for key, value in kwargs.items() if key in _MODEL_FORWARD_KEYS}


def _base_model_forward_kwargs(batch: PackedBatch):
    kwargs: dict[str, Any] = {"input_ids": _as_batch_row(batch.input_ids)}
    if batch.labels is not None:
        kwargs["labels"] = _as_batch_row(batch.labels)
    if batch.loss_mask is not None:
        kwargs["loss_mask"] = _as_batch_row(batch.loss_mask)
    add_loss_context_kwargs(kwargs)
    position_ids = _normalize_ds4_position_ids(getattr(batch, "position_ids", None))
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return kwargs


def _prepare_packed_contiguous_cp_kwargs(model, kwargs):
    ps = parallel_state_from_model(model) or ParallelState()
    if ps.cp_size <= 1:
        return kwargs
    for key in ("input_ids", "labels", "loss_mask", "position_ids"):
        tensor = kwargs.get(key)
        if tensor is not None:
            kwargs[key] = contiguous_slice_for_cp(tensor, ps.cp_rank, ps.cp_size, seq_dim=1)
    return kwargs


def _prepare_contiguous_cp_kwargs(model, kwargs):
    ps = parallel_state_from_model(model) or ParallelState()
    local_seq_len = _infer_cp_local_seq_len(
        input_ids=kwargs["input_ids"],
        position_ids=kwargs.get("position_ids"),
        cp_size=ps.cp_size,
    )
    kwargs["input_ids"] = local_sequence_tensor_for_cp(
        kwargs["input_ids"],
        local_seq_len=local_seq_len,
        cp_rank=ps.cp_rank,
        cp_size=ps.cp_size,
        name="input_ids",
    )
    if kwargs.get("position_ids") is None:
        full_seq_len = local_seq_len * ps.cp_size
        position_ids = contiguous_position_ids_for_cp(
            full_seq_len,
            cp_rank=ps.cp_rank,
            cp_size=ps.cp_size,
            device=kwargs["input_ids"].device,
        ).expand(kwargs["input_ids"].size(0), -1)
    else:
        position_ids = local_position_ids_for_cp(
            kwargs["position_ids"],
            batch=kwargs["input_ids"].size(0),
            local_seq_len=kwargs["input_ids"].size(1),
            cp_rank=ps.cp_rank,
            cp_size=ps.cp_size,
        )
    kwargs["position_ids"] = position_ids
    for key in ("labels", "loss_mask"):
        if kwargs.get(key) is not None:
            kwargs[key] = local_sequence_tensor_for_cp(
                kwargs[key],
                local_seq_len=kwargs["input_ids"].size(1),
                cp_rank=ps.cp_rank,
                cp_size=ps.cp_size,
                name=key,
            )
    return kwargs


def _prepare_model_forward_kwargs(model, batch: PackedBatch):
    # THD-packed inputs (1-D values, or a single padded [1, S] row) carry their own
    # cu_seqlens and go through the packed builder. A dense multi-row [B, S] batch is
    # split per row under contiguous CP, where contiguous_position_ids_for_cp rebuilds
    # the per-rank global position ids.
    input_ids = batch.input_ids
    is_thd_packed = input_ids.dim() == 1 or (input_ids.dim() == 2 and input_ids.size(0) == 1)
    if is_thd_packed:
        return _prepare_packed_batch_kwargs(model, batch)
    kwargs = _base_model_forward_kwargs(batch)
    return _prepare_contiguous_cp_kwargs(model, kwargs)


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    return model(**_prepare_model_forward_kwargs(model, batch))


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output) -> Any:
    # DeepSeek-V4 packs each sequence to the (zigzag) TE alignment but slices CP
    # contiguously for the fused DSA indexer, so reconstruct contiguously.
    ps = parallel_state_from_model(model) or ParallelState()
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    return unpack_thd_to_nested(output, meta, contiguous=True)


def pack_routed_experts(model: nn.Module, batch: PackedBatch, routed_experts):
    """Pack R3 routes using DS4's contiguous CP token layout."""

    return _pack_routed_experts(model, batch, routed_experts, contiguous=True)


def pack_r3_replay_mask(model: nn.Module, batch: PackedBatch) -> torch.Tensor:
    """Pack the causal R3 mask using DS4's contiguous CP token layout."""

    return _pack_r3_replay_mask(model, batch, contiguous=True)


def router_replay_roots(chunk: nn.Module) -> list[nn.Module]:
    """Return main decoder layers only; rollout R3 has no MTP layer axis."""

    model = getattr(chunk, "model", chunk)
    layers = getattr(model, "layers", None)
    if layers is None:
        return [chunk]
    return list(layers.values())


def _apply_mtp_config(model_cfg: DeepseekV4Config, impl_cfg: ImplConfig) -> None:
    override = impl_cfg.num_nextn_predict_layers
    if override is None:
        override = impl_cfg.mtp_num_layers
    if override is not None:
        if override < 0:
            raise ValueError(f"DeepSeek V4 MTP layer count must be >=0, got {override}.")
        model_cfg.num_nextn_predict_layers = int(override)
    if impl_cfg.mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError("mtp_enable=True but DeepSeek V4 config has no MTP layers.")
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
    else:
        model_cfg.num_nextn_predict_layers = 0


def _make_aux_loss_hook():
    """Per-step hook that syncs the MTP auxiliary-loss backward scale to the main
    loss scale (DP size / gradient accumulation), mirroring the sibling protocols
    (kimi_k2 / glm5 / qwen3_5 / qwen3_moe).

    DS4 only injects an MTP auxiliary loss: its MoE router is aux-loss-free
    (``SigmoidTopKRouter(..., compute_aux_loss=False)``) and its CSA indexer runs
    with ``sparse_loss=False``, so -- unlike GLM-5, which also scales the MoE-aux
    and DSA-indexer losses -- only ``MTPLossAutoScaler`` needs scaling here.
    Without this hook the injected MTP gradient keeps ``MTPLossAutoScaler``'s
    class-default scale of 1.0 and is mis-weighted relative to the main loss.
    """
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    def hook(scale: torch.Tensor) -> None:
        MTPLossAutoScaler.set_loss_scale(scale)

    return hook


def _optimizer_backend_name(optimizer: Any) -> str | None:
    if isinstance(optimizer, dict) or isinstance(optimizer, OptimizerConfig):
        return "dist_opt"
    return optimizer


def _configure_attention_backend(chunks: list[nn.Module], *, backend: str | None) -> None:
    backend_name = backend or "torch"
    for chunk in chunks:
        for module in chunk.modules():
            if hasattr(module, "attention_backend"):
                module.attention_backend = backend_name


def _iter_transformer_units(chunk: nn.Module) -> list[nn.Module]:
    # Native DS4 chunks are DeepseekV4Model instances themselves. Keep support
    # for wrapper-style chunks, but do not require a `.model` indirection or
    # recompute/offload silently applies to zero transformer layers.
    model = getattr(chunk, "model", chunk)
    layers = list(getattr(model, "layers", {}).values())
    mtp_layers = list(getattr(model, "mtp", []))
    return [*layers, *mtp_layers]


def _validate_parallel_scope(p: ParallelConfig) -> None:
    """DS4 CSA attention is not tensor-parallel-capable (documented TP=1 case).

    PP / VPP / EP / CP are inherited from the Kimi skeleton and work; only
    TP>1 / ETP>1 are unsupported.  Mirrors GLM-5's gate.
    """
    etp = 1 if p.etp is None else p.etp
    if p.tp > 1:
        raise NotImplementedError(
            "DeepSeek V4 native CSA attention does not support tensor parallelism; "
            f"got tp={p.tp}. Use tp=1 (PP/VPP/EP/CP are supported)."
        )
    if etp > 1:
        raise NotImplementedError(
            "DeepSeek V4 native CSA attention does not support expert tensor parallelism; "
            f"got etp={etp}. Use etp=1 (EP is supported)."
        )


def _ds4_chunk_factory(
    model_cfg: DeepseekV4Config,
    impl_cfg: ImplConfig,
    ps: ParallelState,
    vpp_chunk_id: int | None,
) -> nn.Module:
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model

    # ``prepare`` (below) has already applied MTP config / validated scope, so
    # ``model_cfg.num_nextn_predict_layers`` is authoritative here.
    mtp_enable = bool(impl_cfg.mtp_enable) and model_cfg.num_nextn_predict_layers > 0
    mtp_enable_train = mtp_enable and bool(impl_cfg.mtp_enable_train)
    vpp = None if impl_cfg.parallel.vpp == 1 else impl_cfg.parallel.vpp
    train_cfg = SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=vpp,
        fp8=False,
        use_deepep=impl_cfg.use_deepep,
    )
    return DeepseekV4Model(
        model_cfg,
        train_cfg,
        ps,
        vpp_chunk_id=vpp_chunk_id,
        use_deepep=impl_cfg.use_deepep,
        use_thd=impl_cfg.use_thd,
        hf_path=impl_cfg.hf_path,
        attention_backend_override=impl_cfg.attention_backend_override,
        mtp_enable=mtp_enable,
        mtp_enable_train=mtp_enable_train,
        mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
    )


def _ds4_prepare(model_cfg: DeepseekV4Config, impl_cfg: ImplConfig, ps: ParallelState) -> None:
    _validate_parallel_scope(impl_cfg.parallel)
    _apply_mtp_config(model_cfg, impl_cfg)


def _ds4_fsdp2_unit_modules() -> tuple[type[nn.Module], ...]:
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Layer

    return (DeepseekV4Layer,)


def build_model(model_cfg: DeepseekV4Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    # The shared assembly (VPP chunk build, recompute/offload, dist_opt/fsdp2
    # wiring, sharded-state-dict attach, ModelBundle packing) is absorbed by
    # ``compose_kernel.assemble``. What stays DS4-specific (irreducible
    # specialization, declared through the spec below):
    #   * chunk_factory: DS4-only train_cfg + model kwargs (CSA / hash-MoE /
    #     mHC 4-D hidden knobs) that no other model shares;
    #   * transformer_units: DS4 walks ``chunk.layers.values() + chunk.mtp``
    #     (ModuleDict + MTP list), not the ``chunk.layers`` ModuleList siblings
    #     use -- this is the single #114 unit enumeration;
    #   * prepare: TP/ETP=1 CSA scope gate + MTP-layer config;
    #   * post_chunk_hook: CSA/DSA attention-backend configuration (DS4 has no
    #     cross-entropy-fusion toggle, unlike the sibling models);
    #   * fsdp2_extra_kwargs: ``use_fp32_shards=False`` (DS4-only);
    #   * optimizer_backend_name: DS4 accepts an OptimizerConfig/dict as the
    #     optimizer and maps it to dist_opt.
    # HF weight load/export/save stay in checkpoint.py: they are pure weight-name
    # mapping + fp4/fp8 dequant + expert-index math, not composition assembly.
    spec = ModelSpec(
        name="deepseek_v4",
        chunk_factory=_ds4_chunk_factory,
        transformer_units=_iter_transformer_units,
        module_map=MODULE_MAP,
        forward_step=_forward_step,
        expert_classifier=is_expert_param,
        placement_fn=PLACEMENT_FN,
        fsdp2_unit_modules=_ds4_fsdp2_unit_modules,
        prepare=_ds4_prepare,
        post_chunk_hook=lambda chunks, impl: _configure_attention_backend(
            chunks, backend=impl.attention_backend_override
        ),
        pre_forward_hook_factory=_make_aux_loss_hook,
        fsdp2_extra_kwargs={"use_fp32_shards": False},
        optimizer_backend_name=_optimizer_backend_name,
    )
    return assemble(spec, model_cfg, impl_cfg)


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: DeepseekV4Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(
    chunks: list[nn.Module], model_cfg: DeepseekV4Config, ps: ParallelState, **kwargs
):
    yield from _export_hf_weights_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(
    chunks: list[nn.Module], path: str, model_cfg: DeepseekV4Config, ps: ParallelState, **kwargs
) -> None:
    _save_hf_weights_impl(chunks, path, model_cfg, ps, **kwargs)


def vocab_size(model_cfg: DeepseekV4Config) -> int | None:
    return model_cfg.vocab_size

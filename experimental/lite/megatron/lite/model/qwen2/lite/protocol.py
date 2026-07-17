# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense Qwen2 lite protocol.

This protocol intentionally exposes only the local-runtime slice needed by the
paper-alignment route: TP=1/PP=1 model construction, forward/backward, LoRA
freezing/stats, adapter lifecycle, OLoRA-tail init, and dense Qwen2 HF
checkpoint load/export. Distributed layouts are still follow-up work, so exact
paper reproduction remains blocked until real checkpoint and GPU/RL evidence is
recorded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.model.protocol_utils import add_loss_context_kwargs
from megatron.lite.model.qwen2.config import Qwen2Config
from megatron.lite.model.qwen2.lite.model import Qwen2ForCausalLM
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.modules.delta_mem import (
    DeltaMemConfig,
    apply_delta_mem_base_slice_init,
    normalize_delta_mem_config,
)
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    freeze_non_lora_params,
    normalize_lora_config,
    trainable_param_stats,
)
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

__all__ = [
    "ImplConfig",
    "build_model",
    "build_model_config",
    "export_hf_state_dict",
    "export_hf_weights",
    "export_lora_adapter_state",
    "initialize_lora_olora_tail",
    "load_lora_adapter",
    "load_lora_adapter_state",
    "save_lora_adapter",
    "load_hf_state_dict",
    "load_hf_weights",
    "save_hf_weights",
    "unpack_forward_output",
    "vocab_size",
]


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = None
    optimizer_config: OptimizerConfig | None = None
    deterministic: bool = True
    lora: LoraConfig | Mapping[str, Any] | None = None
    lora_init: bool | str | None = None
    delta_mem: DeltaMemConfig | Mapping[str, Any] | None = None


def build_model_config(source: str | Path | dict, **overrides) -> Qwen2Config:
    cfg = (
        Qwen2Config._from_hf_dict(source)
        if isinstance(source, dict)
        else Qwen2Config.from_hf(str(source))
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _resolve_lora_init(impl_cfg: ImplConfig) -> str | None:
    value = impl_cfg.lora_init
    if value is None and isinstance(impl_cfg.lora, Mapping):
        # Accept both spellings: "init" (LoraConfig field) and "init_lora_weights"
        # (HF PEFT field). Reading only one made the other silently skip the
        # protocol-level init hook (vacuous OLoRA arms in sweeps).
        value = impl_cfg.lora.get("init_lora_weights", impl_cfg.lora.get("init"))
    if value in (None, True, False):
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"Qwen2 LoRA init must be a string, boolean, or None, got {type(value)!r}."
        )
    normalized = value.strip().lower()
    if normalized in ("", "none", "null", "false", "true", "default"):
        return None
    if normalized in ("olora_tail", "olora"):
        return normalized
    raise NotImplementedError(
        f"Qwen2 LoRA init {value!r} is not implemented in MLite yet; "
        "currently supported: 'olora_tail' for tp=1."
    )


def _validate_parallel_scope(p: ParallelConfig) -> None:
    if p.tp != 1:
        raise NotImplementedError("Qwen2 dense lite runtime currently supports tp=1.")
    if p.pp != 1:
        raise NotImplementedError("Qwen2 dense lite runtime currently supports pp=1.")
    etp = 1 if p.etp is None else p.etp
    if etp != 1:
        raise NotImplementedError("Qwen2 dense lite runtime currently supports etp=1.")
    if p.cp != 1:
        raise NotImplementedError("Qwen2 dense lite runtime currently supports cp=1.")
    if p.vpp != 1:
        raise NotImplementedError("Qwen2 dense lite runtime currently supports vpp=1.")


def _parallel_state(p: ParallelConfig) -> ParallelState:
    return init_parallel(p) if dist.is_available() and dist.is_initialized() else ParallelState()


def _packed_to_padded(
    batch: PackedBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    seq_lens = batch.seq_lens.detach().cpu().tolist()
    if not seq_lens:
        raise ValueError("PackedBatch must contain at least one sequence.")
    max_len = max(int(length) for length in seq_lens)
    input_rows = []
    label_rows = []
    mask_rows = []
    position_rows = []
    offset = 0
    for length in seq_lens:
        length = int(length)
        ids = batch.input_ids[offset : offset + length]
        labels = batch.labels[offset : offset + length]
        positions = (
            None
            if batch.position_ids is None
            else batch.position_ids[offset : offset + length]
        )
        mask = (
            torch.ones_like(labels, dtype=torch.float32)
            if batch.loss_mask is None
            else batch.loss_mask[offset : offset + length].to(dtype=torch.float32)
        )
        offset += length
        # mlite batch contract (same as the THD path's roll_labels/roll_loss_mask):
        # ``labels`` are the RAW tokens and the protocol does the next-token shift per
        # sequence, rolling the loss mask with them. Without this the model scores
        # P(x_t | x_<=t) against x_t itself — real-weight CE ~15 instead of ~1.7 while
        # every hidden state matches the HF reference exactly.
        labels = torch.cat([labels[1:], labels.new_zeros(1)])
        mask = torch.cat([mask[1:], mask.new_zeros(1)])
        pad = max_len - length
        if pad:
            ids = torch.cat([ids, ids.new_zeros(pad)])
            labels = torch.cat([labels, labels.new_zeros(pad)])
            if positions is not None:
                positions = torch.cat([positions, positions.new_zeros(pad)])
            mask = torch.cat([mask, mask.new_zeros(pad)])
        input_rows.append(ids)
        label_rows.append(labels)
        mask_rows.append(mask)
        if positions is not None:
            position_rows.append(positions)
    padded_positions = torch.stack(position_rows) if position_rows else None
    return (
        torch.stack(input_rows),
        torch.stack(label_rows),
        torch.stack(mask_rows),
        padded_positions,
    )


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    input_ids, labels, loss_mask, position_ids = _packed_to_padded(batch)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask": loss_mask,
    }
    add_loss_context_kwargs(kwargs, include_return_log_probs=True)
    return model(**kwargs)


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output: torch.Tensor):
    """Reverse padded/packed per-token outputs to jagged nested form.

    verl's postprocess unbinds per-sequence rows from a nested tensor (the THD
    path's contract via unpack_thd_to_nested); returning a flat 1-D tensor here
    unbinds into zero-dim scalars and crashes nested-tensor construction.
    """
    del model
    seq_lens = [int(length) for length in batch.seq_lens.detach().cpu().tolist()]
    total_tokens = sum(seq_lens)
    if output.ndim == 1:
        if output.numel() != total_tokens:
            raise ValueError(
                f"Packed Qwen2 output has {output.numel()} tokens, expected {total_tokens}."
            )
        rows = list(torch.split(output, seq_lens))
    else:
        if output.shape[0] != len(seq_lens):
            raise ValueError(
                f"Qwen2 padded output batch dim {output.shape[0]} does not match "
                f"PackedBatch sequence count {len(seq_lens)}."
            )
        rows = [output[i, :length] for i, length in enumerate(seq_lens)]
    return torch.nested.as_nested_tensor([row.contiguous() for row in rows], layout=torch.jagged)


def _build_dist_opt_optimizer(
    chunks: list[nn.Module],
    model_cfg: Qwen2Config,
    impl_cfg: ImplConfig,
    ps: ParallelState,
):
    from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_training_optimizer

    return build_dist_opt_training_optimizer(
        chunks,
        model_cfg=model_cfg,
        impl_cfg=impl_cfg,
        ps=ps,
        model_name="qwen2",
        is_expert=None,
        deterministic=impl_cfg.deterministic,
    )


def build_model(model_cfg: Qwen2Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    _validate_parallel_scope(impl_cfg.parallel)
    lora_config = normalize_lora_config(impl_cfg.lora)
    lora_init = _resolve_lora_init(impl_cfg)
    if lora_init in ("olora_tail", "olora"):
        if not lora_config.enabled:
            raise ValueError("Qwen2 lora_init='olora_tail' requires enabled LoRA rank > 0.")
        if impl_cfg.parallel.tp != 1:
            raise NotImplementedError("Qwen2 lora_init='olora_tail' currently supports tp=1.")
        if impl_cfg.parallel.pp != 1:
            raise NotImplementedError("Qwen2 lora_init='olora_tail' currently supports pp=1.")
        etp = 1 if impl_cfg.parallel.etp is None else impl_cfg.parallel.etp
        if etp != 1:
            raise NotImplementedError("Qwen2 lora_init='olora_tail' currently supports etp=1.")
        if impl_cfg.parallel.cp != 1:
            raise NotImplementedError("Qwen2 lora_init='olora_tail' currently supports cp=1.")
    raw_delta_mem = impl_cfg.delta_mem
    if (
        raw_delta_mem is not None
        and not isinstance(raw_delta_mem, (DeltaMemConfig, dict))
        and isinstance(raw_delta_mem, Mapping)
    ):
        raw_delta_mem = dict(raw_delta_mem)  # OmegaConf DictConfig etc. (9b9da6d34 lesson)
    delta_mem_config = normalize_delta_mem_config(raw_delta_mem)
    ps = _parallel_state(impl_cfg.parallel)
    model = Qwen2ForCausalLM(
        model_cfg, ps, lora_config=lora_config, delta_mem_config=delta_mem_config
    )
    chunks = [model]
    if impl_cfg.optimizer == "dist_opt" and torch.cuda.is_available():
        model = model.to(torch.bfloat16).cuda()
        chunks[0] = model

    lora_stats = None
    if lora_config.enabled or delta_mem_config.enabled:
        freeze_stats = freeze_non_lora_params(model)
        trainable_stats = trainable_param_stats(model)
        lora_stats = {"chunks": [{**freeze_stats, **trainable_stats}]}

    if lora_init in ("olora_tail", "olora") and impl_cfg.optimizer == "dist_opt":
        raise ValueError(
            "Qwen2 lora_init='olora_tail' requires the fsdp2 optimizer (or none): "
            "dist_opt captures its master param buffer at build time, before weights "
            "load, so a post-load OLoRA init would desync master and model params."
        )
    needs_delta_mem_init = (
        delta_mem_config.enabled and delta_mem_config.output_init == "base_slice_fixed"
    )
    if needs_delta_mem_init and impl_cfg.optimizer == "dist_opt":
        raise ValueError(
            "Qwen2 delta_mem output_init='base_slice_fixed' requires the fsdp2 "
            "optimizer (or none): the post-load Δ-head init would desync dist_opt's "
            "master params, same failure mode as OLoRA-tail."
        )

    post_model_load_hook = None
    if (
        lora_init in ("olora_tail", "olora") or needs_delta_mem_init
    ) and impl_cfg.optimizer != "fsdp2":

        def _post_model_load_hook():
            extras: dict[str, Any] = {}
            if lora_init in ("olora_tail", "olora"):
                extras["lora_init_result"] = initialize_lora_olora_tail(
                    [model], model_cfg, ps, kind=lora_init
                )
            if needs_delta_mem_init:
                extras["delta_mem_init_result"] = apply_delta_mem_base_slice_init(model)
            return {"extras": extras}

        post_model_load_hook = _post_model_load_hook

    optimizer = None
    finalize_grads = None
    optimizer_backend = "none"
    if impl_cfg.optimizer == "dist_opt":
        optimizer, finalize_grads = _build_dist_opt_optimizer(chunks, model_cfg, impl_cfg, ps)
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        attach_model_sharded_state_dict(chunks, ps)
        register_training_hooks(chunks, optimizer)
        optimizer_backend = "dist_opt"
    elif impl_cfg.optimizer == "fsdp2":
        optimizer_backend = "fsdp2"
        if torch.cuda.is_available():
            model = model.to(torch.bfloat16).cuda()
            chunks[0] = model

        def _fsdp2_post_model_load_hook():
            from megatron.lite.model.qwen2.lite.model import Qwen2DecoderLayer
            from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

            extras: dict[str, Any] = {}
            # OLoRA-tail must precede the optimizer build so it captures the
            # OLoRA-initialized params (and the residual-shifted base weights).
            if lora_init in ("olora_tail", "olora"):
                extras["lora_init_result"] = initialize_lora_olora_tail([model], model_cfg, ps, kind=lora_init)
            # Same ordering constraint for the δ-mem base-slice Δ-head init.
            if needs_delta_mem_init:
                extras["delta_mem_init_result"] = apply_delta_mem_base_slice_init(model)
            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(Qwen2DecoderLayer,),
                    deterministic=impl_cfg.deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    leaf_module_names=(),
                ),
                "extras": extras,
            }

        post_model_load_hook = _fsdp2_post_model_load_hook
    elif impl_cfg.optimizer is None:
        optimizer_backend = "none"
    else:
        raise ValueError(f"Unknown qwen2 lite optimizer: {impl_cfg.optimizer!r}.")

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step,
        extras={
            "model_cfg": model_cfg,
            "optimizer_backend": optimizer_backend,
            "post_model_load_hook": post_model_load_hook,
            "lora_config": lora_config,
            "lora_init": lora_init,
            "lora_stats": lora_stats,
            "adapter_lifecycle_supported": True,
            "olora_tail_supported": True,
            "hf_checkpoint_supported": True,
        },
    )


def load_hf_state_dict(chunks, state, model_cfg: Qwen2Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.qwen2.lite.checkpoint import load_hf_state_dict as load_impl

    del kwargs
    if not chunks:
        raise ValueError("Qwen2 load_hf_state_dict requires at least one model chunk.")
    return load_impl(chunks[0], state, model_cfg, ps)


def export_hf_state_dict(chunks, model_cfg: Qwen2Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.qwen2.lite.checkpoint import export_hf_state_dict as export_impl

    return export_impl(chunks, model_cfg, ps, **kwargs)


def load_hf_weights(
    chunk: nn.Module, hf_path: str | Path, model_cfg: Qwen2Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    from megatron.lite.model.qwen2.lite.checkpoint import load_hf_weights as load_impl

    load_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(chunks, model_cfg: Qwen2Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.qwen2.lite.checkpoint import export_hf_weights as export_impl

    yield from export_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(chunks, path: str | Path, model_cfg: Qwen2Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.qwen2.lite.checkpoint import save_hf_weights as save_impl

    save_impl(chunks, path, model_cfg, ps, **kwargs)


def initialize_lora_olora_tail(
    chunks,
    model_cfg: Qwen2Config | None = None,
    ps: ParallelState | None = None,
    kind: str = "olora_tail",
    **_: object,
) -> dict[str, int]:
    """Orthogonal LoRA init for dense Qwen2 LoRA surfaces.

    ``kind="olora_tail"`` (arXiv:2606.02437 §4.1.2, minor-SVD subspace) or
    ``kind="olora"`` (arXiv:2406.01775, QR principal-leaning — the Fig 13
    collapse-contrast arm). Both use the primitive's PiSSA-style base residual,
    so each layer's output is unchanged at init. Dense module pairs by attribute
    convention: attention ``qkv``/``proj`` and MLP ``gate_up``/``down`` with their
    ``*_lora`` adapters. tp=1 only (enforced by the build-time parallel scope);
    PEFT adapter import/export follows separately.
    """
    del model_cfg, ps
    if kind not in ("olora_tail", "olora"):
        raise ValueError(f"Unknown orthogonal LoRA init kind {kind!r}.")
    initialized = 0
    for chunk in chunks:
        for module in chunk.modules():
            for base_attr, lora_attr in (
                ("qkv", "qkv_lora"),
                ("proj", "proj_lora"),
                ("gate_up", "gate_up_lora"),
                ("down", "down_lora"),
            ):
                base = getattr(module, base_attr, None)
                lora = getattr(module, lora_attr, None)
                if base is not None and lora is not None and hasattr(base, "weight"):
                    if kind == "olora_tail":
                        lora.olora_tail_init_(base.weight.data)
                    else:
                        lora.olora_init_(base.weight.data)
                    initialized += 1
    return {"olora_initialized": initialized}


def export_lora_adapter_state(chunks, model_cfg: Qwen2Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.qwen2.lite.lora_adapter import export_lora_adapter_state as impl

    return impl(chunks, model_cfg, ps, **kwargs)


def save_lora_adapter(chunks, model_cfg: Qwen2Config, ps: ParallelState, output_dir, **kwargs):
    from megatron.lite.model.qwen2.lite.lora_adapter import save_lora_adapter as impl

    return impl(chunks, model_cfg, ps, output_dir, **kwargs)


def load_lora_adapter_state(chunks, state, model_cfg: Qwen2Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.qwen2.lite.lora_adapter import load_lora_adapter_state as impl

    return impl(chunks, state, model_cfg, ps, **kwargs)


def load_lora_adapter(chunks, adapter_dir, model_cfg: Qwen2Config, ps: ParallelState, **kwargs):
    from megatron.lite.model.qwen2.lite.lora_adapter import load_lora_adapter as impl

    return impl(chunks, adapter_dir, model_cfg, ps, **kwargs)


def vocab_size(model_cfg) -> int | None:
    return getattr(model_cfg, "vocab_size", None)

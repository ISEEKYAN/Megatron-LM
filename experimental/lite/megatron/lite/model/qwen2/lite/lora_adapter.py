# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF-PEFT adapter import/export for the dense Qwen2 lite runtime (tp=1).

Thin qwen3_moe-pattern implementation. The model fuses attention QKV as a flat
``cat([q; k; v])`` and the MLP as ``cat([gate; up])`` (see checkpoint.py), so a
fused-surface LoRA maps to PEFT's per-projection keys by sharing ``lora_A`` and
row-splitting ``lora_B``; import requires the shared ``lora_A`` copies to agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.qwen2.config import Qwen2Config
from megatron.lite.primitive.modules.lora import LoraConfig
from megatron.lite.primitive.parallel import ParallelState

_PEFT_PREFIX = "base_model.model.model"
_META_FORMAT = "megatron.lite_qwen2_lora_peft_v1"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"
_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_META = "megatron.lite_adapter_meta.json"


def _validate_scope(ps: ParallelState) -> None:
    if getattr(ps, "tp_size", 1) != 1 or getattr(ps, "pp_size", 1) != 1:
        raise NotImplementedError("Qwen2 LoRA adapter import/export supports tp=1/pp=1 only.")


def _unwrap(module: nn.Module) -> nn.Module:
    while hasattr(module, "module"):
        module = module.module
    return module


def _layers(chunks) -> list[tuple[int, nn.Module]]:
    out = []
    for chunk in chunks:
        model = _unwrap(chunk)
        base = model.model if hasattr(model, "model") else model
        out.extend(enumerate(base.layers))
    return out


def _key(layer_idx: int, scope: str, proj: str, part: str) -> str:
    return f"{_PEFT_PREFIX}.layers.{layer_idx}.{scope}.{proj}.lora_{part}.weight"


def _fused_surfaces(cfg: Qwen2Config, layer: nn.Module):
    """(lora_module, scope, [(peft_proj, rows)]) for each fused/plain LoRA surface."""
    q = cfg.num_attention_heads * cfg.head_dim
    kv = cfg.num_key_value_heads * cfg.head_dim
    return (
        (layer.self_attn.qkv_lora, "self_attn", [("q_proj", q), ("k_proj", kv), ("v_proj", kv)]),
        (layer.self_attn.proj_lora, "self_attn", [("o_proj", cfg.hidden_size)]),
        (layer.mlp.gate_up_lora, "mlp", [("gate_proj", cfg.intermediate_size), ("up_proj", cfg.intermediate_size)]),
        (layer.mlp.down_lora, "mlp", [("down_proj", cfg.hidden_size)]),
    )


def export_lora_adapter_state(
    chunks, model_cfg: Qwen2Config, ps: ParallelState, **_: Any
) -> dict[str, torch.Tensor]:
    _validate_scope(ps)
    state: dict[str, torch.Tensor] = {}
    for layer_idx, layer in _layers(chunks):
        for lora, scope, projs in _fused_surfaces(model_cfg, layer):
            if lora is None:
                continue
            a = lora.lora_a.detach().cpu()
            b = lora.lora_b.detach().cpu()
            offset = 0
            for proj, rows in projs:
                state[_key(layer_idx, scope, proj, "A")] = a.clone()
                state[_key(layer_idx, scope, proj, "B")] = b[offset : offset + rows].clone()
                offset += rows
            if offset != b.shape[0]:
                raise ValueError(
                    f"layer {layer_idx} {scope} lora_b has {b.shape[0]} rows, "
                    f"fused split consumed {offset}."
                )
    return state


def load_lora_adapter_state(
    chunks, state: dict[str, torch.Tensor], model_cfg: Qwen2Config, ps: ParallelState, **_: Any
) -> dict[str, int]:
    _validate_scope(ps)
    consumed: set[str] = set()
    loaded = 0
    for layer_idx, layer in _layers(chunks):
        for lora, scope, projs in _fused_surfaces(model_cfg, layer):
            if lora is None:
                continue
            a_parts, b_parts = [], []
            for proj, rows in projs:
                a_key = _key(layer_idx, scope, proj, "A")
                b_key = _key(layer_idx, scope, proj, "B")
                for key in (a_key, b_key):
                    if key not in state:
                        raise KeyError(f"adapter state is missing {key}")
                a_parts.append((proj, state[a_key]))
                b = state[b_key]
                if b.shape[0] != rows:
                    raise ValueError(
                        f"{b_key} has {b.shape[0]} rows, expected {rows} for {proj}."
                    )
                b_parts.append(b)
                consumed.update((a_key, b_key))
            ref_proj, ref_a = a_parts[0]
            for proj, a in a_parts[1:]:
                if not torch.equal(a.to(ref_a.dtype), ref_a):
                    raise ValueError(
                        f"layer {layer_idx} {scope}: fused surface requires identical "
                        f"lora_A across projections, but {proj} differs from {ref_proj}."
                    )
            with torch.no_grad():
                lora.lora_a.copy_(ref_a.to(lora.lora_a.dtype, copy=False))
                lora.lora_b.copy_(torch.cat(b_parts, dim=0).to(lora.lora_b.dtype, copy=False))
            loaded += 1
    unexpected = sorted(set(state) - consumed)
    if unexpected:
        raise ValueError(f"adapter state has {len(unexpected)} unexpected keys: {unexpected[:4]}...")
    return {"loaded_lora_modules": loaded}


def _peft_target_modules(lora_config: LoraConfig) -> list[str]:
    expand = {
        "linear_qkv": ("q_proj", "k_proj", "v_proj"),
        "linear_proj": ("o_proj",),
        "linear_fc1": ("gate_proj", "up_proj"),
        "linear_fc2": ("down_proj",),
    }
    out: set[str] = set()
    for target in lora_config.targets():
        out.update(expand.get(target, (target,)))
    return sorted(out)


def save_lora_adapter(
    chunks,
    model_cfg: Qwen2Config,
    ps: ParallelState,
    output_dir: str | Path,
    *,
    lora_config: LoraConfig,
    metadata: dict[str, Any] | None = None,
    **_: Any,
) -> Path:
    from safetensors.torch import save_file

    _validate_scope(ps)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = export_lora_adapter_state(chunks, model_cfg, ps)
    save_file(state, str(out / _ADAPTER_WEIGHTS))
    (out / _ADAPTER_CONFIG).write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": lora_config.rank,
                "lora_alpha": lora_config.rank if lora_config.alpha is None else lora_config.alpha,
                "lora_dropout": lora_config.dropout,
                "use_rslora": lora_config.use_rslora,
                "init_lora_weights": True if lora_config.init == "default" else lora_config.init,
                "target_modules": _peft_target_modules(lora_config),
                "bias": "none",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (out / _ADAPTER_META).write_text(
        json.dumps(
            {
                "format": _META_FORMAT,
                "lora": {
                    "rank": lora_config.rank,
                    "alpha": lora_config.rank if lora_config.alpha is None else lora_config.alpha,
                    "dropout": lora_config.dropout,
                    "use_rslora": lora_config.use_rslora,
                    "scaling_convention": (
                        "alpha_over_sqrt_rank" if lora_config.use_rslora else "alpha_over_rank"
                    ),
                    "scale": lora_config.scale,
                    "init": lora_config.init,
                },
                "parallel": {"tp": ps.tp_size, "pp": ps.pp_size},
                "model": {
                    "hidden_size": model_cfg.hidden_size,
                    "num_hidden_layers": model_cfg.num_hidden_layers,
                    "num_attention_heads": model_cfg.num_attention_heads,
                    "num_key_value_heads": model_cfg.num_key_value_heads,
                    "head_dim": model_cfg.head_dim,
                    "intermediate_size": model_cfg.intermediate_size,
                },
                **({"metadata": metadata} if metadata else {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return out


def load_lora_adapter(
    chunks,
    adapter_dir: str | Path,
    model_cfg: Qwen2Config,
    ps: ParallelState,
    *,
    lora_config: LoraConfig | None = None,
    **_: Any,
) -> dict[str, int]:
    from safetensors import safe_open

    adapter = Path(adapter_dir)
    weights = adapter / _ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"no {_ADAPTER_WEIGHTS} under {adapter}")
    if lora_config is not None:
        cfg = json.loads((adapter / _ADAPTER_CONFIG).read_text())
        if int(cfg["r"]) != lora_config.rank:
            raise ValueError(
                f"adapter rank {cfg['r']} != model LoRA rank {lora_config.rank}."
            )
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(weights), framework="pt") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return load_lora_adapter_state(chunks, state, model_cfg, ps)


__all__ = [
    "export_lora_adapter_state",
    "load_lora_adapter",
    "load_lora_adapter_state",
    "save_lora_adapter",
]

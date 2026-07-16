# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dense Qwen2 lite native <-> HuggingFace checkpoint mapping.

This is the exact Fig.14 model-family checkpoint slice for
DeepSeek-R1-Distill-Qwen-1.5B.  It is deliberately conservative: TP/PP/CP/ETP
are all fixed at 1, and the mapping only covers the dense Qwen2 parameters used
by the minimal local runtime.  Native attention and MLP weights are fused, so
HF ``q_proj/k_proj/v_proj`` and ``gate_proj/up_proj`` tensors are packed on load
and split back on export.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.model.qwen2.config import Qwen2Config
from megatron.lite.primitive.ckpt.hf_weights import (
    SafeTensorReader,
    _cast_export_tensor,
    _resolve_export_dtype,
    save_safetensors,
    unwrap_model,
)
from megatron.lite.primitive.parallel import ParallelState


def _validate_parallel_scope(ps: ParallelState) -> None:
    if ps.tp_size != 1:
        raise NotImplementedError("Dense Qwen2 HF checkpoint load/export currently supports tp=1.")
    if ps.pp_size != 1:
        raise NotImplementedError("Dense Qwen2 HF checkpoint load/export currently supports pp=1.")
    if ps.cp_size != 1:
        raise NotImplementedError("Dense Qwen2 HF checkpoint load/export currently supports cp=1.")
    if ps.etp_size != 1:
        raise NotImplementedError("Dense Qwen2 HF checkpoint load/export currently supports etp=1.")
    if ps.ep_size != 1:
        raise NotImplementedError("Dense Qwen2 HF checkpoint load/export currently supports ep=1.")


def _as_model(chunks_or_model: nn.Module | list[nn.Module] | tuple[nn.Module, ...]) -> nn.Module:
    if isinstance(chunks_or_model, nn.Module):
        return chunks_or_model
    chunks = list(chunks_or_model)
    if len(chunks) != 1:
        raise NotImplementedError("Dense Qwen2 HF checkpoint mapping currently supports one chunk.")
    return chunks[0]


def _expected_hf_keys(config: Qwen2Config) -> list[str]:
    keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        keys.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
        )
        if config.attention_bias:
            keys.extend(
                [
                    f"{prefix}.self_attn.q_proj.bias",
                    f"{prefix}.self_attn.k_proj.bias",
                    f"{prefix}.self_attn.v_proj.bias",
                ]
            )
        keys.extend(
            [
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    return keys


def _get_tensor(hf_state: Mapping[str, Any], name: str) -> torch.Tensor:
    if name not in hf_state:
        raise KeyError(f"Missing dense Qwen2 HF checkpoint tensor: {name}")
    tensor = hf_state[name]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Dense Qwen2 HF checkpoint tensor {name!r} is not a torch.Tensor.")
    return tensor


def _copy_param(param: torch.Tensor, tensor: torch.Tensor, *, name: str) -> None:
    if tuple(param.shape) != tuple(tensor.shape):
        raise ValueError(
            f"Dense Qwen2 checkpoint tensor {name!r} has shape {tuple(tensor.shape)}, "
            f"expected {tuple(param.shape)}."
        )
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def _cat_qkv(hf_state: Mapping[str, Any], prefix: str, config: Qwen2Config) -> torch.Tensor:
    q = _get_tensor(hf_state, f"{prefix}.q_proj.weight")
    k = _get_tensor(hf_state, f"{prefix}.k_proj.weight")
    v = _get_tensor(hf_state, f"{prefix}.v_proj.weight")
    expected_shapes = {
        "q_proj": (config.num_attention_heads * config.head_dim, config.hidden_size),
        "k_proj": (config.num_key_value_heads * config.head_dim, config.hidden_size),
        "v_proj": (config.num_key_value_heads * config.head_dim, config.hidden_size),
    }
    for label, tensor in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
        if tuple(tensor.shape) != expected_shapes[label]:
            raise ValueError(
                f"Dense Qwen2 HF tensor {prefix}.{label}.weight has shape "
                f"{tuple(tensor.shape)}, expected {expected_shapes[label]}."
            )
    return torch.cat([q, k, v], dim=0).contiguous()


def _cat_qkv_bias(hf_state: Mapping[str, Any], prefix: str, config: Qwen2Config) -> torch.Tensor:
    q = _get_tensor(hf_state, f"{prefix}.q_proj.bias")
    k = _get_tensor(hf_state, f"{prefix}.k_proj.bias")
    v = _get_tensor(hf_state, f"{prefix}.v_proj.bias")
    expected_shapes = {
        "q_proj": (config.num_attention_heads * config.head_dim,),
        "k_proj": (config.num_key_value_heads * config.head_dim,),
        "v_proj": (config.num_key_value_heads * config.head_dim,),
    }
    for label, tensor in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
        if tuple(tensor.shape) != expected_shapes[label]:
            raise ValueError(
                f"Dense Qwen2 HF tensor {prefix}.{label}.bias has shape "
                f"{tuple(tensor.shape)}, expected {expected_shapes[label]}."
            )
    return torch.cat([q, k, v], dim=0).contiguous()


def _cat_gate_up(hf_state: Mapping[str, Any], prefix: str, config: Qwen2Config) -> torch.Tensor:
    gate = _get_tensor(hf_state, f"{prefix}.gate_proj.weight")
    up = _get_tensor(hf_state, f"{prefix}.up_proj.weight")
    expected = (config.intermediate_size, config.hidden_size)
    if tuple(gate.shape) != expected:
        raise ValueError(
            f"Dense Qwen2 HF tensor {prefix}.gate_proj.weight has shape {tuple(gate.shape)}, "
            f"expected {expected}."
        )
    if tuple(up.shape) != expected:
        raise ValueError(
            f"Dense Qwen2 HF tensor {prefix}.up_proj.weight has shape {tuple(up.shape)}, "
            f"expected {expected}."
        )
    return torch.cat([gate, up], dim=0).contiguous()


def load_hf_state_dict(
    model: nn.Module,
    hf_state: Mapping[str, Any],
    config: Qwen2Config,
    ps: ParallelState,
) -> dict[str, Any]:
    """Load an in-memory HF Qwen2 state dict into the native fused runtime."""

    _validate_parallel_scope(ps)
    base_model = unwrap_model(model)
    qwen_model = getattr(base_model, "model", None)
    if qwen_model is None or not hasattr(qwen_model, "layers"):
        raise TypeError("Dense Qwen2 checkpoint loader expects Qwen2ForCausalLM-like model.")
    if len(qwen_model.layers) != config.num_hidden_layers:
        raise ValueError(
            f"Dense Qwen2 model has {len(qwen_model.layers)} layers, "
            f"but config expects {config.num_hidden_layers}."
        )

    loaded_native = 0
    loaded_hf_keys: set[str] = set()

    embed = _get_tensor(hf_state, "model.embed_tokens.weight")
    _copy_param(qwen_model.embed_tokens.weight, embed, name="model.embed_tokens.weight")
    loaded_native += 1
    loaded_hf_keys.add("model.embed_tokens.weight")

    norm = _get_tensor(hf_state, "model.norm.weight")
    _copy_param(qwen_model.norm.weight, norm, name="model.norm.weight")
    loaded_native += 1
    loaded_hf_keys.add("model.norm.weight")

    lm_head = _get_tensor(hf_state, "lm_head.weight")
    _copy_param(base_model.lm_head.weight, lm_head, name="lm_head.weight")
    loaded_native += 1
    loaded_hf_keys.add("lm_head.weight")

    for layer_idx, layer in enumerate(qwen_model.layers):
        hf_layer = f"model.layers.{layer_idx}"
        _copy_param(
            layer.input_layernorm.weight,
            _get_tensor(hf_state, f"{hf_layer}.input_layernorm.weight"),
            name=f"{hf_layer}.input_layernorm.weight",
        )
        loaded_native += 1
        loaded_hf_keys.add(f"{hf_layer}.input_layernorm.weight")

        qkv = _cat_qkv(hf_state, f"{hf_layer}.self_attn", config)
        _copy_param(layer.self_attn.qkv.weight, qkv, name=f"{hf_layer}.self_attn.qkv[fused]")
        loaded_native += 1
        loaded_hf_keys.update(
            {
                f"{hf_layer}.self_attn.q_proj.weight",
                f"{hf_layer}.self_attn.k_proj.weight",
                f"{hf_layer}.self_attn.v_proj.weight",
            }
        )
        if layer.self_attn.qkv.bias is not None:
            qkv_bias = _cat_qkv_bias(hf_state, f"{hf_layer}.self_attn", config)
            _copy_param(
                layer.self_attn.qkv.bias,
                qkv_bias,
                name=f"{hf_layer}.self_attn.qkv.bias[fused]",
            )
            loaded_native += 1
            loaded_hf_keys.update(
                {
                    f"{hf_layer}.self_attn.q_proj.bias",
                    f"{hf_layer}.self_attn.k_proj.bias",
                    f"{hf_layer}.self_attn.v_proj.bias",
                }
            )

        _copy_param(
            layer.self_attn.proj.weight,
            _get_tensor(hf_state, f"{hf_layer}.self_attn.o_proj.weight"),
            name=f"{hf_layer}.self_attn.o_proj.weight",
        )
        loaded_native += 1
        loaded_hf_keys.add(f"{hf_layer}.self_attn.o_proj.weight")

        _copy_param(
            layer.post_attention_layernorm.weight,
            _get_tensor(hf_state, f"{hf_layer}.post_attention_layernorm.weight"),
            name=f"{hf_layer}.post_attention_layernorm.weight",
        )
        loaded_native += 1
        loaded_hf_keys.add(f"{hf_layer}.post_attention_layernorm.weight")

        gate_up = _cat_gate_up(hf_state, f"{hf_layer}.mlp", config)
        _copy_param(layer.mlp.gate_up.weight, gate_up, name=f"{hf_layer}.mlp.gate_up[fused]")
        loaded_native += 1
        loaded_hf_keys.update(
            {
                f"{hf_layer}.mlp.gate_proj.weight",
                f"{hf_layer}.mlp.up_proj.weight",
            }
        )

        _copy_param(
            layer.mlp.down.weight,
            _get_tensor(hf_state, f"{hf_layer}.mlp.down_proj.weight"),
            name=f"{hf_layer}.mlp.down_proj.weight",
        )
        loaded_native += 1
        loaded_hf_keys.add(f"{hf_layer}.mlp.down_proj.weight")

    return {
        "loaded_native_tensors": loaded_native,
        "loaded_hf_tensors": len(loaded_hf_keys),
        "missing_keys": [],
    }


def _full_local(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize fsdp2 DTensor shards to a full tensor (symmetric collective)."""
    if hasattr(tensor, "full_tensor"):
        return tensor.full_tensor()
    return tensor


def _export_tensor(
    tensor: torch.Tensor,
    export_dtype: torch.dtype | None,
) -> torch.Tensor:
    return _cast_export_tensor(_full_local(tensor).detach().cpu().contiguous(), export_dtype)


def _iter_hf_state_tensors(
    model: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
    config: Qwen2Config,
    ps: ParallelState,
    *,
    rank0_only: bool = False,
    export_dtype: str | torch.dtype | None = None,
    merge_lora: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    _validate_parallel_scope(ps)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank0_only and rank != 0:
        return

    base_model = unwrap_model(_as_model(model))
    qwen_model = getattr(base_model, "model", None)
    if qwen_model is None or not hasattr(qwen_model, "layers"):
        raise TypeError("Dense Qwen2 checkpoint exporter expects Qwen2ForCausalLM-like model.")
    if len(qwen_model.layers) != config.num_hidden_layers:
        raise ValueError(
            f"Dense Qwen2 model has {len(qwen_model.layers)} layers, "
            f"but config expects {config.num_hidden_layers}."
        )

    def _surface(module: nn.Module, lora) -> torch.Tensor:
        # rollout weight sync must see the CURRENT policy: base + adapter delta.
        weight = _full_local(module.weight)
        if merge_lora and lora is not None:
            weight = weight + lora.materialized_delta_weight().to(
                dtype=weight.dtype, device=weight.device
            )
        return weight

    resolved_dtype = _resolve_export_dtype(export_dtype)
    yield "model.embed_tokens.weight", _export_tensor(
        qwen_model.embed_tokens.weight, resolved_dtype
    )
    yield "model.norm.weight", _export_tensor(qwen_model.norm.weight, resolved_dtype)
    yield "lm_head.weight", _export_tensor(base_model.lm_head.weight, resolved_dtype)

    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    for layer_idx, layer in enumerate(qwen_model.layers):
        hf_layer = f"model.layers.{layer_idx}"
        yield f"{hf_layer}.input_layernorm.weight", _export_tensor(
            layer.input_layernorm.weight, resolved_dtype
        )

        qkv_w = _surface(layer.self_attn.qkv, layer.self_attn.qkv_lora)
        q, k, v = torch.split(qkv_w, [q_size, kv_size, kv_size], dim=0)
        yield f"{hf_layer}.self_attn.q_proj.weight", _export_tensor(q, resolved_dtype)
        yield f"{hf_layer}.self_attn.k_proj.weight", _export_tensor(k, resolved_dtype)
        yield f"{hf_layer}.self_attn.v_proj.weight", _export_tensor(v, resolved_dtype)
        if layer.self_attn.qkv.bias is not None:
            q_bias, k_bias, v_bias = torch.split(
                layer.self_attn.qkv.bias,
                [q_size, kv_size, kv_size],
                dim=0,
            )
            yield f"{hf_layer}.self_attn.q_proj.bias", _export_tensor(q_bias, resolved_dtype)
            yield f"{hf_layer}.self_attn.k_proj.bias", _export_tensor(k_bias, resolved_dtype)
            yield f"{hf_layer}.self_attn.v_proj.bias", _export_tensor(v_bias, resolved_dtype)
        yield f"{hf_layer}.self_attn.o_proj.weight", _export_tensor(
            _surface(layer.self_attn.proj, layer.self_attn.proj_lora), resolved_dtype
        )

        yield f"{hf_layer}.post_attention_layernorm.weight", _export_tensor(
            layer.post_attention_layernorm.weight, resolved_dtype
        )

        gate, up = torch.split(
            _surface(layer.mlp.gate_up, layer.mlp.gate_up_lora),
            [config.intermediate_size, config.intermediate_size],
            dim=0,
        )
        yield f"{hf_layer}.mlp.gate_proj.weight", _export_tensor(gate, resolved_dtype)
        yield f"{hf_layer}.mlp.up_proj.weight", _export_tensor(up, resolved_dtype)
        yield f"{hf_layer}.mlp.down_proj.weight", _export_tensor(
            _surface(layer.mlp.down, layer.mlp.down_lora), resolved_dtype
        )


def export_hf_state_dict(
    model: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
    config: Qwen2Config,
    ps: ParallelState,
    *,
    rank0_only: bool = False,
    export_dtype: str | torch.dtype | None = None,
    merge_lora: bool = False,
) -> dict[str, torch.Tensor]:
    """Export native fused Qwen2 tensors to HF Qwen2 state-dict names."""

    return dict(
        _iter_hf_state_tensors(
            model,
            config,
            ps,
            rank0_only=rank0_only,
            export_dtype=export_dtype,
            merge_lora=merge_lora,
        )
    )


def _read_hf_state_dict(path: str | Path, config: Qwen2Config) -> dict[str, torch.Tensor]:
    reader = SafeTensorReader(str(path))
    return {name: reader.get_tensor(name) for name in _expected_hf_keys(config)}


class _LazySafeTensorMapping(Mapping[str, torch.Tensor]):
    def __init__(self, path: str | Path, config: Qwen2Config):
        self.reader = SafeTensorReader(str(path))
        self.keys_tuple = tuple(_expected_hf_keys(config))

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self:
            raise KeyError(key)
        return self.reader.get_tensor(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys_tuple)

    def __len__(self) -> int:
        return len(self.keys_tuple)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.keys_tuple


def load_hf_weights(model: nn.Module, path: str | Path, config: Qwen2Config, ps: ParallelState) -> None:
    """Load an HF safetensors directory into the dense Qwen2 native runtime."""

    if not path:
        return
    load_hf_state_dict(model, _LazySafeTensorMapping(path, config), config, ps)


def export_hf_weights(
    model: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
    config: Qwen2Config,
    ps: ParallelState,
    **kwargs,
):
    yield from _iter_hf_state_tensors(model, config, ps, **kwargs)


def save_hf_weights(
    model: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
    path: str | Path,
    config: Qwen2Config,
    ps: ParallelState,
    **kwargs,
) -> None:
    out = export_hf_state_dict(model, config, ps, rank0_only=True, **kwargs)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0 and out:
        save_safetensors(out, str(path))
    if dist.is_initialized():
        dist.barrier()


__all__ = [
    "export_hf_state_dict",
    "export_hf_weights",
    "load_hf_state_dict",
    "load_hf_weights",
    "save_hf_weights",
]

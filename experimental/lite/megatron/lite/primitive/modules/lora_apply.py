# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Post-build LoRA applicator — mirrors ``apply_qat_to_chunks`` opt-in surface."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.primitive.modules.lora import (
    LinearLoRA,
    LoraSpec,
    SharedGroupedLinearLoRA,
    freeze_non_lora_params,
    normalize_lora_spec,
    trainable_param_stats,
)


class LoRAWrappedLinear(nn.Module):
    """Wrap a linear surface as ``base(x) + adapter(adapter_x)``.

    Unknown attributes (``weight``, ``linear``, ``quant_method``, TP flags, …)
    delegate to ``base`` so checkpoint / QAT parametrization paths stay valid.
    """

    def __init__(
        self,
        base: nn.Module,
        adapter: nn.Module,
        *,
        adapter_input_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self._adapter_input_fn = adapter_input_fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = self.base(x, *args, **kwargs)
        adapter_x = self._adapter_input_fn(x) if self._adapter_input_fn is not None else x
        return out + self.adapter(adapter_x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


class LoRAWrappedGroupedLinear(nn.Module):
    """Grouped-linear wrapper: ``base(x, splits) + adapter(x, splits)``."""

    def __init__(self, base: nn.Module, adapter: SharedGroupedLinearLoRA):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, x: torch.Tensor, splits: list[int], *args, **kwargs) -> torch.Tensor:
        out = self.base(x, splits, *args, **kwargs)
        return out + self.adapter(x, splits)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


def _qkv_adapter_input_fn(base_qkv: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    linear = base_qkv.linear
    if not hasattr(linear, "layer_norm_weight"):
        return lambda x: x
    eps = float(getattr(linear, "eps", 1e-6))
    zero_centered = bool(getattr(linear, "zero_centered_gamma", False))

    def _fn(x: torch.Tensor) -> torch.Tensor:
        weight = linear.layer_norm_weight
        if zero_centered:
            weight = weight + 1
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(variance + eps)
        return (x_norm * weight.float()).to(x.dtype)

    return _fn


def _attach_gqa_qkv(attn, spec: LoraSpec) -> bool:
    if not spec.targets_module("linear_qkv"):
        return False
    if isinstance(attn.qkv, LoRAWrappedLinear):
        return False
    ps = attn.ps
    adapter = LinearLoRA(
        attn.qkv.local_in,
        attn.qkv.local_out,
        spec.rank,
        alpha=spec.alpha,
        dropout=spec.dropout,
        use_rslora=spec.use_rslora,
        sequence_parallel_input=attn.qkv.use_sp,
        tp_group=ps.tp_group,
        rank_partition_size=ps.tp_size,
        rank_partitioned_a=ps.tp_size > 1,
        a_tensor_model_parallel=ps.tp_size > 1,
        b_tensor_model_parallel=ps.tp_size > 1,
    )
    attn.qkv = LoRAWrappedLinear(
        attn.qkv, adapter, adapter_input_fn=_qkv_adapter_input_fn(attn.qkv)
    )
    return True


def _attach_gqa_proj(attn, spec: LoraSpec) -> bool:
    if not spec.targets_module("linear_proj"):
        return False
    if isinstance(attn.proj, LoRAWrappedLinear):
        return False
    ps = attn.ps
    adapter = LinearLoRA(
        attn.proj.local_in,
        attn.proj.local_out,
        spec.rank,
        alpha=spec.alpha,
        dropout=spec.dropout,
        use_rslora=spec.use_rslora,
        tp_group=ps.tp_group,
        tp_rank=ps.tp_rank,
        sequence_parallel_scatter_output=attn.proj.use_sp,
        input_parallel_reduce=ps.tp_size > 1,
        output_partition_size=ps.tp_size,
        output_partitioned_b=ps.tp_size > 1,
        a_tensor_model_parallel=ps.tp_size > 1,
        b_tensor_model_parallel=ps.tp_size > 1,
    )
    attn.proj = LoRAWrappedLinear(attn.proj, adapter)
    return True


def _attach_expert_fc(
    experts, attr: str, target: str, spec: LoraSpec, *, in_features: int, out_features: int
) -> bool:
    if not spec.targets_module(target):
        return False
    base = getattr(experts, attr)
    if isinstance(base, LoRAWrappedGroupedLinear):
        return False
    adapter = SharedGroupedLinearLoRA(
        experts.num_local_experts,
        in_features,
        out_features,
        spec.rank,
        alpha=spec.alpha,
        dropout=spec.dropout,
        use_rslora=spec.use_rslora,
    )
    setattr(experts, attr, LoRAWrappedGroupedLinear(base, adapter))
    return True


def _attach_swiglu_mlp(mlp, spec: LoraSpec) -> int:
    attached = 0
    hidden = mlp.gate_up.in_features
    intermediate = mlp.down.in_features
    if spec.targets_module("linear_fc1") and not isinstance(mlp.gate_up, LoRAWrappedLinear):
        adapter = LinearLoRA(
            hidden,
            mlp.gate_up.out_features,
            spec.rank,
            alpha=spec.alpha,
            dropout=spec.dropout,
            use_rslora=spec.use_rslora,
        )
        mlp.gate_up = LoRAWrappedLinear(mlp.gate_up, adapter)
        attached += 1
    if spec.targets_module("linear_fc2") and not isinstance(mlp.down, LoRAWrappedLinear):
        adapter = LinearLoRA(
            intermediate,
            mlp.down.out_features,
            spec.rank,
            alpha=spec.alpha,
            dropout=spec.dropout,
            use_rslora=spec.use_rslora,
        )
        mlp.down = LoRAWrappedLinear(mlp.down, adapter)
        attached += 1
    return attached


def get_linear_lora_adapter(module: nn.Module, attr: str) -> LinearLoRA | None:
    child = getattr(module, attr, None)
    if isinstance(child, LoRAWrappedLinear):
        return child.adapter
    return None


def get_grouped_lora_adapter(module: nn.Module, attr: str) -> SharedGroupedLinearLoRA | None:
    child = getattr(module, attr, None)
    if isinstance(child, LoRAWrappedGroupedLinear):
        return child.adapter
    return None


def iter_lora_adapter_modules(module: nn.Module):
    """Yield native LoRA adapter modules (for PEFT export / OLoRA init)."""
    for child in module.modules():
        if isinstance(child, LoRAWrappedLinear):
            yield child.adapter
        elif isinstance(child, LoRAWrappedGroupedLinear):
            yield child.adapter
        elif isinstance(child, (LinearLoRA, SharedGroupedLinearLoRA)):
            yield child


def apply_lora_to_chunks(
    chunks: Sequence[nn.Module],
    spec: LoraSpec | dict[str, Any] | None,
    *,
    ps=None,
) -> dict[str, int]:
    """Post-build batch attach LoRA wrappers + freeze base weights.

    Must run after chunk build / HF load and before optimizer / DDP wrap.
    """
    spec = normalize_lora_spec(spec)
    stats = {
        "attached_modules": 0,
        "skipped_existing": 0,
        "skipped_ignored": 0,
        "frozen_tensors": 0,
        "trainable_tensors": 0,
    }
    if not spec.enabled or spec.rank <= 0:
        return stats

    from megatron.lite.primitive.modules.experts import Experts
    from megatron.lite.primitive.modules.gqa import GQAttention
    from megatron.lite.primitive.modules.mlp import SwiGLUMLP

    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, GQAttention):
                if _attach_gqa_qkv(module, spec):
                    stats["attached_modules"] += 1
                if _attach_gqa_proj(module, spec):
                    stats["attached_modules"] += 1
            elif isinstance(module, Experts):
                config_hidden = module.fc1.in_features
                fc1_out = module.fc1.out_features
                fc2_in = module.fc2.in_features
                fc2_out = module.fc2.out_features
                if _attach_expert_fc(
                    module, "fc1", "linear_fc1", spec, in_features=config_hidden, out_features=fc1_out
                ):
                    stats["attached_modules"] += 1
                if _attach_expert_fc(
                    module, "fc2", "linear_fc2", spec, in_features=fc2_in, out_features=fc2_out
                ):
                    stats["attached_modules"] += 1
            elif isinstance(module, SwiGLUMLP):
                stats["attached_modules"] += _attach_swiglu_mlp(module, spec)

        freeze_stats = freeze_non_lora_params(chunk)
        trainable_stats = trainable_param_stats(chunk)
        stats["frozen_tensors"] += freeze_stats["frozen_tensors"]
        stats["trainable_tensors"] += trainable_stats["trainable_tensors"]

    return stats


__all__ = [
    "LoRAWrappedGroupedLinear",
    "LoRAWrappedLinear",
    "apply_lora_to_chunks",
    "get_grouped_lora_adapter",
    "get_linear_lora_adapter",
    "iter_lora_adapter_modules",
]

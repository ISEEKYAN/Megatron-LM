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
        use_base_normalized_input: bool = False,
    ):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self._adapter_input_fn = adapter_input_fn
        self._use_base_normalized_input = bool(use_base_normalized_input)
        if self._use_base_normalized_input and adapter_input_fn is not None:
            raise ValueError(
                "Choose either the base normalized input or adapter_input_fn, not both."
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self._use_base_normalized_input:
            out, adapter_x = self.base.forward_with_normalized_input(x, *args, **kwargs)
        else:
            out = self.base(x, *args, **kwargs)
            adapter_x = (
                self._adapter_input_fn(x) if self._adapter_input_fn is not None else x
            )
        return out + self.adapter(adapter_x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            if name in {"weight", "parametrizations"}:
                owner = _linear_weight_owner(base)
                if owner is not None and hasattr(owner, name):
                    return getattr(owner, name)
            return getattr(base, name)


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


def _linear_weight_owner(module: nn.Module) -> nn.Module | None:
    inner = getattr(module, "linear", None)
    if isinstance(inner, nn.Module) and isinstance(getattr(inner, "weight", None), nn.Parameter):
        return inner
    if isinstance(getattr(module, "weight", None), nn.Parameter):
        return module
    return None


def _local_linear_features(module: nn.Module) -> tuple[int, int]:
    """Return the local input/output dimensions from the real weight surface."""

    owner = _linear_weight_owner(module)
    candidates = (
        [owner.weight]
        if owner is not None
        else [param for param in module.parameters(recurse=False) if param.ndim >= 2]
    )
    if not candidates:
        candidates = [param for param in module.parameters() if param.ndim >= 2]
    shapes = {(int(param.shape[-1]), int(param.shape[-2])) for param in candidates}
    if len(shapes) != 1:
        raise ValueError(
            f"Cannot infer one local linear shape from {type(module).__name__}: {sorted(shapes)}"
        )
    return next(iter(shapes))


def _place_adapter_like(adapter: nn.Module, base: nn.Module) -> nn.Module:
    reference = next(base.parameters(), None)
    if reference is None:
        raise ValueError(f"Cannot place LoRA adapter for parameterless {type(base).__name__}.")
    return adapter.to(device=reference.device, dtype=reference.dtype)


def _enable_qkv_normalized_output(base_qkv: nn.Module) -> bool:
    linear = base_qkv.linear
    if not hasattr(linear, "layer_norm_weight"):
        return False
    if not hasattr(base_qkv, "forward_with_normalized_input"):
        raise TypeError(
            f"{type(base_qkv).__name__} does not expose its normalized linear input."
        )
    linear.return_layernorm_output = True
    linear.return_layernorm_output_gathered = False
    return True


def _attach_gqa_qkv(attn, spec: LoraSpec) -> bool:
    if not spec.targets_module("linear_qkv"):
        return False
    if isinstance(attn.qkv, LoRAWrappedLinear):
        return False
    ps = attn.ps
    local_in, local_out = _local_linear_features(attn.qkv)
    adapter = _place_adapter_like(
        LinearLoRA(
            local_in,
            local_out,
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
        ),
        attn.qkv,
    )
    use_base_normalized_input = _enable_qkv_normalized_output(attn.qkv)
    attn.qkv = LoRAWrappedLinear(
        attn.qkv, adapter, use_base_normalized_input=use_base_normalized_input
    )
    return True


def _attach_gqa_proj(attn, spec: LoraSpec) -> bool:
    if not spec.targets_module("linear_proj"):
        return False
    if isinstance(attn.proj, LoRAWrappedLinear):
        return False
    ps = attn.ps
    local_in, local_out = _local_linear_features(attn.proj)
    adapter = _place_adapter_like(
        LinearLoRA(
            local_in,
            local_out,
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
        ),
        attn.proj,
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
    adapter = _place_adapter_like(
        SharedGroupedLinearLoRA(
            experts.num_local_experts,
            in_features,
            out_features,
            spec.rank,
            alpha=spec.alpha,
            dropout=spec.dropout,
            use_rslora=spec.use_rslora,
        ),
        base,
    )
    setattr(experts, attr, LoRAWrappedGroupedLinear(base, adapter))
    return True


def _attach_swiglu_mlp(mlp, spec: LoraSpec) -> int:
    attached = 0
    hidden = mlp.gate_up.in_features
    intermediate = mlp.down.in_features
    if spec.targets_module("linear_fc1") and not isinstance(mlp.gate_up, LoRAWrappedLinear):
        adapter = _place_adapter_like(
            LinearLoRA(
                hidden,
                mlp.gate_up.out_features,
                spec.rank,
                alpha=spec.alpha,
                dropout=spec.dropout,
                use_rslora=spec.use_rslora,
            ),
            mlp.gate_up,
        )
        mlp.gate_up = LoRAWrappedLinear(mlp.gate_up, adapter)
        attached += 1
    if spec.targets_module("linear_fc2") and not isinstance(mlp.down, LoRAWrappedLinear):
        adapter = _place_adapter_like(
            LinearLoRA(
                intermediate,
                mlp.down.out_features,
                spec.rank,
                alpha=spec.alpha,
                dropout=spec.dropout,
                use_rslora=spec.use_rslora,
            ),
            mlp.down,
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
                config_hidden, fc1_out = _local_linear_features(module.fc1)
                fc2_in, fc2_out = _local_linear_features(module.fc2)
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

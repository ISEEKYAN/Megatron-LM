# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Post-build LoRA applicator — mirrors ``apply_qat_to_chunks`` opt-in surface."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
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
        self._merged_delta: torch.Tensor | None = None
        if self._use_base_normalized_input and adapter_input_fn is not None:
            raise ValueError(
                "Choose either the base normalized input or adapter_input_fn, not both."
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self._merged_delta is not None:
            return self.base(x, *args, **kwargs)
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
        self._merged_deltas: tuple[torch.Tensor, ...] | None = None

    def forward(
        self, x: torch.Tensor, splits: list[int], *args, **kwargs
    ) -> torch.Tensor:
        out = self.base(x, splits, *args, **kwargs)
        if self._merged_deltas is not None:
            return out
        return out + self.adapter(x, splits)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


@dataclass(frozen=True)
class LoraTargetRule:
    """One model-declared linear attribute handled by the generic applicator."""

    owner_type: str
    attr: str
    target: str


def _linear_weight_owner(module: nn.Module) -> nn.Module | None:
    inner = getattr(module, "linear", None)
    if isinstance(inner, nn.Module) and isinstance(
        getattr(inner, "weight", None), nn.Parameter
    ):
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
        raise ValueError(
            f"Cannot place LoRA adapter for parameterless {type(base).__name__}."
        )
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


def _attach_gqa_qkv(attn, spec: LoraSpec, *, module_path: str = "") -> bool:
    if not spec.targets_module("linear_qkv") or spec.ignores_module(
        _module_path(module_path, "qkv")
    ):
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


def _module_path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _target_is_ignored(spec: LoraSpec, target: str, parent: str, child: str) -> bool:
    return spec.targets_module(target) and spec.ignores_module(
        _module_path(parent, child)
    )


def _attach_gqa_proj(attn, spec: LoraSpec, *, module_path: str = "") -> bool:
    if not spec.targets_module("linear_proj") or spec.ignores_module(
        _module_path(module_path, "proj")
    ):
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
    experts,
    attr: str,
    target: str,
    spec: LoraSpec,
    *,
    in_features: int,
    out_features: int,
    tp_group=None,
    module_path: str = "",
) -> bool:
    if not spec.targets_module(target) or spec.ignores_module(
        _module_path(module_path, attr)
    ):
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
            tp_group=tp_group,
        ),
        base,
    )
    setattr(experts, attr, LoRAWrappedGroupedLinear(base, adapter))
    return True


def _attach_swiglu_mlp(mlp, spec: LoraSpec, *, module_path: str = "") -> int:
    attached = 0
    hidden = mlp.gate_up.in_features
    intermediate = mlp.down.in_features
    if (
        spec.targets_module("linear_fc1")
        and not spec.ignores_module(_module_path(module_path, "gate_up"))
        and not isinstance(mlp.gate_up, LoRAWrappedLinear)
    ):
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
    if (
        spec.targets_module("linear_fc2")
        and not spec.ignores_module(_module_path(module_path, "down"))
        and not isinstance(mlp.down, LoRAWrappedLinear)
    ):
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


def _attach_declared_linear(
    owner: nn.Module, rule: LoraTargetRule, spec: LoraSpec, *, module_path: str
) -> bool:
    if type(owner).__name__ != rule.owner_type or not spec.targets_module(rule.target):
        return False
    target_path = _module_path(module_path, rule.attr)
    if spec.ignores_module(target_path):
        return False
    base = getattr(owner, rule.attr, None)
    if not isinstance(base, nn.Module) or isinstance(base, LoRAWrappedLinear):
        return False

    local_in, local_out = _local_linear_features(base)
    kwargs: dict[str, Any] = {}
    use_base_normalized_input = False
    if hasattr(base, "local_out"):
        tp_size = int(getattr(base, "tp_size", 1))
        kwargs.update(
            tp_group=getattr(base, "tp_group", None),
            tp_rank=int(getattr(base, "tp_rank", 0)),
            sequence_parallel_input=bool(getattr(base, "use_sp", False)),
            rank_partition_size=tp_size,
            rank_partitioned_a=tp_size > 1,
            a_tensor_model_parallel=tp_size > 1,
            b_tensor_model_parallel=tp_size > 1,
        )
        if hasattr(base, "linear"):
            use_base_normalized_input = _enable_qkv_normalized_output(base)
    elif hasattr(base, "local_in"):
        tp_size = int(getattr(base, "tp_size", 1))
        kwargs.update(
            tp_group=getattr(base, "tp_group", None),
            tp_rank=int(getattr(base, "tp_rank", 0)),
            sequence_parallel_scatter_output=bool(getattr(base, "use_sp", False)),
            input_parallel_reduce=tp_size > 1,
            output_partition_size=tp_size,
            output_partitioned_b=tp_size > 1,
            a_tensor_model_parallel=tp_size > 1,
            b_tensor_model_parallel=tp_size > 1,
        )

    adapter = _place_adapter_like(
        LinearLoRA(
            local_in,
            local_out,
            spec.rank,
            alpha=spec.alpha,
            dropout=spec.dropout,
            use_rslora=spec.use_rslora,
            **kwargs,
        ),
        base,
    )
    setattr(
        owner,
        rule.attr,
        LoRAWrappedLinear(
            base, adapter, use_base_normalized_input=use_base_normalized_input
        ),
    )
    return True


def get_linear_lora_adapter(module: nn.Module, attr: str) -> LinearLoRA | None:
    child = getattr(module, attr, None)
    if isinstance(child, LoRAWrappedLinear):
        return child.adapter
    return None


def get_grouped_lora_adapter(
    module: nn.Module, attr: str
) -> SharedGroupedLinearLoRA | None:
    child = getattr(module, attr, None)
    if isinstance(child, LoRAWrappedGroupedLinear):
        return child.adapter
    return None


def iter_lora_adapter_modules(module: nn.Module):
    """Yield native LoRA adapter modules (for PEFT export / OLoRA init)."""
    seen: set[int] = set()
    for child in module.modules():
        adapter = None
        if isinstance(child, LoRAWrappedLinear):
            adapter = child.adapter
        elif isinstance(child, LoRAWrappedGroupedLinear):
            adapter = child.adapter
        elif isinstance(child, (LinearLoRA, SharedGroupedLinearLoRA)):
            adapter = child
        if adapter is not None and id(adapter) not in seen:
            seen.add(id(adapter))
            yield adapter


def _lora_wrappers(chunks: Sequence[nn.Module]):
    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, (LoRAWrappedLinear, LoRAWrappedGroupedLinear)):
                yield module


def _grouped_base_weights(
    wrapper: LoRAWrappedGroupedLinear,
) -> tuple[nn.Parameter, ...]:
    weights = tuple(
        getattr(wrapper.base, f"weight{index}", None)
        for index in range(wrapper.adapter.num_local_experts)
    )
    if not weights or any(not isinstance(weight, nn.Parameter) for weight in weights):
        raise TypeError(
            f"Cannot merge grouped LoRA into {type(wrapper.base).__name__}: "
            "expected one weight{index} parameter per local expert."
        )
    return weights


def merge_lora_in_chunks(chunks: Sequence[nn.Module]) -> dict[str, int]:
    """Merge attached adapters into base weights without removing wrappers."""

    wrappers = list(_lora_wrappers(chunks))
    if not wrappers:
        raise RuntimeError("Cannot merge LoRA: no LoRA wrappers are attached.")
    already_merged = [
        wrapper
        for wrapper in wrappers
        if getattr(wrapper, "_merged_delta", None) is not None
        or getattr(wrapper, "_merged_deltas", None) is not None
    ]
    if already_merged:
        raise RuntimeError(
            f"Cannot merge LoRA: {len(already_merged)} wrapper(s) are already merged."
        )

    plans: list[
        tuple[nn.Module, tuple[nn.Parameter, ...], tuple[torch.Tensor, ...]]
    ] = []
    for wrapper in wrappers:
        if isinstance(wrapper, LoRAWrappedLinear):
            owner = _linear_weight_owner(wrapper.base)
            if owner is None:
                raise TypeError(
                    f"Cannot merge LoRA into {type(wrapper.base).__name__}: no 2-D weight."
                )
            delta = wrapper.adapter.materialized_delta_weight().to(
                device=owner.weight.device, dtype=owner.weight.dtype
            )
            weights = (owner.weight,)
            deltas = (delta,)
        else:
            weights = _grouped_base_weights(wrapper)
            delta = wrapper.adapter.materialized_delta_weight()
            deltas = tuple(
                delta.to(device=weight.device, dtype=weight.dtype) for weight in weights
            )
        for weight, delta in zip(weights, deltas, strict=True):
            if weight.shape != delta.shape:
                raise ValueError(
                    f"Cannot merge LoRA: base shape {tuple(weight.shape)} != "
                    f"delta shape {tuple(delta.shape)}."
                )
        plans.append((wrapper, weights, deltas))

    with torch.no_grad():
        for wrapper, weights, deltas in plans:
            for weight, delta in zip(weights, deltas, strict=True):
                weight.add_(delta)
            if isinstance(wrapper, LoRAWrappedLinear):
                wrapper._merged_delta = deltas[0]
            else:
                wrapper._merged_deltas = deltas
    return {"merged_modules": len(plans)}


def unmerge_lora_in_chunks(chunks: Sequence[nn.Module]) -> dict[str, int]:
    """Undo an in-place merge and reactivate the attached adapters."""

    wrappers = list(_lora_wrappers(chunks))
    if not wrappers:
        raise RuntimeError("Cannot unmerge LoRA: no LoRA wrappers are attached.")
    not_merged = [
        wrapper
        for wrapper in wrappers
        if getattr(wrapper, "_merged_delta", None) is None
        and getattr(wrapper, "_merged_deltas", None) is None
    ]
    if not_merged:
        raise RuntimeError(
            f"Cannot unmerge LoRA: {len(not_merged)} wrapper(s) are not merged."
        )

    plans = []
    for wrapper in wrappers:
        if isinstance(wrapper, LoRAWrappedLinear):
            owner = _linear_weight_owner(wrapper.base)
            if owner is None:
                raise TypeError(
                    f"Cannot unmerge LoRA from {type(wrapper.base).__name__}: no 2-D weight."
                )
            plans.append((wrapper, (owner.weight,), (wrapper._merged_delta,)))
        else:
            plans.append(
                (wrapper, _grouped_base_weights(wrapper), wrapper._merged_deltas)
            )

    with torch.no_grad():
        for wrapper, weights, deltas in plans:
            for weight, delta in zip(weights, deltas, strict=True):
                weight.sub_(delta)
            if isinstance(wrapper, LoRAWrappedLinear):
                wrapper._merged_delta = None
            else:
                wrapper._merged_deltas = None
    return {"unmerged_modules": len(plans)}


def remove_lora_from_chunks(chunks: Sequence[nn.Module]) -> dict[str, int]:
    """Remove adapters and restore the pre-apply trainability state."""

    wrappers = list(_lora_wrappers(chunks))
    if not wrappers:
        raise RuntimeError("Cannot remove LoRA: no LoRA wrappers are attached.")
    if any(
        getattr(wrapper, "_merged_delta", None) is not None
        or getattr(wrapper, "_merged_deltas", None) is not None
        for wrapper in wrappers
    ):
        raise RuntimeError(
            "Cannot remove LoRA while adapters are merged; unmerge first."
        )
    missing_snapshots = [
        chunk
        for chunk in chunks
        if getattr(chunk, "_mlite_lora_requires_grad_state", None) is None
    ]
    if missing_snapshots:
        raise RuntimeError(
            "LoRA trainability snapshot is missing; refusing partial removal."
        )

    removed = 0
    for chunk in chunks:
        for parent in list(chunk.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, (LoRAWrappedLinear, LoRAWrappedGroupedLinear)):
                    setattr(parent, name, child.base)
                    removed += 1
        prior = getattr(chunk, "_mlite_lora_requires_grad_state", None)
        for parameter in chunk.parameters():
            if id(parameter) in prior:
                parameter.requires_grad_(prior[id(parameter)])
        delattr(chunk, "_mlite_lora_requires_grad_state")
    return {"removed_modules": removed}


def _named_lora_adapter_tensors(chunks: Sequence[nn.Module]):
    seen: set[int] = set()
    for chunk_index, chunk in enumerate(chunks):
        for module_name, module in chunk.named_modules():
            if isinstance(module, (LoRAWrappedLinear, LoRAWrappedGroupedLinear)):
                adapter = module.adapter
                if id(adapter) in seen:
                    continue
                seen.add(id(adapter))
                prefix = f"chunk{chunk_index}.{module_name}".rstrip(".")
                yield f"{prefix}.lora_a", adapter.lora_a
                yield f"{prefix}.lora_b", adapter.lora_b


def save_lora_adapter_state(
    chunks: Sequence[nn.Module], path: str | Path
) -> dict[str, int]:
    """Save only attached adapter tensors using stable wrapper paths."""

    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in _named_lora_adapter_tensors(chunks)
    }
    if not state:
        raise RuntimeError(
            "Cannot save LoRA adapter state: no LoRA adapters are attached."
        )
    torch.save({"format": "mlite_lora_adapter_v1", "state": state}, Path(path))
    return {"saved_tensors": len(state)}


def load_lora_adapter_state(
    chunks: Sequence[nn.Module], path: str | Path
) -> dict[str, int]:
    """Load adapter tensors only after validating the complete key/shape set."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"LoRA adapter checkpoint does not exist: {checkpoint_path}"
        )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "mlite_lora_adapter_v1"
    ):
        raise RuntimeError(f"Invalid LoRA adapter checkpoint format: {checkpoint_path}")
    saved = payload.get("state")
    if not isinstance(saved, dict):
        raise RuntimeError(f"Invalid LoRA adapter checkpoint state: {checkpoint_path}")
    current = dict(_named_lora_adapter_tensors(chunks))
    if set(saved) != set(current):
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        raise RuntimeError(
            "LoRA adapter checkpoint does not match attached adapters: "
            f"missing={missing}, unexpected={unexpected}."
        )
    for name, parameter in current.items():
        tensor = saved[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape:
            raise RuntimeError(
                f"LoRA adapter checkpoint tensor {name!r} has shape "
                f"{getattr(tensor, 'shape', None)}, expected {tuple(parameter.shape)}."
            )
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(
                saved[name].to(device=parameter.device, dtype=parameter.dtype)
            )
    return {"loaded_tensors": len(current)}


def apply_lora_to_chunks(
    chunks: Sequence[nn.Module],
    spec: LoraSpec | dict[str, Any] | None,
    *,
    ps=None,
    model_targets: Sequence[LoraTargetRule] = (),
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
    validate_lora_parallel_support(
        spec, etp_size=getattr(ps, "etp_size", 1) if ps is not None else 1
    )
    existing = list(_lora_wrappers(chunks))
    if existing:
        raise RuntimeError(
            f"Cannot apply LoRA: {len(existing)} LoRA wrapper(s) are already attached."
        )
    for chunk in chunks:
        if hasattr(chunk, "_mlite_lora_requires_grad_state"):
            raise RuntimeError("Cannot apply LoRA: stale trainability snapshot exists.")
        chunk._mlite_lora_requires_grad_state = {
            id(parameter): parameter.requires_grad for parameter in chunk.parameters()
        }

    from megatron.lite.primitive.modules.experts import Experts
    from megatron.lite.primitive.modules.gqa import GQAttention
    from megatron.lite.primitive.modules.mlp import SwiGLUMLP

    expert_tp_group = (
        ps.tp_group
        if ps is not None and ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1
        else None
    )
    for chunk in chunks:
        for module_path, module in list(chunk.named_modules()):
            if isinstance(module, GQAttention):
                stats["skipped_ignored"] += int(
                    _target_is_ignored(spec, "linear_qkv", module_path, "qkv")
                )
                stats["skipped_ignored"] += int(
                    _target_is_ignored(spec, "linear_proj", module_path, "proj")
                )
                if _attach_gqa_qkv(module, spec, module_path=module_path):
                    stats["attached_modules"] += 1
                if _attach_gqa_proj(module, spec, module_path=module_path):
                    stats["attached_modules"] += 1
            elif isinstance(module, Experts):
                stats["skipped_ignored"] += int(
                    _target_is_ignored(spec, "linear_fc1", module_path, "fc1")
                )
                stats["skipped_ignored"] += int(
                    _target_is_ignored(spec, "linear_fc2", module_path, "fc2")
                )
                config_hidden, fc1_out = _local_linear_features(module.fc1)
                fc2_in, fc2_out = _local_linear_features(module.fc2)
                if _attach_expert_fc(
                    module,
                    "fc1",
                    "linear_fc1",
                    spec,
                    in_features=config_hidden,
                    out_features=fc1_out,
                    tp_group=expert_tp_group,
                    module_path=module_path,
                ):
                    stats["attached_modules"] += 1
                if _attach_expert_fc(
                    module,
                    "fc2",
                    "linear_fc2",
                    spec,
                    in_features=fc2_in,
                    out_features=fc2_out,
                    tp_group=expert_tp_group,
                    module_path=module_path,
                ):
                    stats["attached_modules"] += 1
            elif isinstance(module, SwiGLUMLP):
                stats["skipped_ignored"] += int(
                    _target_is_ignored(spec, "linear_fc1", module_path, "gate_up")
                )
                stats["skipped_ignored"] += int(
                    _target_is_ignored(spec, "linear_fc2", module_path, "down")
                )
                stats["attached_modules"] += _attach_swiglu_mlp(
                    module, spec, module_path=module_path
                )
            for rule in model_targets:
                if (
                    type(module).__name__ == rule.owner_type
                    and spec.targets_module(rule.target)
                    and spec.ignores_module(_module_path(module_path, rule.attr))
                ):
                    stats["skipped_ignored"] += 1
                    continue
                if _attach_declared_linear(module, rule, spec, module_path=module_path):
                    stats["attached_modules"] += 1

        freeze_stats = freeze_non_lora_params(chunk)
        trainable_stats = trainable_param_stats(chunk)
        stats["frozen_tensors"] += freeze_stats["frozen_tensors"]
        stats["trainable_tensors"] += trainable_stats["trainable_tensors"]

    if stats["attached_modules"] == 0:
        for chunk in chunks:
            prior = chunk._mlite_lora_requires_grad_state
            for parameter in chunk.parameters():
                parameter.requires_grad_(prior[id(parameter)])
            delattr(chunk, "_mlite_lora_requires_grad_state")
        raise RuntimeError("LoRA is enabled but no declared target modules matched.")
    return stats


def validate_lora_parallel_support(
    spec: LoraSpec | dict[str, Any] | None, *, etp_size: int | None
) -> None:
    """Reject unsupported LoRA+ETP before model construction or mutation."""

    normalized = normalize_lora_spec(spec)
    effective_etp = 1 if etp_size is None else int(etp_size)
    if normalized.enabled and normalized.rank > 0 and effective_etp > 1:
        raise NotImplementedError(
            "Megatron Lite LoRA does not support ETP>1; use etp_size=1."
        )


__all__ = [
    "LoRAWrappedGroupedLinear",
    "LoRAWrappedLinear",
    "LoraTargetRule",
    "apply_lora_to_chunks",
    "get_grouped_lora_adapter",
    "get_linear_lora_adapter",
    "iter_lora_adapter_modules",
    "load_lora_adapter_state",
    "merge_lora_in_chunks",
    "remove_lora_from_chunks",
    "save_lora_adapter_state",
    "unmerge_lora_in_chunks",
    "validate_lora_parallel_support",
]

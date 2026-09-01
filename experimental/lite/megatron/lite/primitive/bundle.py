# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ModelBundle — return type of protocol.build_model()."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - supported Torch always has DTensor
    DTensor = ()  # type: ignore[assignment]

from megatron.lite.primitive.parallel.state import ParallelState


@dataclass
class ModelBundle:
    """Everything runtime needs to run a training loop.

    Returned by protocol.build_model(). Model owns the construction
    of all fields — runtime just consumes them.
    """

    chunks: list[nn.Module]
    parallel_state: ParallelState
    optimizer: Any | None = None
    finalize_grads: Callable[[], None] | None = None
    forward_step: Callable[..., dict] | None = None
    # extra metadata (expert_classifier, model_cfg, etc.)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Report the effective memory features at the shared protocol exit."""
        recompute_wrapped = _count_marked_modules(self.chunks, "_megatron_lite_recompute_wrapped")
        expert_shard = _expert_shard_ratio(self.chunks, self.parallel_state)
        optimizer_devices = _optimizer_state_devices(self.optimizer)
        _log_rank0(
            "memory feature audit: "
            f"recompute_wrapped={recompute_wrapped}, "
            "activation_offload=unsupported, "
            f"expert_shard_ratio={expert_shard}, "
            f"optimizer_state_devices={optimizer_devices}"
        )


def _count_marked_modules(chunks: list[nn.Module], marker: str) -> int:
    return sum(bool(getattr(module, marker, False)) for chunk in chunks for module in chunk.modules())


def _expert_shard_ratio(chunks: list[nn.Module], parallel_state: ParallelState) -> str:
    """Report observed routed-expert DTensor sharding, never topology intent."""
    del parallel_state  # Placement is a property of the assembled parameters.
    expert_params = [
        parameter
        for chunk in chunks
        for name, parameter in chunk.named_parameters()
        if ".experts." in name or name.startswith("experts.")
    ]
    dtensor_experts = [parameter for parameter in expert_params if isinstance(parameter, DTensor)]
    if not dtensor_experts:
        return "n/a (no DTensor expert params)"

    sharded = sum(
        any(type(placement).__name__ == "Shard" for placement in parameter.placements)
        for parameter in dtensor_experts
    )
    return f"{sharded}/{len(dtensor_experts)} (DTensor placements)"


def _optimizer_state_devices(optimizer: Any | None) -> str:
    if optimizer is None:
        return "none"

    devices: set[str] = set()
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            devices.add(value.device.type)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        for attribute in ("state", "optimizer", "optimizers", "chained_optimizers"):
            nested = getattr(value, attribute, None)
            if nested is not None:
                visit(nested)

    visit(optimizer)
    return ",".join(sorted(devices)) if devices else "uninitialized"


def _log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[megatron.lite] {msg}", flush=True)

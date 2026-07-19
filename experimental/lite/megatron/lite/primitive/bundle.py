# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ModelBundle — return type of protocol.build_model()."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

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
    has_experts = any(
        name == "experts" or name.endswith(".experts")
        for chunk in chunks
        for name, _module in chunk.named_modules()
    )
    if not has_experts:
        return "n/a"
    return f"1/{parallel_state.ep_size} (ep)"


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

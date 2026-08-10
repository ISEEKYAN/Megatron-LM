# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared distributed-test helpers for Megatron Lite primitive contracts."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist


def canonical_lora_bank_names(banks) -> tuple[str, ...]:
    """Return a rank-independent order for mixed LoRA bank collectives."""
    return tuple(sorted(banks))


def build_lora_collective_descriptor(
    kind: str, reduction_tensor: torch.Tensor, bank_dtype: torch.dtype
) -> tuple[str, tuple[int, ...], str]:
    """Describe the actual tensor dtype and shape consumed by a bank collective."""
    assert reduction_tensor.dtype is torch.float32, (
        "LoRA reduction tensor dtype must be torch.float32"
    )
    if kind == "fc":
        assert bank_dtype is torch.bfloat16, "FC LoRA bank dtype must be torch.bfloat16"
        collective_dtype = bank_dtype
    else:
        assert kind == "attention", f"invalid LoRA bank kind: {kind}"
        collective_dtype = reduction_tensor.dtype
    return kind, tuple(reduction_tensor.shape), str(collective_dtype)


def preflight_lora_bank_collective_order(
    banks, build_descriptor: Callable[[str], tuple[str, tuple[int, ...], str]]
) -> tuple[tuple[str, str, tuple[int, ...], str], ...]:
    """Validate WORLD-wide `(name, kind, shape, dtype)` before collectives."""
    envelope: dict[str, Any] = {"error": None, "record": None}
    try:
        records = []
        for name in canonical_lora_bank_names(banks):
            descriptor = build_descriptor(name)
            assert type(descriptor) is tuple and len(descriptor) == 3
            kind, shape, dtype = descriptor
            assert kind in {"fc", "attention"}, f"invalid LoRA bank kind: {kind}"
            assert type(shape) is tuple and all(type(dim) is int for dim in shape)
            assert type(dtype) is str
            records.append((name, kind, shape, dtype))
        envelope["record"] = tuple(
            sorted(records, key=lambda record: (record[1] != "fc", record[0]))
        )
    except Exception as error:
        envelope["error"] = f"{type(error).__name__}: {error}"
    envelopes: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(envelopes, envelope, group=dist.group.WORLD)
    errors = [item["error"] for item in envelopes if item is not None and item["error"]]
    if errors:
        raise RuntimeError(
            f"mixed LoRA bank collective preflight failed: local record error: {errors[0]}"
        )
    records = [item["record"] for item in envelopes if item is not None]
    if not records or any(record != records[0] for record in records):
        raise RuntimeError(
            "mixed LoRA bank collective preflight failed: records differ across WORLD"
        )
    return records[0]


def select_lora_bank_owner_group(parallel_state, *, is_expert_bank: bool):
    """Return the dist-opt replica group for an FC or attention LoRA bank."""
    return parallel_state.ep_dp_group if is_expert_bank else parallel_state.dp_group


def gather_owner_factor_records_or_raise(
    owner_group,
    build_record: Callable[[], Any],
    validate_records: Callable[[list[Any]], None],
) -> list[Any]:
    """Validate factor records without allowing lane-local errors to strand WORLD.

    The owner lane always exchanges an envelope first.  Every rank then enters
    one WORLD error bridge, so a malformed local shard makes all ranks raise
    the same error before the next world collective can begin.
    """
    envelope: dict[str, Any] = {"error": None, "record": None}
    try:
        envelope["record"] = build_record()
    except Exception as error:
        envelope["error"] = f"{type(error).__name__}: {error}"
    envelopes: list[dict[str, Any] | None] = [None] * dist.get_world_size(owner_group)
    dist.all_gather_object(envelopes, envelope, group=owner_group)

    lane_error: str | None = None
    records: list[Any] = []
    try:
        errors = [
            item["error"] for item in envelopes if item is not None and item["error"]
        ]
        if errors:
            raise RuntimeError(errors[0])
        records = [
            item["record"]
            for item in envelopes
            if item is not None and item["record"] is not None
        ]
        validate_records(records)
    except Exception as error:
        lane_error = f"{type(error).__name__}: {error}"

    world_errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(world_errors, lane_error, group=dist.group.WORLD)
    errors = [error for error in world_errors if error is not None]
    if errors:
        raise RuntimeError(f"owner-group factor validation failed: {errors[0]}")
    return records

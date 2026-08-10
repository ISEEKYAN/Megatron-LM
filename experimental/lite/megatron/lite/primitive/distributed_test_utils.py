# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared distributed-test helpers for Megatron Lite primitive contracts."""

from __future__ import annotations

from typing import Any, Callable

import torch.distributed as dist


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
    dist.all_gather_object(world_errors, lane_error)
    errors = [error for error in world_errors if error is not None]
    if errors:
        raise RuntimeError(f"owner-group factor validation failed: {errors[0]}")
    return records

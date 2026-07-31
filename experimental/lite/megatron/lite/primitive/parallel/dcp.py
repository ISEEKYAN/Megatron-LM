# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dynamic context-parallel scheduling for MLite's packed runtime contract.

Megatron-Core owns the scheduling policy and dynamic process groups.  This
module adapts MLite ``PackedBatch`` objects to that policy, binds exactly one
scheduled forward to its selected group, and restores detached runtime results
after backward has already completed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import torch
from megatron.lite.runtime.contracts import LossContext, PackedBatch
from megatron.lite.runtime.contracts.loss import split_loss_context

_SAMPLE_IDS = "_mlite_dcp_sample_ids"
_GROUP_LEADER = "_mlite_dcp_group_leader"
_LOCAL_CP_SIZE = "_mlite_dcp_local_cp_size"


def _require_positive_power_of_two(value: int, name: str) -> None:
    if type(value) is not int or value < 1 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value!r}.")


def _batch_offsets(batch: PackedBatch) -> torch.Tensor:
    if batch.seq_lens.ndim != 1 or batch.seq_lens.numel() == 0:
        raise ValueError("Dynamic CP requires non-empty 1-D PackedBatch.seq_lens.")
    if int(batch.seq_lens.min().item()) < 1:
        raise ValueError("Dynamic CP does not accept empty sequences.")
    offsets = torch.empty(
        batch.seq_lens.numel() + 1, dtype=torch.int64, device=batch.seq_lens.device
    )
    offsets[0] = 0
    offsets[1:] = batch.seq_lens.to(torch.int64).cumsum(0)
    total = int(offsets[-1].item())
    for name in ("input_ids", "labels"):
        value = getattr(batch, name)
        if value.ndim != 1 or value.numel() != total:
            raise ValueError(
                f"PackedBatch.{name} must be 1-D with {total} tokens for Dynamic CP."
            )
    for name in ("loss_mask", "position_ids"):
        value = getattr(batch, name)
        if value is not None and (value.ndim != 1 or value.numel() != total):
            raise ValueError(
                f"PackedBatch.{name} must be 1-D with {total} tokens for Dynamic CP."
            )
    if batch.routed_experts is not None or batch.r3_replay_mask is not None:
        raise NotImplementedError(
            "Dynamic CP does not support router replay in the first release."
        )
    return offsets


def _select_source_batch(source_batch: Any, sample_ids: list[int]) -> Any:
    if source_batch is None:
        return None
    index = torch.tensor(sample_ids, dtype=torch.int64)
    device = getattr(source_batch, "device", None)
    if device is not None:
        index = index.to(device)
    index_select = getattr(source_batch, "index_select", None)
    if callable(index_select):
        try:
            return index_select(0, index)
        except (TypeError, RuntimeError):
            pass
    try:
        return source_batch[index]
    except (IndexError, KeyError, TypeError) as exc:
        raise TypeError(
            "Dynamic CP LossContext.source_batch must support sample-axis index selection."
        ) from exc


def _select_batch(batch: PackedBatch, sample_ids: list[int]) -> PackedBatch:
    offsets = _batch_offsets(batch)
    token_slices = [
        slice(int(offsets[sample_id]), int(offsets[sample_id + 1]))
        for sample_id in sample_ids
    ]

    def _select(name: str) -> torch.Tensor | None:
        value = getattr(batch, name)
        return (
            None if value is None else torch.cat([value[part] for part in token_slices])
        )

    return PackedBatch(
        input_ids=_select("input_ids"),
        labels=_select("labels"),
        seq_lens=batch.seq_lens[sample_ids],
        loss_mask=_select("loss_mask"),
        position_ids=_select("position_ids"),
        extras={_SAMPLE_IDS: sample_ids},
    )


def _cp_members(assignments: list[list[int]], rank: int) -> list[int]:
    sample_ids = assignments[rank]
    if not sample_ids:
        raise RuntimeError(
            "Megatron-Core Dynamic CP returned an empty rank assignment."
        )
    return [peer for peer, peer_ids in enumerate(assignments) if peer_ids == sample_ids]


def schedule(
    data_iterator: Iterator[Any],
    *,
    num_microbatches: int,
    dp_size: int,
    cp_size: int,
    dcp_group: Any,
    max_seqlen_per_dp_cp_rank: int,
    min_cp_size: int = 1,
    scheduler_cls: type | None = None,
) -> tuple[list[tuple[PackedBatch, LossContext | None]], int, int]:
    """Schedule a replicated MLite batch with MCore's Dynamic CP policy."""
    if num_microbatches != 1:
        raise ValueError("Dynamic CP schedules exactly one replicated global batch.")
    _require_positive_power_of_two(min_cp_size, "min_dynamic_context_parallel_size")
    total_ranks = dp_size * cp_size
    if dcp_group is None or dcp_group.size() != total_ranks:
        raise ValueError("Dynamic CP group size must equal dp_size * cp_size.")
    if total_ranks < 2 or total_ranks % 2:
        raise ValueError("Dynamic CP requires an even DP×CP group.")
    if total_ranks < min_cp_size:
        raise ValueError("Dynamic CP minimum group size exceeds the DP×CP topology.")
    if max_seqlen_per_dp_cp_rank < 1:
        raise ValueError("max_seqlen_per_dp_cp_rank must be positive.")

    batch, context = split_loss_context(next(data_iterator))
    if not isinstance(batch, PackedBatch):
        raise TypeError("Dynamic CP requires a replicated PackedBatch input.")
    _batch_offsets(batch)
    if batch.extras:
        raise ValueError(
            "Dynamic CP input PackedBatch.extras must be empty; scheduler metadata is runtime-owned."
        )
    seq_lens = [int(length) for length in batch.seq_lens.tolist()]
    if max(seq_lens) > max_seqlen_per_dp_cp_rank * total_ranks:
        raise ValueError("Dynamic CP sequence exceeds the DP×CP capacity.")

    if scheduler_cls is None:
        try:
            from megatron.core.datasets import data_schedule
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Dynamic CP requires Megatron-Core DefaultDynamicCPScheduler."
            ) from exc
        scheduler_cls = data_schedule.DefaultDynamicCPScheduler
    scheduler = scheduler_cls(
        max_seqlen_per_dp_cp_rank=max_seqlen_per_dp_cp_rank,
        cp_size=cp_size,
        dp_size=dp_size,
        microbatch_group_size_per_vp_stage=None,
        min_cp_size=min_cp_size,
    )
    groups = scheduler.get_groups_and_subsamples(list(enumerate(seq_lens)))
    rank = dcp_group.rank()
    scheduled = []
    seen_leader_ids: list[int] = []
    for assignments in groups:
        if len(assignments) != total_ranks or any(not ids for ids in assignments):
            raise RuntimeError("Dynamic CP must assign every DP×CP rank.")
        sample_ids = [int(sample_id) for sample_id in assignments[rank]]
        members = _cp_members(assignments, rank)
        local_cp_size = len(members)
        if members != list(range(members[0], members[0] + local_cp_size)):
            raise RuntimeError(
                f"Dynamic CP group members must be contiguous, got {members}."
            )
        if members[0] % local_cp_size:
            raise RuntimeError(
                f"Dynamic CP group {members} is not aligned to its size."
            )
        selected = _select_batch(batch, sample_ids)
        selected.extras.update(
            {_LOCAL_CP_SIZE: local_cp_size, _GROUP_LEADER: rank == members[0]}
        )
        selected_context = (
            None
            if context is None
            else replace(
                context,
                source_batch=_select_source_batch(context.source_batch, sample_ids),
            )
        )
        scheduled.append((selected, selected_context))
        if rank == members[0]:
            seen_leader_ids.extend(sample_ids)

    gathered_ids: list[list[int] | None] = [None] * total_ranks
    torch.distributed.all_gather_object(gathered_ids, seen_leader_ids, group=dcp_group)
    all_ids = [sample_id for ids in gathered_ids for sample_id in (ids or [])]
    if sorted(all_ids) != list(range(len(seq_lens))):
        raise RuntimeError(
            "Dynamic CP scheduling must cover every sample exactly once across group leaders."
        )
    return scheduled, len(scheduled), len(seq_lens)


def initialize_groups(parallel_state: Any, config: Any) -> dict[int, Any]:
    """Pre-create MCore's power-of-two subgroups in global collective order."""
    min_cp_size = int(config.min_dynamic_context_parallel_size)
    _require_positive_power_of_two(min_cp_size, "min_dynamic_context_parallel_size")
    try:
        from megatron.core.parallel_state import create_dynamic_dp_cp_groups
    except ImportError as exc:
        raise RuntimeError(
            "Dynamic CP requires Megatron-Core dynamic group APIs."
        ) from exc

    rank = torch.distributed.get_rank()
    tp = int(config.tp)
    cp = int(config.cp)
    pp = int(config.pp)
    dp = int(parallel_state.dp_size)
    total = dp * cp
    if total < 2 or total % 2:
        raise ValueError("Dynamic CP requires an even DP×CP group.")
    if total < min_cp_size:
        raise ValueError("Dynamic CP minimum group size exceeds the DP×CP topology.")

    def _rank(tp_rank: int, cp_rank: int, dp_rank: int, pp_rank: int) -> int:
        return ((pp_rank * dp + dp_rank) * cp + cp_rank) * tp + tp_rank

    local_groups: dict[int, Any] = {}
    for pp_rank in range(pp):
        for tp_rank in range(tp):
            ranks = [
                _rank(tp_rank, cp_rank, dp_rank, pp_rank)
                for dp_rank in range(dp)
                for cp_rank in range(cp)
            ]
            groups = create_dynamic_dp_cp_groups(
                rank, ranks, pg_options=None, min_cp_size=min_cp_size
            )
            if rank in ranks:
                local_groups.update(groups)
                local_groups[total] = parallel_state.dp_cp_group
    if not local_groups:
        raise RuntimeError("Dynamic CP group initialization did not include this rank.")
    return local_groups


def validate_runtime_config(runtime_config: Any) -> None:
    """Validate the first-release support matrix without initializing distributed state."""
    parallel = runtime_config.parallel
    if parallel.max_seqlen_per_dp_cp_rank is None:
        raise ValueError("dynamic_context_parallel requires max_seqlen_per_dp_cp_rank.")
    if not bool(runtime_config.impl_cfg.get("use_thd", True)):
        raise ValueError("dynamic_context_parallel requires packed THD inputs.")
    if parallel.pp != 1:
        raise NotImplementedError(
            "dynamic_context_parallel does not support pipeline parallelism."
        )
    if parallel.vpp != 1:
        raise NotImplementedError(
            "dynamic_context_parallel does not support virtual pipeline parallelism."
        )


def initialize_runtime(parallel_state: Any, runtime_config: Any) -> dict[int, Any]:
    """Validate the runtime contract and initialize its dynamic groups."""
    validate_runtime_config(runtime_config)
    parallel = runtime_config.parallel
    return initialize_groups(parallel_state, parallel)


@contextmanager
def bind_group(
    parallel_state: Any, groups: dict[int, Any], local_cp_size: int
) -> Iterator[None]:
    """Temporarily expose one MCore-created dynamic group to existing THD code."""
    group = groups.get(local_cp_size)
    if group is None or group.size() != local_cp_size:
        raise RuntimeError(f"No Dynamic CP group registered for size={local_cp_size}.")
    old = (parallel_state.cp_size, parallel_state.cp_rank, parallel_state.cp_group)
    try:
        parallel_state.cp_size = local_cp_size
        parallel_state.cp_rank = group.rank()
        parallel_state.cp_group = group
        yield
    finally:
        parallel_state.cp_size, parallel_state.cp_rank, parallel_state.cp_group = old


def prepare_runtime(
    data_iterator: Iterator[Any],
    *,
    num_microbatches: int,
    parallel_state: Any,
    config: Any,
    groups: dict[int, Any],
    forward_step: Callable,
    loss_fn: Callable | None,
    input_num_microbatches: int,
    output_sample_order: list[list[int]] | None = None,
) -> tuple[
    Iterator[Any], int, Callable, Callable | None, Callable | None, Callable, Callable
]:
    """Return scheduled inputs and a group-bound forward callback."""
    scheduled, count, batch_size = schedule(
        data_iterator,
        num_microbatches=num_microbatches,
        dp_size=parallel_state.dp_size,
        cp_size=parallel_state.cp_size,
        dcp_group=parallel_state.dp_cp_group,
        max_seqlen_per_dp_cp_rank=int(config.max_seqlen_per_dp_cp_rank),
        min_cp_size=int(config.min_dynamic_context_parallel_size),
    )

    def wrapped(model: Any, batch: PackedBatch):
        output = forward_step(model, batch)
        if not isinstance(output, dict):
            raise TypeError("Dynamic CP forward_step must return a dict.")
        output[_SAMPLE_IDS] = list(batch.extras[_SAMPLE_IDS])
        output[_GROUP_LEADER] = bool(batch.extras[_GROUP_LEADER])
        return output

    wrapped_loss = None
    finish_records = None
    if loss_fn is not None:

        def wrapped_loss(output: Any, batch: PackedBatch, *args):
            return loss_fn(output, batch, *args)

        if count != input_num_microbatches:
            original_loss = wrapped_loss
            runtime_scale = count / input_num_microbatches

            def wrapped_loss(output: Any, batch: PackedBatch, *args):
                result = original_loss(output, batch, *args)
                output.setdefault("_runtime_reported_loss", result[0].detach())
                return (result[0] * runtime_scale, *result[1:])

        def finish_records(records):
            return collect_records(
                records,
                batch_size=batch_size,
                dcp_group=parallel_state.dp_cp_group,
                output_sample_order=output_sample_order,
            )

    def microbatch_context(batch: PackedBatch):
        return bind_group(parallel_state, groups, int(batch.extras[_LOCAL_CP_SIZE]))

    def pre_forward_scale(batch: PackedBatch) -> int:
        return int(batch.extras[_LOCAL_CP_SIZE])

    return (
        iter(scheduled),
        count,
        wrapped,
        wrapped_loss,
        finish_records,
        microbatch_context,
        pre_forward_scale,
    )


def _detach_output_parts(
    model_output: dict[str, Any] | None, sample_ids: list[int]
) -> dict[str, list[torch.Tensor]]:
    if model_output is None:
        return {}
    parts_by_key = {}
    for key, value in model_output.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Dynamic CP runtime output {key!r} must be a tensor, "
                f"got {type(value).__name__}."
            )
        parts = list(value.unbind())
        if len(parts) != len(sample_ids):
            raise ValueError(
                f"Dynamic CP output {key!r} has {len(parts)} rows for "
                f"sample ids {sample_ids}."
            )
        parts_by_key[key] = [part.detach().cpu() for part in parts]
    return parts_by_key


def _detach_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_value(item) for item in value)
    return value


def collect_records(
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    dcp_group: Any,
    output_sample_order: list[list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Collect detached records in the caller-requested postprocess order."""
    local = []
    output_device = None
    for record in records:
        if not record.get(_GROUP_LEADER, False):
            continue
        sample_ids = [int(value) for value in record.get(_SAMPLE_IDS, [])]
        if not sample_ids:
            raise RuntimeError("Dynamic CP leader record has no sample identity.")
        model_output = record.get("model_output")
        if isinstance(model_output, dict) and output_device is None:
            output_device = next(
                (
                    value.device
                    for value in model_output.values()
                    if isinstance(value, torch.Tensor)
                ),
                None,
            )
        local.append(
            {
                "sample_ids": sample_ids,
                "model_output": _detach_output_parts(model_output, sample_ids),
                "loss": _detach_value(record.get("loss")),
                "metrics": _detach_value(record.get("metrics", {})),
            }
        )
    gathered: list[list[dict[str, Any]] | None] = [None] * dcp_group.size()
    torch.distributed.all_gather_object(gathered, local, group=dcp_group)
    leaders = [record for rank_records in gathered for record in (rank_records or [])]
    ids = [sample_id for record in leaders for sample_id in record["sample_ids"]]
    if sorted(ids) != list(range(batch_size)):
        raise RuntimeError(
            "Dynamic CP records must cover each original sample exactly once, "
            f"got {sorted(ids)}."
        )

    values_by_key: dict[str, dict[int, torch.Tensor]] = {}
    for record in leaders:
        for key, parts in record["model_output"].items():
            values = values_by_key.setdefault(key, {})
            for sample_id, part in zip(record["sample_ids"], parts, strict=True):
                values[sample_id] = part
    sample_order = (
        list(range(batch_size))
        if output_sample_order is None
        else [int(sample_id) for group in output_sample_order for sample_id in group]
    )
    if sorted(sample_order) != list(range(batch_size)):
        raise ValueError(
            "Dynamic CP output_sample_order must contain each sample exactly once, "
            f"got {sample_order}."
        )

    restored = {}
    for key, values in values_by_key.items():
        if sorted(values) != list(range(batch_size)):
            raise RuntimeError(f"Dynamic CP output {key!r} is missing samples.")
        nested = torch.nested.as_nested_tensor(
            [values[sample_id] for sample_id in sample_order], layout=torch.jagged
        )
        restored[key] = (
            nested.to(output_device) if output_device is not None else nested
        )

    leaders.sort(key=lambda record: min(record["sample_ids"]))
    completed = [
        {"loss": record["loss"], "metrics": record["metrics"]} for record in leaders
    ]
    if not completed:
        raise RuntimeError("Dynamic CP produced no group-leader records.")
    completed[0]["model_output"] = restored
    return completed


__all__ = [
    "bind_group",
    "collect_records",
    "initialize_groups",
    "initialize_runtime",
    "prepare_runtime",
    "schedule",
    "validate_runtime_config",
]

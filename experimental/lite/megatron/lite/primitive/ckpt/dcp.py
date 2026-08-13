# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
DCP (Distributed Checkpoint) framework for training checkpoints.

Model-agnostic: takes a placement function to describe how each parameter is sharded.
HF weight loading/saving is model-specific and lives in models/<name>/checkpoint.py.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.distributed.checkpoint as dcp  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.protocols import (
    ExpertClassifierFn,
    PlacementFn,
    default_expert_classifier,
    default_placement_fn,
)
from torch.distributed.device_mesh import DeviceMesh  # pyright: ignore[reportMissingImports]
from torch.distributed.tensor import DTensor  # pyright: ignore[reportMissingImports]
from torch.distributed.tensor import (  # pyright: ignore[reportMissingImports]
    Replicate,
    Shard,
)


def save_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int | str,
    path: str | None = None,
    config=None,
    ps: ParallelState | None = None,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
    *,
    use_dcp: bool | None = True,
    save_rng: bool = True,
    save_model: bool = True,
    save_optimizer: bool = True,
) -> None:
    """Save training checkpoint using DTensor + DCP for automatic resharding."""
    if path is None and isinstance(step, str):
        path = step
        step = 0
    if path is None:
        raise ValueError("checkpoint path is required")
    step = int(step)
    if use_dcp is None:
        use_dcp = True
    if not use_dcp:
        _save_local_training_checkpoint(model, optimizer, step, path, save_rng=save_rng)
        return
    if _supports_dist_opt_distckpt(model, optimizer):
        ckpt_path = os.path.join(path, f"step_{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        _save_dist_opt_checkpoint(
            model, optimizer, step, ckpt_path, save_model=save_model, save_optimizer=save_optimizer
        )
        if save_rng:
            _save_rng_sidecar(ckpt_path)
        log_rank0(f"Saved dist_opt checkpoint at step {step} to {ckpt_path}")
        return
    mfsdp_chunks = _mfsdp_chunks(model)
    if mfsdp_chunks is not None:
        ckpt_path = os.path.join(path, f"step_{step}")
        os.makedirs(ckpt_path, exist_ok=True)
        _save_mfsdp_checkpoint(
            mfsdp_chunks,
            optimizer,
            step,
            ckpt_path,
            ps=ps,
            save_model=save_model,
            save_optimizer=save_optimizer,
        )
        if save_rng:
            _save_rng_sidecar(ckpt_path)
        log_rank0(f"Saved M-FSDP checkpoint at step {step} to {ckpt_path}")
        return
    if config is None or ps is None:
        raise ValueError("DCP checkpointing requires config and ParallelState.")
    if not isinstance(model, nn.Module):
        raise TypeError("DCP checkpointing currently expects a single nn.Module.")
    dense_mesh, expert_mesh = _build_meshes(config)
    state_dict: dict = {"step": step}
    # Pipeline stages own DIFFERENT parameters but their local layers re-index
    # to 0..N, so without a per-stage prefix the DCP FQNs collide across pp ranks
    # (stage0 layer0 and stage1 layer1 both -> "model.0.layers.0..."), corrupting
    # the round-trip. Mirror distckpt's pp-aware keying: disjoint keyspace per stage.
    model_prefix = f"model_pp{ps.pp_rank}" if ps.pp_size > 1 else "model"

    if save_model:
        for name, param in model.named_parameters():
            placements = get_placements(name)
            mesh = expert_mesh if is_expert(name) else dense_mesh
            state_dict[f"{model_prefix}.{name}"] = _dcp_tensor_from_param(param, mesh, placements)

    ckpt_path = os.path.join(path, f"step_{step}")
    os.makedirs(ckpt_path, exist_ok=True)
    dcp.save(state_dict, checkpoint_id=ckpt_path)
    if save_optimizer:
        _save_optimizer_checkpoint(optimizer, ckpt_path)
    if save_rng:
        _save_rng_sidecar(ckpt_path)
    log_rank0(f"Saved training checkpoint at step {step} to {ckpt_path}")


def load_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    config=None,
    ps: ParallelState | None = None,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
    *,
    use_dcp: bool | None = True,
    load_rng: bool = True,
    load_parameter_state_update_legacy_format: bool = False,
    load_model: bool = True,
    load_optimizer: bool = True,
) -> int:
    """Load training checkpoint with automatic resharding across different parallel configs."""
    if use_dcp is None:
        use_dcp = True
    if not use_dcp:
        return _load_local_training_checkpoint(
            model,
            optimizer,
            path,
            load_rng=load_rng,
            load_parameter_state_update_legacy_format=load_parameter_state_update_legacy_format,
        )
    ckpt_path = _resolve_step_checkpoint_path(path)
    if _supports_dist_opt_distckpt(model, optimizer):
        step = _load_dist_opt_checkpoint(
            model, optimizer, ckpt_path, load_model=load_model, load_optimizer=load_optimizer
        )
        if load_rng:
            _load_rng_sidecar(ckpt_path)
        log_rank0(f"Loaded dist_opt checkpoint from {path} at step {step}")
        return step
    mfsdp_chunks = _mfsdp_chunks(model)
    if mfsdp_chunks is not None:
        step = _load_mfsdp_checkpoint(
            mfsdp_chunks,
            optimizer,
            ckpt_path,
            ps=ps,
            load_model=load_model,
            load_optimizer=load_optimizer,
        )
        if load_rng:
            _load_rng_sidecar(ckpt_path)
        log_rank0(f"Loaded M-FSDP checkpoint from {path} at step {step}")
        return step
    if config is None or ps is None:
        raise ValueError("DCP checkpointing requires config and ParallelState.")
    if not isinstance(model, nn.Module):
        raise TypeError("DCP checkpointing currently expects a single nn.Module.")
    dense_mesh, expert_mesh = _build_meshes(config)

    state_dict: dict = {"step": 0}
    # Same pp-aware keying as save (see save_training_checkpoint): per-stage
    # disjoint keyspace so pp ranks don't read each other's colliding FQNs.
    model_prefix = f"model_pp{ps.pp_rank}" if ps.pp_size > 1 else "model"

    if load_model:
        for name, param in model.named_parameters():
            placements = get_placements(name)
            mesh = expert_mesh if is_expert(name) else dense_mesh
            state_dict[f"{model_prefix}.{name}"] = _empty_dcp_tensor_like_param(
                param, mesh, placements
            )

    dcp.load(state_dict, checkpoint_id=ckpt_path)

    if load_model:
        for name, param in model.named_parameters():
            key = f"{model_prefix}.{name}"
            if key in state_dict:
                t = state_dict[key]
                with torch.no_grad():
                    _copy_tensor_(param, t)

    if load_optimizer:
        _load_optimizer_checkpoint(optimizer, ckpt_path)

    step = state_dict.get("step", 0)
    if load_rng:
        _load_rng_sidecar(ckpt_path)
    log_rank0(f"Loaded training checkpoint from {path} at step {step}")
    return step


def _resolve_step_checkpoint_path(path: str) -> str:
    if os.path.basename(path).startswith("step_"):
        return path

    step_dirs = sorted(
        [d for d in os.listdir(path) if d.startswith("step_")], key=lambda d: int(d.split("_")[1])
    )
    if step_dirs:
        return os.path.join(path, step_dirs[-1])
    return path


def _supports_dist_opt_distckpt(model: nn.Module | Iterable[nn.Module], optimizer) -> bool:
    try:
        from megatron.lite.primitive.ckpt.distckpt import supports_dist_opt_distckpt
    except ModuleNotFoundError as exc:
        if exc.name != "megatron.core":
            raise
        return False

    return supports_dist_opt_distckpt(model, optimizer)


def _save_dist_opt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int,
    path: str,
    *,
    save_model: bool,
    save_optimizer: bool,
) -> None:
    from megatron.lite.primitive.ckpt.distckpt import save_dist_opt_checkpoint

    save_dist_opt_checkpoint(
        model, optimizer, step, path, save_model=save_model, save_optimizer=save_optimizer
    )


def _load_dist_opt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    *,
    load_model: bool,
    load_optimizer: bool,
) -> int:
    from megatron.lite.primitive.ckpt.distckpt import load_dist_opt_checkpoint

    return load_dist_opt_checkpoint(
        model, optimizer, path, load_model=load_model, load_optimizer=load_optimizer
    )


def _optimizer_checkpoint_path(path: str) -> str:
    rank = dist.get_rank() if dist.is_initialized() else 0
    return os.path.join(path, f"optimizer_rank_{rank}.pt")


def _save_optimizer_checkpoint(optimizer, path: str) -> None:
    if optimizer is None:
        log_rank0("Skipping optimizer checkpoint save because optimizer is None")
        return
    state_dict_fn = getattr(optimizer, "state_dict", None)
    if not callable(state_dict_fn):
        raise TypeError(f"Optimizer {type(optimizer).__name__} does not provide state_dict().")
    torch.save(state_dict_fn(), _optimizer_checkpoint_path(path))


def _load_optimizer_checkpoint(optimizer, path: str) -> None:
    if optimizer is None:
        log_rank0("Skipping optimizer checkpoint load because optimizer is None")
        return
    ckpt_path = _optimizer_checkpoint_path(path)
    if not os.path.exists(ckpt_path):
        log_rank0(f"No optimizer checkpoint found at {ckpt_path}; loading model state only")
        return
    load_state_dict_fn = getattr(optimizer, "load_state_dict", None)
    if not callable(load_state_dict_fn):
        raise TypeError(f"Optimizer {type(optimizer).__name__} does not provide load_state_dict().")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_state_dict_fn(state)


def _model_chunks(model: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    if isinstance(model, nn.Module):
        return [model]
    chunks = list(model)
    if not all(isinstance(chunk, nn.Module) for chunk in chunks):
        raise TypeError("checkpoint model chunks must be nn.Module instances.")
    return chunks


def _mfsdp_chunks(model: nn.Module | Iterable[nn.Module]) -> list[nn.Module] | None:
    if isinstance(model, nn.ModuleList):
        chunks = list(model)
    else:
        chunks = _model_chunks(model)
    if chunks and all(
        hasattr(chunk, "param_and_grad_buffer")
        and hasattr(chunk, "named_optimizer_parameters")
        for chunk in chunks
    ):
        return chunks
    return None


def _mfsdp_optimizer_parts(optimizer):
    inner = getattr(optimizer, "_inner_optimizer", None)
    torch_optimizer = getattr(inner, "optimizer", None)
    params = getattr(inner, "params", None)
    if inner is None or torch_optimizer is None or params is None:
        raise TypeError(
            "M-FSDP DCP requires the standalone MFSdpOptimizer state interface."
        )
    return inner, torch_optimizer


def _mfsdp_bucket_identity(bucket) -> bytes:
    layout = [
        (spec.name, tuple(spec.shape), int(spec.full_offset), int(spec.numel))
        for spec in bucket.specs
    ]
    payload = repr(
        (
            int(bucket.bucket_id),
            int(bucket.logical_numel),
            int(bucket.chunk_size_factor),
            bool(bucket.is_expert),
            layout,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _mfsdp_bucket_mesh(bucket) -> DeviceMesh:
    if not dist.is_initialized():
        raise RuntimeError("M-FSDP DCP requires torch.distributed initialization.")
    group = bucket.process_group or dist.group.WORLD
    return DeviceMesh.from_group(group, bucket.device.type)


def _mfsdp_standard_shard_range(logical_numel: int, world_size: int, rank: int):
    chunk_size = (logical_numel + world_size - 1) // world_size
    start = min(rank * chunk_size, logical_numel)
    end = min(start + chunk_size, logical_numel)
    return start, end


def _mfsdp_gather_padded_bucket(bucket, local: torch.Tensor) -> torch.Tensor:
    if local.numel() != bucket.local_numel:
        raise RuntimeError(
            f"M-FSDP bucket {bucket.bucket_id} local tensor has {local.numel()} "
            f"elements, expected {bucket.local_numel}."
        )
    communication_local = local.detach().to(bucket.device)
    if bucket.world_size == 1:
        return communication_local.clone()
    full = torch.empty(
        bucket.full_numel,
        dtype=communication_local.dtype,
        device=bucket.device,
    )
    dist.all_gather_into_tensor(
        full, communication_local.contiguous(), group=bucket.process_group
    )
    return full


def _mfsdp_logical_dtensor(bucket, local_padded: torch.Tensor) -> DTensor:
    full = _mfsdp_gather_padded_bucket(bucket, local_padded)
    start, end = _mfsdp_standard_shard_range(
        bucket.logical_numel, bucket.world_size, bucket.rank
    )
    local = full.narrow(0, start, end - start).clone()
    return DTensor.from_local(
        local,
        _mfsdp_bucket_mesh(bucket),
        [Shard(0)],
        run_check=False,
        shape=torch.Size((bucket.logical_numel,)),
        stride=(1,),
    )


def _mfsdp_replicated_dtensor(bucket, tensor: torch.Tensor) -> DTensor:
    tensor = tensor.to(bucket.device)
    return DTensor.from_local(
        tensor,
        _mfsdp_bucket_mesh(bucket),
        [Replicate()],
        run_check=False,
        shape=tensor.shape,
        stride=tensor.stride(),
    )


def _mfsdp_param_optimizer_state(
    inner, param, *, create: bool = False
) -> dict[str, Any]:
    cpu_group = getattr(inner, "cpu_group", None)
    if cpu_group is not None and cpu_group.owns_param(param):
        return cpu_group.checkpoint_state(param)
    torch_optimizer = getattr(inner, "optimizer", inner)
    if create:
        return torch_optimizer.state.setdefault(param, {})
    return torch_optimizer.state.get(param, {})


def _mfsdp_pack_optimizer_tensor(bucket, inner, state_name: str):
    packed = torch.zeros(
        bucket.local_numel,
        dtype=torch.float32,
        device=bucket.device,
    )
    for spec in bucket.specs:
        if spec.shard_param is None or not spec.full_param.requires_grad:
            continue
        value = _mfsdp_param_optimizer_state(inner, spec.shard_param).get(state_name)
        if value is None and state_name == "master_param":
            # A partial-offload checkpoint may be restored with another DP
            # size, moving the deterministic CPU/GPU split boundary.  GPU-owned
            # shards already are the FP32 master, so save them in the common
            # master field as well instead of leaving an unrecoverable zero.
            value = spec.shard_param.detach()
        if value is None:
            continue
        if not torch.is_tensor(value) or value.numel() != spec.shard_numel:
            raise RuntimeError(
                f"M-FSDP optimizer state {state_name!r} for {spec.name!r} "
                "does not match its parameter shard."
            )
        packed.narrow(0, spec.local_offset, spec.shard_numel).copy_(
            value.detach().reshape(-1).to(dtype=packed.dtype)
        )
    return packed


def _mfsdp_optimizer_metadata(bucket, inner):
    initialized = torch.zeros(
        len(bucket.specs), dtype=torch.uint8, device=bucket.device
    )
    step_present = torch.zeros_like(initialized)
    steps = torch.zeros(
        len(bucket.specs), dtype=torch.float64, device=bucket.device
    )
    for index, spec in enumerate(bucket.specs):
        if spec.shard_param is None or not spec.full_param.requires_grad:
            continue
        state = _mfsdp_param_optimizer_state(inner, spec.shard_param)
        if not state:
            continue
        initialized[index] = 1
        if "step" in state:
            step_present[index] = 1
            step = state["step"]
            steps[index] = float(step.item() if torch.is_tensor(step) else step)
    if bucket.world_size > 1:
        # Empty local parameter shards intentionally have no FusedAdam state.
        # Metadata is nevertheless checkpointed as Replicate(), so make it
        # genuinely rank-invariant: a parameter is initialized when any DP
        # rank owns state, and all initialized shards share the same schema and
        # step value.
        dist.all_reduce(initialized, op=dist.ReduceOp.MAX, group=bucket.process_group)
        dist.all_reduce(step_present, op=dist.ReduceOp.MAX, group=bucket.process_group)
        dist.all_reduce(steps, op=dist.ReduceOp.MAX, group=bucket.process_group)
    return initialized, step_present, steps


_MFSDP_GROUP_STEP_PRESENT = "_mfsdp_group_step_present"


def _mfsdp_optimizer_param_groups(torch_optimizer) -> list[dict[str, Any]]:
    """Return DP-independent optimizer-group options without parameter objects."""
    saved_groups = []
    for group in torch_optimizer.param_groups:
        saved = {
            key: copy.deepcopy(value) for key, value in group.items() if key != "params"
        }
        if _MFSDP_GROUP_STEP_PRESENT in saved:
            raise RuntimeError(
                f"Optimizer parameter group uses reserved M-FSDP key "
                f"{_MFSDP_GROUP_STEP_PRESENT!r}."
            )
        # TE FusedAdam creates its group-level update counter lazily on the
        # first step. DCP only loads keys present in the destination template,
        # so always request a fixed ``step`` slot and separately preserve
        # whether the source optimizer actually owned that key.
        saved[_MFSDP_GROUP_STEP_PRESENT] = "step" in saved
        saved.setdefault("step", 0)
        saved_groups.append(saved)
    return saved_groups


def _mfsdp_restore_optimizer_param_groups(
    torch_optimizer, saved_groups: list[dict[str, Any]]
) -> None:
    if len(saved_groups) != len(torch_optimizer.param_groups):
        raise RuntimeError(
            "M-FSDP checkpoint optimizer parameter-group count does not match "
            "the target optimizer."
        )
    for group, saved_group in zip(
        torch_optimizer.param_groups, saved_groups, strict=True
    ):
        saved = copy.deepcopy(saved_group)
        step_present = bool(saved.pop(_MFSDP_GROUP_STEP_PRESENT, False))
        if not step_present:
            saved.pop("step", None)
        params = group["params"]
        group.clear()
        group.update(saved)
        group["params"] = params


def _mfsdp_optimizer_group_state(inner) -> list[dict[str, Any]] | dict[str, Any]:
    gpu = _mfsdp_optimizer_param_groups(inner.optimizer)
    if inner.cpu_group is None:
        return gpu
    return {"gpu": gpu, "cpu": inner.cpu_group.checkpoint_metadata()}


def _mfsdp_restore_optimizer_group_state(
    inner, saved: list[dict[str, Any]] | dict[str, Any]
) -> None:
    if inner.cpu_group is None:
        if not isinstance(saved, list):
            raise RuntimeError(
                "M-FSDP checkpoint contains CPU optimizer metadata but the "
                "target optimizer has no CPU-offloaded group."
            )
        _mfsdp_restore_optimizer_param_groups(inner.optimizer, saved)
        return
    if not isinstance(saved, dict) or "gpu" not in saved or "cpu" not in saved:
        raise RuntimeError(
            "M-FSDP CPU-offloaded DCP requires both GPU and CPU optimizer metadata."
        )
    _mfsdp_restore_optimizer_param_groups(inner.optimizer, saved["gpu"])
    inner.cpu_group.load_checkpoint_metadata(saved["cpu"])


def _mfsdp_domain_key(bucket, ps) -> str:
    if ps is None:
        if dist.get_world_size() != bucket.world_size:
            raise ValueError(
                "M-FSDP cross-DP DCP requires ParallelState to distinguish "
                "fixed model-parallel coordinates."
            )
        return "expert.pp0.ep0.etp0" if bucket.is_expert else "dense.pp0.cp0.tp0"
    if bucket.is_expert:
        return (
            f"expert.pp{int(getattr(ps, 'pp_rank', 0))}."
            f"ep{int(getattr(ps, 'ep_rank', 0))}."
            f"etp{int(getattr(ps, 'etp_rank', 0))}"
        )
    return (
        f"dense.pp{int(getattr(ps, 'pp_rank', 0))}."
        f"cp{int(getattr(ps, 'cp_rank', 0))}."
        f"tp{int(getattr(ps, 'tp_rank', 0))}"
    )


def _mfsdp_checkpoint_template(
    chunks,
    optimizer,
    ps,
    *,
    include_model: bool,
    include_optimizer: bool,
):
    inner = None
    if include_optimizer:
        inner, _torch_optimizer = _mfsdp_optimizer_parts(optimizer)
    domains: dict[str, dict[str, dict[str, Any]]] = {}
    for chunk_index, chunk in enumerate(chunks):
        for bucket in chunk.param_and_grad_buffer.buckets:
            identity = torch.tensor(
                list(_mfsdp_bucket_identity(bucket)),
                dtype=torch.uint8,
                device=bucket.main_param_buffer.device,
            )
            bucket_state = {
                "identity": _mfsdp_replicated_dtensor(bucket, identity),
            }
            if include_model:
                bucket_state["main_param"] = _mfsdp_logical_dtensor(
                    bucket, bucket.main_param_buffer
                )
            if include_optimizer and bucket.requires_grad:
                assert inner is not None
                initialized, step_present, steps = _mfsdp_optimizer_metadata(
                    bucket, inner
                )
                bucket_state["state_initialized"] = _mfsdp_replicated_dtensor(
                    bucket, initialized
                )
                bucket_state["step_present"] = _mfsdp_replicated_dtensor(
                    bucket, step_present
                )
                bucket_state["step"] = _mfsdp_replicated_dtensor(bucket, steps)
                bucket_state["exp_avg"] = _mfsdp_logical_dtensor(
                    bucket,
                    _mfsdp_pack_optimizer_tensor(bucket, inner, "exp_avg"),
                )
                bucket_state["exp_avg_sq"] = _mfsdp_logical_dtensor(
                    bucket,
                    _mfsdp_pack_optimizer_tensor(bucket, inner, "exp_avg_sq"),
                )
                if inner.cpu_group is not None:
                    bucket_state["master_param"] = _mfsdp_logical_dtensor(
                        bucket,
                        _mfsdp_pack_optimizer_tensor(
                            bucket, inner, "master_param"
                        ),
                    )
            domain = domains.setdefault(_mfsdp_domain_key(bucket, ps), {})
            chunk_state = domain.setdefault(str(chunk_index), {})
            chunk_state[str(bucket.bucket_id)] = bucket_state
    return {"version": 3, "domains": domains}


def _save_mfsdp_checkpoint(
    chunks,
    optimizer,
    step: int,
    path: str,
    *,
    ps,
    save_model: bool,
    save_optimizer: bool,
) -> None:
    if save_optimizer and optimizer is None:
        raise ValueError("M-FSDP optimizer checkpointing requires an optimizer.")
    state_dict: dict[str, Any] = {"step": int(step)}
    if save_model or save_optimizer:
        state_dict["mfsdp"] = _mfsdp_checkpoint_template(
            chunks,
            optimizer,
            ps,
            include_model=save_model,
            include_optimizer=save_optimizer,
        )
    if save_optimizer:
        inner, _torch_optimizer = _mfsdp_optimizer_parts(optimizer)
        state_dict["optimizer_param_groups"] = _mfsdp_optimizer_group_state(inner)
    dcp.save(state_dict, checkpoint_id=path)


def _mfsdp_copy_logical_to_bucket(bucket, logical: torch.Tensor, target: torch.Tensor):
    target.zero_()
    local_start = bucket.rank * bucket.local_numel
    local_end = min(local_start + bucket.local_numel, bucket.logical_numel)
    if local_end > local_start:
        source = logical.narrow(0, local_start, local_end - local_start)
        target.narrow(0, 0, local_end - local_start).copy_(
            source.to(device=target.device, dtype=target.dtype)
        )


def _mfsdp_unpack_optimizer_tensor(
    bucket,
    inner,
    logical: torch.Tensor,
    state_name: str,
    initialized: torch.Tensor,
    step_present: torch.Tensor,
    steps: torch.Tensor,
) -> None:
    local = torch.zeros(
        bucket.local_numel,
        dtype=logical.dtype,
        device=logical.device,
    )
    _mfsdp_copy_logical_to_bucket(bucket, logical, local)
    for index, spec in enumerate(bucket.specs):
        if spec.shard_param is None or not spec.full_param.requires_grad:
            continue
        if state_name == "master_param" and (
            inner.cpu_group is None
            or not inner.cpu_group.owns_param(spec.shard_param)
        ):
            continue
        if not bool(initialized[index].item()):
            if inner.cpu_group is None or not inner.cpu_group.owns_param(
                spec.shard_param
            ):
                inner.optimizer.state.pop(spec.shard_param, None)
            continue
        state = _mfsdp_param_optimizer_state(inner, spec.shard_param, create=True)
        value = (
            local.narrow(0, spec.local_offset, spec.shard_numel)
            .view_as(spec.shard_param)
        )
        current = state.get(state_name)
        if torch.is_tensor(current):
            current.copy_(value.to(device=current.device, dtype=current.dtype))
        else:
            state[state_name] = value.to(spec.shard_param.device).clone()
        if state_name == "master_param":
            master = state[state_name]
            with torch.no_grad():
                spec.shard_param.copy_(
                    master.to(
                        device=spec.shard_param.device,
                        dtype=spec.shard_param.dtype,
                    )
                )
        if bool(step_present[index].item()):
            step_value = float(steps[index].item())
            current_step = state.get("step")
            if torch.is_tensor(current_step):
                current_step.fill_(step_value)
            elif current_step is not None:
                state["step"] = int(step_value)
            else:
                state["step"] = torch.tensor(
                    step_value,
                    dtype=torch.float32,
                    device=spec.shard_param.device,
                )
        else:
            state.pop("step", None)


def _load_mfsdp_checkpoint(
    chunks, optimizer, path: str, *, ps, load_model: bool, load_optimizer: bool
) -> int:
    if not load_model and not load_optimizer:
        state_dict: dict[str, Any] = {"step": 0}
        dcp.load(state_dict, checkpoint_id=path)
        return int(state_dict.get("step", 0))
    if load_optimizer and optimizer is None:
        raise ValueError("M-FSDP optimizer checkpoint loading requires an optimizer.")
    state_dict: dict[str, Any] = {
        "step": 0,
        "mfsdp": _mfsdp_checkpoint_template(
            chunks,
            optimizer,
            ps,
            include_model=load_model,
            include_optimizer=load_optimizer,
        ),
    }
    inner = None
    if load_optimizer:
        inner, _torch_optimizer = _mfsdp_optimizer_parts(optimizer)
        state_dict["optimizer_param_groups"] = _mfsdp_optimizer_group_state(inner)
    dcp.load(state_dict, checkpoint_id=path)
    if load_optimizer:
        assert inner is not None
        _mfsdp_restore_optimizer_group_state(
            inner, state_dict["optimizer_param_groups"]
        )
    loaded_domains = state_dict["mfsdp"]["domains"]
    for chunk_index, chunk in enumerate(chunks):
        for bucket in chunk.param_and_grad_buffer.buckets:
            loaded_bucket = loaded_domains[_mfsdp_domain_key(bucket, ps)][
                str(chunk_index)
            ][str(bucket.bucket_id)]
            loaded_identity = bytes(loaded_bucket["identity"].to_local().cpu().tolist())
            expected_identity = _mfsdp_bucket_identity(bucket)
            if loaded_identity != expected_identity:
                raise RuntimeError(
                    f"M-FSDP checkpoint bucket layout mismatch for bucket "
                    f"{bucket.bucket_id}."
                )
            if load_model:
                logical_param = loaded_bucket["main_param"].full_tensor()
                _mfsdp_copy_logical_to_bucket(
                    bucket, logical_param, bucket.main_param_buffer
                )
                bucket.copy_main_weights_to_model_weights()
                bucket.invalidate_full_parameters()
            if load_optimizer and bucket.requires_grad:
                assert inner is not None
                initialized = loaded_bucket["state_initialized"].to_local()
                step_present = loaded_bucket["step_present"].to_local()
                steps = loaded_bucket["step"].to_local()
                for state_name in ("exp_avg", "exp_avg_sq"):
                    logical_state = loaded_bucket[state_name].full_tensor()
                    _mfsdp_unpack_optimizer_tensor(
                        bucket,
                        inner,
                        logical_state,
                        state_name,
                        initialized,
                        step_present,
                        steps,
                    )
                if "master_param" in loaded_bucket:
                    logical_master = loaded_bucket["master_param"].full_tensor()
                    _mfsdp_unpack_optimizer_tensor(
                        bucket,
                        inner,
                        logical_master,
                        "master_param",
                        initialized,
                        step_present,
                        steps,
                    )
    return int(state_dict.get("step", 0))


def _to_local_tensor(tensor: Any) -> torch.Tensor:
    local_tensor = getattr(tensor, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def _is_dtensor_like(tensor: Any) -> bool:
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def _dcp_tensor_from_param(param: torch.Tensor, mesh: DeviceMesh, placements: list) -> DTensor:
    if _is_dtensor_like(param):
        return _dtensor_from_dtensor_like_param(param, _to_local_tensor(param).detach())
    return DTensor.from_local(_to_local_tensor(param).detach(), mesh, placements)


def _empty_dcp_tensor_like_param(
    param: torch.Tensor, mesh: DeviceMesh, placements: list
) -> DTensor:
    if _is_dtensor_like(param):
        return _dtensor_from_dtensor_like_param(param, torch.empty_like(_to_local_tensor(param)))
    return DTensor.from_local(torch.empty_like(_to_local_tensor(param)), mesh, placements)


def _dtensor_from_dtensor_like_param(param: torch.Tensor, local_tensor: torch.Tensor) -> DTensor:
    return DTensor.from_local(
        local_tensor,
        param.device_mesh,
        param.placements,
        shape=tuple(param.shape),
        stride=tuple(param.stride()),
    )


def _copy_tensor_(target: torch.Tensor, src: torch.Tensor) -> None:
    local_target = _to_local_tensor(target)
    local_src = _to_local_tensor(src).to(device=local_target.device, dtype=local_target.dtype)
    if isinstance(local_target, torch.Tensor) and local_target is not target:
        local_target.copy_(local_src)
    else:
        target.copy_(local_src)


def _chunk_tensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        state[f"param.{name}"] = _to_local_tensor(param.detach()).cpu().clone()
    for name, buffer in module.named_buffers():
        state[f"buffer.{name}"] = _to_local_tensor(buffer.detach()).cpu().clone()
    return state


def _load_chunk_tensor_state(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    missing: list[str] = []
    for key, src in state.items():
        kind, name = key.split(".", 1)
        if kind == "param" and name in params:
            with torch.no_grad():
                _copy_tensor_(params[name], src)
        elif kind == "buffer" and name in buffers:
            with torch.no_grad():
                _copy_tensor_(buffers[name], src)
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(f"checkpoint contains unknown tensor keys: {missing}")


def _local_checkpoint_file(path: str | os.PathLike[str]) -> Path:
    ckpt_path = Path(path)
    if ckpt_path.is_dir() or ckpt_path.suffix == "":
        if _is_distributed_checkpoint_ranked():
            return ckpt_path / f"training_state_{_rank_suffix()}.pt"
        return ckpt_path / "training_state.pt"
    return ckpt_path


def _local_optimizer_parameter_state_file(ckpt_file: Path) -> Path:
    return ckpt_file.with_name(f"{ckpt_file.stem}.optimizer_parameter_state{ckpt_file.suffix}")


def _rank_suffix() -> str:
    if dist.is_available() and dist.is_initialized():
        return f"rank_{dist.get_rank():05d}"
    return "rank_00000"


def _is_distributed_checkpoint_ranked() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rng_sidecar_file(path: str | os.PathLike[str]) -> Path:
    return Path(path) / f"rng_state_{_rank_suffix()}.pt"


def _cpu_clone(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


def _get_cuda_rng_state() -> torch.Tensor | None:
    if not torch.cuda.is_initialized():
        return None
    return _cpu_clone(torch.cuda.get_rng_state())


def _get_cuda_rng_tracker_states() -> dict[str, torch.Tensor]:
    if not torch.cuda.is_initialized():
        return {}

    from megatron.core import tensor_parallel

    states = tensor_parallel.get_cuda_rng_tracker().get_states()
    return {name: _cpu_clone(state) for name, state in states.items() if state is not None}


def _get_rng_state() -> dict[str, Any]:
    return {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": _cpu_clone(torch.get_rng_state()),
        "cuda_rng_state": _get_cuda_rng_state(),
        "rng_tracker_states": _get_cuda_rng_tracker_states(),
    }


def _restore_cuda_rng_tracker_states(states: dict[str, torch.Tensor]) -> None:
    if not states or not torch.cuda.is_initialized():
        return
    try:
        from megatron.core import tensor_parallel

        tracker = tensor_parallel.get_cuda_rng_tracker()
        graph_safe = tensor_parallel.is_graph_safe_cuda_rng_tracker(tracker)
        restored = {
            name: tensor_parallel.convert_cuda_rng_state(state, to_graphable=graph_safe)
            for name, state in states.items()
        }
        tracker.set_states(restored)
    except Exception as exc:
        raise RuntimeError("Failed to restore Megatron tensor-parallel RNG tracker state.") from exc


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_state = state.get("cuda_rng_state")
    if cuda_rng_state is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state(cuda_rng_state)
    _restore_cuda_rng_tracker_states(state.get("rng_tracker_states", {}))


def _save_rng_sidecar(path: str | os.PathLike[str]) -> None:
    rng_file = _rng_sidecar_file(path)
    rng_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_get_rng_state(), rng_file)


def _load_rng_sidecar(path: str | os.PathLike[str]) -> None:
    rng_file = _rng_sidecar_file(path)
    if not rng_file.exists():
        log_rank0(f"RNG sidecar not found at {rng_file}; skipping RNG restore.")
        return
    _restore_rng_state(torch.load(rng_file, map_location="cpu", weights_only=False))


def _save_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int,
    path: str,
    *,
    save_rng: bool = True,
) -> None:
    chunks = _model_chunks(model)
    ckpt_file = _local_checkpoint_file(path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    save_parameter_state = getattr(optimizer, "save_parameter_state", None)
    optimizer_parameter_state_file = (
        _local_optimizer_parameter_state_file(ckpt_file) if callable(save_parameter_state) else None
    )
    state = {
        "format": "megatron_lite.local_training.v1",
        "step": int(step),
        "model": [_chunk_tensor_state(chunk) for chunk in chunks],
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_parameter_state": (
            optimizer_parameter_state_file.name
            if optimizer_parameter_state_file is not None
            else None
        ),
        "rng_state": _get_rng_state() if save_rng else None,
    }
    torch.save(state, ckpt_file)
    if optimizer_parameter_state_file is not None:
        save_parameter_state(str(optimizer_parameter_state_file))
    log_rank0(f"Saved local training checkpoint at step {step} to {ckpt_file}")


def _load_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    *,
    load_rng: bool = True,
    load_parameter_state_update_legacy_format: bool = False,
) -> int:
    ckpt_file = _local_checkpoint_file(path)
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if state.get("format") != "megatron_lite.local_training.v1":
        raise RuntimeError(f"Unsupported local checkpoint format in {ckpt_file}")
    chunks = _model_chunks(model)
    chunk_states = state.get("model")
    if not isinstance(chunk_states, list) or len(chunk_states) != len(chunks):
        raise RuntimeError("Checkpoint model chunk count does not match target model.")
    for chunk, chunk_state in zip(chunks, chunk_states, strict=True):
        _load_chunk_tensor_state(chunk, chunk_state)
        sync_full_parameters_to_shards = getattr(
            chunk, "sync_full_parameters_to_shards", None
        )
        if callable(sync_full_parameters_to_shards):
            sync_full_parameters_to_shards()
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
        parameter_state_name = state.get("optimizer_parameter_state")
        load_parameter_state = getattr(optimizer, "load_parameter_state", None)
        if parameter_state_name is not None and callable(load_parameter_state):
            load_parameter_state(
                str(ckpt_file.with_name(parameter_state_name)),
                update_legacy_format=load_parameter_state_update_legacy_format,
            )
        else:
            reload_model_params = getattr(optimizer, "reload_model_params", None)
            if callable(reload_model_params):
                reload_model_params()
    if load_rng:
        _restore_rng_state(state.get("rng_state"))
    step = int(state.get("step", 0))
    log_rank0(f"Loaded local training checkpoint from {ckpt_file} at step {step}")
    return step


def _build_meshes(config):
    """Build separate meshes for dense and expert parameters.

    Dense mesh  [PP, DP, CP, TP]  — matches init_parallel dense decomposition.
    Expert mesh [PP, EDP, EP, ETP] — matches init_parallel expert decomposition.

    Both meshes use C-order layout so the innermost (rightmost) dimension
    corresponds to the fastest-changing rank index, consistent with
    init_parallel's rank = (...) * inner_size + inner_rank formula.
    """
    ws = dist.get_world_size()
    tp = int(config.tp or 1)
    ep = int(config.ep or 1)
    etp = max(int(config.etp or 1), 1)
    cp = max(int(config.cp or 1), 1)
    pp = max(int(config.pp or 1), 1)

    dense_dp = ws // (tp * cp * pp)
    expert_dp = ws // (etp * ep * pp)

    ranks = torch.arange(ws)
    dense_mesh = DeviceMesh("cuda", ranks.reshape(pp, dense_dp, cp, tp))
    expert_mesh = DeviceMesh("cuda", ranks.reshape(pp, expert_dp, ep, etp))
    return dense_mesh, expert_mesh


def log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[megatron.lite] {msg}", flush=True)


# ======================================================================
# QKV / FC1 canonicalize for DCP (interleaved-TP ↔ canonical layout)
# ======================================================================


def _ag(data, size, group, dim=0):
    from megatron.lite.primitive.ckpt.hf_weights import allgather_concat

    return allgather_concat(data, size, group, dim)


def canonicalize_qkv_for_dcp(model, num_attention_heads, num_key_value_heads, head_dim, ps):
    """Rearrange fused QKV from interleaved-TP to canonical (Q|K|V) for DCP save."""
    if ps.tp_size <= 1:
        return
    from megatron.lite.primitive.utils import ensure_divisible

    nq = ensure_divisible(num_attention_heads, ps.tp_size) * head_dim
    nkv = ensure_divisible(num_key_value_heads, ps.tp_size) * head_dim
    for name, param in model.named_parameters():
        if "qkv" not in name or "layer_norm" in name:
            continue
        full = _ag(param.data, ps.tp_size, ps.tp_group)
        cs = param.data.shape[0]
        q, k, v = [], [], []
        for r in range(ps.tp_size):
            s = full[r * cs : (r + 1) * cs]
            q.append(s[:nq])
            k.append(s[nq : nq + nkv])
            v.append(s[nq + nkv :])
        canon = torch.cat([torch.cat(q), torch.cat(k), torch.cat(v)], dim=0)
        param.data.copy_(canon.chunk(ps.tp_size, dim=0)[ps.tp_rank])


def decanon_qkv_after_dcp(model, num_attention_heads, num_key_value_heads, head_dim, ps):
    """Reverse of canonicalize_qkv_for_dcp."""
    if ps.tp_size <= 1:
        return
    qs = num_attention_heads * head_dim
    kvs = num_key_value_heads * head_dim
    for name, param in model.named_parameters():
        if "qkv" not in name or "layer_norm" in name:
            continue
        full = _ag(param.data, ps.tp_size, ps.tp_group)
        ql = full[:qs].chunk(ps.tp_size)[ps.tp_rank]
        kl = full[qs : qs + kvs].chunk(ps.tp_size)[ps.tp_rank]
        vl = full[qs + kvs :].chunk(ps.tp_size)[ps.tp_rank]
        param.data.copy_(torch.cat([ql, kl, vl], dim=0))


def canonicalize_fc1_for_dcp(model, ps):
    """Rearrange fused gate-up FC1 from interleaved-ETP to canonical for DCP save."""
    if ps.etp_size <= 1:
        return
    for name, param in model.named_parameters():
        if "experts" not in name or "fc1" not in name:
            continue
        full = _ag(param.data, ps.etp_size, ps.etp_group)
        cs = param.data.shape[0]
        ffn = cs // 2
        g, u = [], []
        for r in range(ps.etp_size):
            s = full[r * cs : (r + 1) * cs]
            g.append(s[:ffn])
            u.append(s[ffn:])
        canon = torch.cat([torch.cat(g), torch.cat(u)], dim=0)
        param.data.copy_(canon.chunk(ps.etp_size, dim=0)[ps.etp_rank])


def decanon_fc1_after_dcp(model, ps):
    """Reverse of canonicalize_fc1_for_dcp."""
    if ps.etp_size <= 1:
        return
    for name, param in model.named_parameters():
        if "experts" not in name or "fc1" not in name:
            continue
        full = _ag(param.data, ps.etp_size, ps.etp_group)
        ffn = full.shape[0] // 2
        gl = full[:ffn].chunk(ps.etp_size)[ps.etp_rank]
        ul = full[ffn:].chunk(ps.etp_size)[ps.etp_rank]
        param.data.copy_(torch.cat([gl, ul], dim=0))


__all__ = [
    "canonicalize_fc1_for_dcp",
    "canonicalize_qkv_for_dcp",
    "decanon_fc1_after_dcp",
    "decanon_qkv_after_dcp",
    "load_training_checkpoint",
    "save_training_checkpoint",
]

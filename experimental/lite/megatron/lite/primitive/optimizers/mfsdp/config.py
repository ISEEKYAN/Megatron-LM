# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Standalone Megatron-FSDP config lowering and validation."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

_SUPPORTED_OPTIMIZERS = {"adam", "sgd"}
_REQUIRED_DDP_KNOB_VALUES = {
    "use_distributed_optimizer": True,
    "use_megatron_fsdp": True,
}
_UNSUPPORTED_OPTIMIZATION_KNOBS = {"use_hsdp", "hsdp"}


@dataclass(frozen=True, slots=True)
class MFSDPConfig:
    """Narrow configuration owned by the standalone M-FSDP primitive.

    The fields are the subset of Megatron-FSDP that MLite exercises.  Keeping
    this contract local avoids importing the much broader MCore DDP config and
    makes unsupported sharding modes fail before any buffers are allocated.
    """

    sharding_strategy: str = "optim_grads_params"
    bucket_size: int | None = 40_000_000
    suggested_communication_unit_size: int | None = None
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    average_gradients: bool = True
    calculate_per_token_loss: bool = False
    main_params_dtype: torch.dtype = torch.float32
    main_grads_dtype: torch.dtype = torch.float32
    grad_comm_dtype: torch.dtype = torch.float32
    use_decoupled_grad: bool = False
    gradient_accumulation_fusion: bool = False
    nccl_ub: bool = False
    fsdp_double_buffer: bool = False
    maxpool_double_buffer: bool = False
    fsdp_manual_registration: bool = False
    disable_symmetric_registration: bool = False
    all_gather_in_start_param_sync: bool = True


@dataclass(frozen=True, slots=True)
class MixedPrecisionPolicy:
    """Dtype policy for compute, persistent main parameters, and gradients."""

    compute_dtype: torch.dtype
    main_params_dtype: torch.dtype = torch.float32
    main_grads_dtype: torch.dtype = torch.float32
    grad_comm_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        for name in (
            "compute_dtype",
            "main_params_dtype",
            "main_grads_dtype",
            "grad_comm_dtype",
        ):
            dtype = getattr(self, name)
            if not torch.empty((), dtype=dtype).is_floating_point():
                raise ValueError(f"M-FSDP {name} must be floating point, got {dtype}.")


@dataclass(frozen=True, slots=True)
class MFSDPProcessGroups:
    """Process groups explicitly supplied by MLite composition."""

    dense_dp: dist.ProcessGroup | None
    expert_dp: dist.ProcessGroup | None
    dense_ag: dist.ProcessGroup | None
    expert_ag: dist.ProcessGroup | None
    tp: dist.ProcessGroup | None
    etp: dist.ProcessGroup | None
    ep: dist.ProcessGroup | None
    pp: dist.ProcessGroup | None

    def data_group(self, *, expert: bool) -> dist.ProcessGroup | None:
        return self.expert_dp if expert else self.dense_dp

    def gather_group(self, *, expert: bool) -> dist.ProcessGroup | None:
        return self.expert_ag if expert else self.dense_ag

    def registration_groups(self) -> tuple[dist.ProcessGroup, ...]:
        groups: list[dist.ProcessGroup] = []
        for group in (self.dense_dp, self.expert_dp, self.dense_ag, self.expert_ag):
            if group is not None and all(group is not existing for existing in groups):
                groups.append(group)
        return tuple(groups)


def build_mfsdp_process_groups(ps: Any) -> MFSDPProcessGroups:
    """Read groups already owned by MLite; never initialize global MCore state."""
    dense_dp = getattr(ps, "dp_cp_group", None) or getattr(ps, "dp_group", None)
    expert_dp = getattr(ps, "ep_dp_group", None) or dense_dp
    dense_ag = getattr(ps, "dp_cp_ag_group", None) or getattr(ps, "dp_ag_group", None)
    expert_ag = getattr(ps, "ep_dp_ag_group", None)
    return MFSDPProcessGroups(
        dense_dp=dense_dp,
        expert_dp=expert_dp,
        dense_ag=dense_ag or dense_dp,
        expert_ag=expert_ag or expert_dp,
        tp=getattr(ps, "tp_group", None),
        etp=getattr(ps, "etp_group", None),
        ep=getattr(ps, "ep_group", None),
        pp=getattr(ps, "pp_group", None),
    )


def group_size(group: dist.ProcessGroup | None) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def group_rank(group: dist.ProcessGroup | None) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group)


SKIP_TP_DUPLICATE_SYNC_ATTR = "_mlite_mfsdp_skip_tp_duplicate_sync"
PARAM_NAME_ATTR = "_mlite_mfsdp_param_name"


def annotate_parallel_parameters(
    module: nn.Module, is_expert: Callable[[str], bool], *, tp_size: int, etp_size: int
) -> None:
    """Normalize parameter ownership from topology and explicit attributes."""
    sequence_parallel_ids = {id(param) for param in getattr(module, "sp_params", ())}
    for name, param in module.named_parameters():
        setattr(param, PARAM_NAME_ATTR, name)
        expert = bool(is_expert(name))
        param._mfsdp_is_expert = expert
        if not hasattr(param, "allreduce"):
            param.allreduce = not expert

        replicated = bool(
            id(param) in sequence_parallel_ids
            or getattr(param, "sequence_parallel", False)
            or getattr(param, "average_gradients_across_tp_domain", False)
            or getattr(param, "shared", False)
        )
        if id(param) in sequence_parallel_ids:
            param.sequence_parallel = True
        active_tp_size = max(int(etp_size if expert else tp_size), 1)
        explicitly_sharded = bool(getattr(param, "tensor_model_parallel", False))
        sharded = (
            not replicated
            and param.ndim > 1
            and (explicitly_sharded or active_tp_size > 1)
        )
        param.tensor_model_parallel = sharded
        if sharded:
            setattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR, True)
        elif hasattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR):
            delattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR)


def build_mfsdp_config(
    opt: Any, *, calculate_per_token_loss: bool = False
) -> MFSDPConfig:
    """Lower runtime optimizer options into the supported M-FSDP surface."""
    values = dict(getattr(opt, "override_optimizer_config", None) or {})

    def option(name: str, default: Any, *aliases: str) -> Any:
        for key in (name, *aliases):
            if key in values:
                return values[key]
            if hasattr(opt, key):
                value = getattr(opt, key)
                if value is not None:
                    return value
        return default

    strategy = str(
        option(
            "mfsdp_sharding_strategy",
            "optim_grads_params",
            "data_parallel_sharding_strategy",
            "megatron_fsdp_sharding_strategy",
        )
    )
    if strategy != "optim_grads_params":
        raise ValueError(
            "The standalone M-FSDP path supports only "
            "data_parallel_sharding_strategy='optim_grads_params'."
        )
    raw_bucket_size = option("bucket_size", 40_000_000)
    bucket_size = None if raw_bucket_size is None else int(raw_bucket_size)
    if bucket_size is not None and bucket_size <= 0:
        raise ValueError("M-FSDP bucket_size must be a positive element count or None.")
    raw_communication_size = option("suggested_communication_unit_size", None)
    suggested_communication_unit_size = (
        None if raw_communication_size is None else int(raw_communication_size)
    )
    if (
        suggested_communication_unit_size is not None
        and suggested_communication_unit_size <= 0
    ):
        raise ValueError(
            "M-FSDP suggested_communication_unit_size must be positive or None."
        )

    main_params_dtype = _coerce_dtype(
        option("megatron_fsdp_main_params_dtype", torch.float32),
        name="megatron_fsdp_main_params_dtype",
    )
    main_grads_dtype = _coerce_dtype(
        option("megatron_fsdp_main_grads_dtype", torch.float32),
        name="megatron_fsdp_main_grads_dtype",
    )
    grad_comm_dtype = _coerce_dtype(
        option("megatron_fsdp_grad_comm_dtype", main_grads_dtype),
        name="megatron_fsdp_grad_comm_dtype",
    )
    use_decoupled_grad = _precision_aware_enabled(opt)
    if not use_decoupled_grad and (
        main_params_dtype is not torch.float32
        or main_grads_dtype is not torch.float32
    ):
        raise ValueError(
            "M-FSDP main parameter and main gradient dtypes must be FP32 when "
            "use_precision_aware_optimizer is disabled."
        )
    nccl_ub = bool(option("nccl_ub", False))
    maxpool_double_buffer = bool(
        option(
            "megatron_fsdp_max_pool_double_buffer",
            False,
            "maxpool_double_buffer",
        )
    )
    fsdp_manual_registration = bool(option("fsdp_manual_registration", False))
    if fsdp_manual_registration and not nccl_ub:
        raise ValueError(
            "M-FSDP fsdp_manual_registration requires nccl_ub=True."
        )
    return MFSDPConfig(
        sharding_strategy=strategy,
        bucket_size=bucket_size,
        suggested_communication_unit_size=suggested_communication_unit_size,
        overlap_grad_reduce=bool(option("overlap_grad_reduce", True)),
        overlap_param_gather=bool(option("overlap_param_gather", True)),
        average_gradients=(
            False
            if calculate_per_token_loss
            else bool(option("average_in_collective", True))
        ),
        calculate_per_token_loss=bool(calculate_per_token_loss),
        main_params_dtype=main_params_dtype,
        main_grads_dtype=main_grads_dtype,
        grad_comm_dtype=grad_comm_dtype,
        use_decoupled_grad=use_decoupled_grad,
        # The official Megatron-FSDP H100 and MoE recipes explicitly disable
        # fused wgrad accumulation.  Keep it available as an opt-in MCore
        # capability, but make the standalone recommended path explicit rather
        # than forcing the Megatron-LM CLI default onto every TE module.
        gradient_accumulation_fusion=bool(
            option("gradient_accumulation_fusion", False)
        ),
        nccl_ub=nccl_ub,
        # MCore keeps the additional communication residency opt-in.  NCCL user
        # buffers are the exception: registered allocations require alternating
        # slots and therefore force the bounded double-buffer path.
        fsdp_double_buffer=(
            bool(option("fsdp_double_buffer", False))
            or nccl_ub
            or maxpool_double_buffer
        ),
        maxpool_double_buffer=maxpool_double_buffer,
        fsdp_manual_registration=fsdp_manual_registration,
        disable_symmetric_registration=bool(
            option("disable_symmetric_registration", False)
        ),
        all_gather_in_start_param_sync=bool(
            option("fsdp_all_gather_in_start_param_sync", True)
        ),
    )


def _coerce_dtype(value: Any, *, name: str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.float32
    normalized = str(value).lower().removeprefix("torch.")
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if normalized not in mapping:
        raise ValueError(f"{name} has unsupported dtype {value!r}.")
    return mapping[normalized]


def validate_mfsdp_config(engine_cfg, *, has_optimizer_factory: bool = False) -> None:
    """Validate the supported surface for optimizer='megatron_fsdp'."""
    validate_mfsdp_topology(engine_cfg.parallel)
    opt = engine_cfg.optimizer
    validate_optimizer_name(
        getattr(opt, "optimizer", "adam"), has_optimizer_factory=has_optimizer_factory
    )
    validate_optimization_knobs(opt)
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        raise ValueError(
            "Megatron-FSDP requires CUDA_DEVICE_MAX_CONNECTIONS > 1 or unset."
        )


def validate_mfsdp_topology(parallel_cfg) -> None:
    """Validate supported topology surfaces for Megatron-FSDP."""
    pp = int(getattr(parallel_cfg, "pp", 1) or 1)
    vpp = int(getattr(parallel_cfg, "vpp", 1) or 1)

    if vpp > 1 and pp <= 1:
        raise ValueError("optimizer_impl='megatron_fsdp' requires pp>1 when vpp>1.")


def validate_optimizer_name(
    optimizer_name: str, *, has_optimizer_factory: bool = False
) -> None:
    if optimizer_name not in _SUPPORTED_OPTIMIZERS and not has_optimizer_factory:
        raise ValueError(
            "optimizer_impl='megatron_fsdp' supports only adam/sgd, "
            f"got {optimizer_name!r}; provide optimizer_factory for an optional "
            "algorithm such as Muon."
        )


def _precision_aware_enabled(opt) -> bool:
    precision_aware = getattr(opt, "use_precision_aware_optimizer", None)
    raw_overrides = dict(getattr(opt, "override_optimizer_config", None) or {})
    if "use_precision_aware_optimizer" in raw_overrides:
        precision_aware = raw_overrides["use_precision_aware_optimizer"]
    if precision_aware not in {None, True, False}:
        raise ValueError("use_precision_aware_optimizer must be a boolean when set.")
    return bool(precision_aware)


def validate_optimization_knobs(opt) -> None:
    """Validate optimizer and communication knobs before construction."""
    values = dict(getattr(opt, "override_optimizer_config", None) or {})
    for key in _UNSUPPORTED_OPTIMIZATION_KNOBS:
        value = values.get(key, getattr(opt, key, None))
        if _truthy_feature_value(value):
            raise ValueError(
                "optimizer_impl='megatron_fsdp' does not accept hsdp/use_hsdp aliases; "
                "set num_distributed_optimizer_instances explicitly."
            )
    instances = values.get(
        "num_distributed_optimizer_instances",
        getattr(opt, "num_distributed_optimizer_instances", None),
    )
    if _invalid_distributed_optimizer_instances(instances):
        raise ValueError(
            "optimizer_impl='megatron_fsdp' requires "
            "num_distributed_optimizer_instances to be a positive integer."
        )
    if instances is not None and str(instances).lower() not in {"", "none", "null"}:
        if int(instances) > 1:
            raise ValueError(
                "optimizer_impl='megatron_fsdp' does not support "
                "num_distributed_optimizer_instances>1."
            )
    nccl_ub = values.get("nccl_ub", getattr(opt, "nccl_ub", None))
    if _truthy_feature_value(nccl_ub) and _cuda_alloc_conf_expands_segments():
        raise ValueError(
            "optimizer_impl='megatron_fsdp' requires PYTORCH_CUDA_ALLOC_CONF without "
            "expandable_segments:True when nccl_ub=True; unset it before enabling UBR."
        )
    for key, required_value in _REQUIRED_DDP_KNOB_VALUES.items():
        value = values.get(key, getattr(opt, key, required_value))
        if value != required_value:
            raise ValueError(
                f"optimizer_impl='megatron_fsdp' requires {key}={required_value!r}."
            )
    if (
        _precision_aware_enabled(opt)
        and str(getattr(opt, "optimizer", "adam")) != "adam"
    ):
        raise ValueError(
            "M-FSDP precision-aware gradients require Adam with TransformerEngine "
            "FusedAdam decoupled-gradient support."
        )


def _truthy_feature_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, int | float) and value == 0:
        return False
    if isinstance(value, str) and value.lower() in {
        "",
        "0",
        "false",
        "none",
        "null",
        "off",
    }:
        return False
    return True


def _cuda_alloc_conf_expands_segments() -> bool:
    value = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    for item in value.split(","):
        key, _, raw = item.strip().partition(":")
        if key.lower() == "expandable_segments" and raw.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
    return False


def _invalid_distributed_optimizer_instances(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, int | float):
        return int(value) != value or value < 1
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"", "none", "null"}:
            return False
        try:
            return int(normalized) < 1
        except ValueError:
            return True
    return True

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Standalone Megatron-FSDP config lowering and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

_SUPPORTED_OPTIMIZERS = {"adam", "sgd"}
_REQUIRED_DDP_KNOB_VALUES = {
    "use_distributed_optimizer": True,
    "use_megatron_fsdp": True,
}
_UNSUPPORTED_OPTIMIZATION_KNOBS = {
    "use_hsdp",
    "hsdp",
}


@dataclass(frozen=True, slots=True)
class MFSDPConfig:
    """Narrow configuration owned by the standalone M-FSDP primitive.

    The fields are the subset of Megatron-FSDP that MLite exercises.  Keeping
    this contract local avoids importing the much broader MCore DDP config and
    makes unsupported sharding modes fail before any buffers are allocated.
    """

    sharding_strategy: str = "optim_grads_params"
    bucket_size: int | None = 40_000_000
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    average_gradients: bool = True
    main_params_dtype: torch.dtype = torch.float32
    main_grads_dtype: torch.dtype = torch.float32
    grad_comm_dtype: torch.dtype = torch.float32
    nccl_ub: bool = False
    fsdp_double_buffer: bool = False
    disable_symmetric_registration: bool = False
    all_gather_in_start_param_sync: bool = True


def build_mfsdp_config(opt: Any) -> MFSDPConfig:
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
    raw_bucket_size = option(
        "bucket_size", 40_000_000, "suggested_communication_unit_size"
    )
    bucket_size = None if raw_bucket_size is None else int(raw_bucket_size)
    if bucket_size is not None and bucket_size <= 0:
        raise ValueError("M-FSDP bucket_size must be a positive element count or None.")

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
    nccl_ub = bool(option("nccl_ub", False))
    return MFSDPConfig(
        sharding_strategy=strategy,
        bucket_size=bucket_size,
        overlap_grad_reduce=bool(option("overlap_grad_reduce", True)),
        overlap_param_gather=bool(option("overlap_param_gather", True)),
        average_gradients=bool(option("average_in_collective", True)),
        main_params_dtype=main_params_dtype,
        main_grads_dtype=main_grads_dtype,
        grad_comm_dtype=grad_comm_dtype,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=bool(option("fsdp_double_buffer", False)) or nccl_ub,
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


def validate_mfsdp_config(engine_cfg) -> None:
    """Validate the supported surface for optimizer='megatron_fsdp'."""
    validate_mfsdp_topology(engine_cfg.parallel)
    opt = engine_cfg.optimizer
    validate_optimizer_name(getattr(opt, "optimizer", "adam"))
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


def validate_optimizer_name(optimizer_name: str) -> None:
    if optimizer_name not in _SUPPORTED_OPTIMIZERS:
        raise ValueError(
            "optimizer_impl='megatron_fsdp' supports only adam/sgd, "
            f"got {optimizer_name!r}."
        )


def validate_precision_aware_disabled(opt) -> None:
    precision_aware = getattr(opt, "use_precision_aware_optimizer", None)
    raw_overrides = dict(getattr(opt, "override_optimizer_config", None) or {})
    if "use_precision_aware_optimizer" in raw_overrides:
        precision_aware = raw_overrides["use_precision_aware_optimizer"]
    if precision_aware not in {None, True, False}:
        raise ValueError("use_precision_aware_optimizer must be a boolean when set.")
    if precision_aware is True:
        raise ValueError(
            "optimizer_impl='megatron_fsdp' does not support "
            "use_precision_aware_optimizer=True in this image; Slurm smoke hit "
            "transformer_engine::multi_tensor_scale_cuda segfault."
        )


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
    validate_precision_aware_disabled(opt)


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

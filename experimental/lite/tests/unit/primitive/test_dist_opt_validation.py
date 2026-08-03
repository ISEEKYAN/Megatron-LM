# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import inspect

import pytest

from megatron.lite.primitive.optimizers.megatron_wrap import (
    _resolve_grad_reduce_in_fp32,
    build_dist_opt_training_optimizer,
    validate_dist_opt_config,
    validate_dist_opt_session,
)
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.config import ParallelConfig


def _engine_cfg(*, model_name: str, pp: int = 1, vpp: int = 1) -> MegatronLiteConfig:
    return MegatronLiteConfig(model_name=model_name, parallel=ParallelConfig(pp=pp, vpp=vpp))


def test_dist_opt_validation_accepts_model_agnostic_config():
    validate_dist_opt_config(_engine_cfg(model_name="synthetic_custom_model", pp=1, vpp=1))


def test_dist_opt_validation_keeps_vpp_parallel_constraint():
    with pytest.raises(ValueError, match="dist_opt requires pp>1 when vpp>1"):
        validate_dist_opt_config(_engine_cfg(model_name="synthetic_custom_model", pp=1, vpp=2))


def test_validate_dist_opt_session_alias_matches_config_validator():
    assert validate_dist_opt_session is validate_dist_opt_config
    validate_dist_opt_session(_engine_cfg(model_name="another_synthetic_model", pp=2, vpp=2))


def test_dist_opt_grad_reduction_defaults_to_fp32():
    assert _resolve_grad_reduce_in_fp32(_engine_cfg(model_name="default_model")) is True


def test_dist_opt_grad_reduction_can_use_parameter_dtype():
    engine_cfg = _engine_cfg(model_name="memory_bounded_model")
    engine_cfg.grad_reduce_in_fp32 = False

    assert _resolve_grad_reduce_in_fp32(engine_cfg) is False


def test_training_optimizer_exposes_grad_reduction_precision():
    parameter = inspect.signature(build_dist_opt_training_optimizer).parameters[
        "grad_reduce_in_fp32"
    ]

    assert parameter.default is True

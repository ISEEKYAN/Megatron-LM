# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest

from megatron.lite.runtime.contracts.config import OptimizerConfig


def test_muon_contract_matches_pinned_megatron_defaults() -> None:
    config = OptimizerConfig(optimizer="muon")

    assert config.muon_momentum == 0.95
    assert config.muon_split_qkv is True
    assert config.muon_nesterov is False
    assert config.muon_scale_mode == "spectral"
    assert config.muon_fp32_matmul_prec == "medium"
    assert config.muon_coefficient_type == "quintic"
    assert config.muon_num_ns_steps == 5
    assert config.muon_tp_mode == "blockwise"
    assert config.muon_extra_scale_factor == 1.0
    assert config.muon_scalar_optimizer == "adam"


def test_layerwise_overlap_and_offload_contract_matches_pinned_megatron_defaults() -> (
    None
):
    config = OptimizerConfig()

    assert config.use_layer_wise_param_layout is False
    assert config.overlap_grad_reduce is False
    assert config.overlap_param_gather is False
    assert config.overlap_param_gather_with_optimizer_step is False
    assert config.optimizer_cpu_offload is False
    assert config.optimizer_offload_fraction == 0.0
    assert config.use_torch_optimizer_for_cpu_offload is False
    assert config.overlap_cpu_optimizer_d2h_h2d is False
    assert config.pin_cpu_grads is True
    assert config.pin_cpu_params is True
    assert config.offload_optimizer_states is False


def test_legacy_offload_fraction_alias_normalizes_to_canonical_field() -> None:
    config = OptimizerConfig(offload_fraction=0.25)

    assert config.optimizer_offload_fraction == 0.25


def test_conflicting_offload_fraction_alias_fails_loudly() -> None:
    with pytest.raises(
        ValueError, match="offload_fraction.*optimizer_offload_fraction"
    ):
        OptimizerConfig(offload_fraction=0.25, optimizer_offload_fraction=0.5)


def test_muon_scalar_optimizer_is_currently_adam_only() -> None:
    with pytest.raises(ValueError, match="muon_scalar_optimizer.*adam"):
        OptimizerConfig(optimizer="muon", muon_scalar_optimizer="lion")


def test_muon_rejects_cross_step_param_gather_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_param_gather_with_optimizer_step"):
        OptimizerConfig(optimizer="muon", overlap_param_gather_with_optimizer_step=True)

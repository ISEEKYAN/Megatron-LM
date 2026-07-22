# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU-only gate for the shared 8-GPU comparison configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_LITE_ROOT = str(Path(__file__).resolve().parents[3])
sys.path = [path for path in sys.path if path != _LITE_ROOT]
sys.path.insert(0, _LITE_ROOT)


def test_full_parallel_contract_builds_all_three_arms_without_gpu_init() -> None:
    from examples.bench.full_parallel_contract import (
        FullParallelContract,
        config_only_gate,
    )

    artifact = config_only_gate(FullParallelContract(hf_path="/tmp/Qwen3.5-35B-A3B"))

    assert artifact["topology"] == "TP2 x PP2 x EP2 x CP1 = 8 ranks (DP=1)"
    assert set(artifact["arms"]) == {"megatron", "distopt_mlite", "fsdp2_mlite"}
    for arm in artifact["arms"].values():
        backend_cfg = arm["dry_run"]["runtime"]["backend_cfg"]
        assert backend_cfg["parallel"]["tp"] == 2
        assert backend_cfg["parallel"]["pp"] == 2
        assert backend_cfg["parallel"]["ep"] == 2
        assert backend_cfg["parallel"]["cp"] == 1
        assert backend_cfg["model_name"] == "qwen3_5"
        assert backend_cfg["seed"] == 1234
        assert arm["dry_run"]["session"] == {
            "device": "cuda",
            "no_optimizer": False,
            "num_microbatches": 1,
            "same_data_across_dp": True,
            "seed": 1234,
            "seq_len": 4096,
            "steps": 12,
            "use_thd": False,
            "warmup": 2,
        }


def test_every_arm_lowers_muon_and_scheduler_onto_the_base_optimizer() -> None:
    """Muon + the training horizon must live on the base OptimizerConfig.

    Regression for the moe BLOCKERS: the megatron arm previously left
    ``optimizer_algorithm="adam"`` / ``total_training_steps=-1`` on the base
    config and hid Muon in a Megatron-Core override dict, so the real builders
    would raise (algorithm conflict) or skip the LR scheduler while the dict-only
    gate reported green.
    """
    from examples.bench.bench import build_runtime_config
    from examples.bench.full_parallel_contract import (
        FullParallelContract,
        build_arm_configs,
    )

    contract = FullParallelContract(hf_path="/tmp/Qwen3.5-35B-A3B")
    arms = build_arm_configs(contract)
    for name, cli_cfg in arms.items():
        backend_cfg = build_runtime_config(cli_cfg).backend_cfg
        assert backend_cfg.optimizer.optimizer_algorithm == "muon", name
        assert backend_cfg.optimizer.total_training_steps == contract.steps, name
        assert backend_cfg.optimizer.muon_momentum == contract.muon_momentum, name

    megatron_cfg = build_runtime_config(arms["megatron"]).backend_cfg
    # The Megatron-Core override surface must stay clean: a declared algorithm
    # there conflicts with the base config at build_dist_opt_optimizer_config.
    assert megatron_cfg.override_optimizer_config == {}


def test_full_parallel_contract_rejects_a_topology_that_is_not_eight_ranks() -> None:
    from examples.bench.full_parallel_contract import FullParallelContract

    with pytest.raises(ValueError, match=r"TP \* PP \* EP \* CP"):
        FullParallelContract(cp=2).validate()


def test_full_parallel_contract_rejects_non_muon_optimizer() -> None:
    from examples.bench.full_parallel_contract import FullParallelContract

    with pytest.raises(ValueError, match="requires Muon"):
        FullParallelContract(optimizer_algorithm="adam").validate()


def test_full_parallel_contract_routes_muon_to_the_fsdp2_optimizer() -> None:
    """The contract must build a real Muon child, never silently AdamW only."""
    import torch

    from megatron.lite.primitive.optimizers.fsdp2.optimizer import build_fsdp2_muon
    from megatron.lite.primitive.optimizers.fsdp2.muon import FP32Muon
    from megatron.lite.runtime.contracts.config import OptimizerConfig

    class MatrixAndBias(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.randn(4, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

    model = MatrixAndBias()
    model.matrix.is_managed_by_layer_wise_optimizer = True
    optimizer = build_fsdp2_muon(
        [model], OptimizerConfig(optimizer_algorithm="muon"), ps=None
    )

    assert any(isinstance(child, FP32Muon) for child in optimizer.optimizer.optimizers)

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


def test_full_parallel_contract_rejects_a_topology_that_is_not_eight_ranks() -> None:
    from examples.bench.full_parallel_contract import FullParallelContract

    with pytest.raises(ValueError, match=r"TP \* PP \* EP \* CP"):
        FullParallelContract(cp=2).validate()


def test_full_parallel_contract_rejects_non_muon_optimizer() -> None:
    from examples.bench.full_parallel_contract import FullParallelContract

    with pytest.raises(ValueError, match="requires Muon"):
        FullParallelContract(optimizer_algorithm="adam").validate()

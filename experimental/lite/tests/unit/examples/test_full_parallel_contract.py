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


# ---------------------------------------------------------------------------
# C-MLITE-SCHED-NULL regression: the mlite arms must build a real LR scheduler.
#
# Megatron-Core is absent on a bare CPU host, so ``OptimizerParamScheduler``
# cannot be instantiated here. These tests inject a minimal stand-in module so
# the *wiring* (mlite builds a scheduler, and stepping it advances the LR) is
# proven on CPU rather than only asserted structurally.
# ---------------------------------------------------------------------------


class _FakeParamScheduler:
    """Linear-warmup stand-in for megatron.core's OptimizerParamScheduler."""

    def __init__(self, optimizer, *, init_lr, max_lr, min_lr, lr_warmup_steps, **_kwargs):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup = lr_warmup_steps
        self.num_steps = 0
        self._apply()

    def _apply(self) -> None:
        if self.warmup > 0 and self.num_steps < self.warmup:
            lr = self.init_lr + (self.max_lr - self.init_lr) * self.num_steps / self.warmup
        else:
            lr = self.max_lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self, increment: int = 1) -> None:
        self.num_steps += increment
        self._apply()


class _StubOptimizerWithGroups:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.0, "weight_decay": 0.0}]


def _inject_fake_scheduler(monkeypatch) -> None:
    import types

    module = types.ModuleType("megatron.core.optimizer_param_scheduler")
    module.OptimizerParamScheduler = _FakeParamScheduler
    core = sys.modules.get("megatron.core")
    if core is None:
        core = types.ModuleType("megatron.core")
        monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer_param_scheduler", module)


def test_build_lr_scheduler_builds_a_real_lr_advancing_scheduler(monkeypatch) -> None:
    _inject_fake_scheduler(monkeypatch)

    from examples.bench.full_parallel_contract import FullParallelContract
    from megatron.lite.primitive.optimizers.megatron_wrap import build_lr_scheduler
    from megatron.lite.runtime.contracts.config import OptimizerConfig

    contract = FullParallelContract(hf_path="/tmp/Qwen3.5-35B-A3B")
    opt = OptimizerConfig(**contract.optimizer_overrides())
    optimizer = _StubOptimizerWithGroups()

    scheduler = build_lr_scheduler(optimizer, opt)
    assert scheduler is not None

    # Warmup must move the LR off its start value and toward ``max_lr``.
    lr_start = optimizer.param_groups[0]["lr"]
    scheduler.step(1)
    lr_after_one = optimizer.param_groups[0]["lr"]
    assert lr_after_one > lr_start
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(contract.lr)


def test_build_lr_scheduler_returns_none_when_horizon_unset() -> None:
    """No horizon -> no scheduler; the runtime would freeze the LR (guard for it)."""
    from megatron.lite.primitive.optimizers.megatron_wrap import build_lr_scheduler
    from megatron.lite.runtime.contracts.config import OptimizerConfig

    opt = OptimizerConfig(optimizer_algorithm="muon", total_training_steps=-1)
    assert build_lr_scheduler(_StubOptimizerWithGroups(), opt) is None
    assert build_lr_scheduler(None, opt) is None


def test_mlite_runtime_resolves_a_scheduler_for_the_contract(monkeypatch) -> None:
    """The mlite runtime path must build a scheduler (not the old ``None``)."""
    _inject_fake_scheduler(monkeypatch)

    from examples.bench.full_parallel_contract import FullParallelContract
    from megatron.lite.runtime.backends.mlite.runtime import _resolve_lr_scheduler
    from megatron.lite.runtime.contracts.config import OptimizerConfig

    contract = FullParallelContract(hf_path="/tmp/Qwen3.5-35B-A3B")
    opt = OptimizerConfig(**contract.optimizer_overrides())

    scheduler = _resolve_lr_scheduler(_StubOptimizerWithGroups(), opt)
    assert scheduler is not None
    # No optimizer / no config -> nothing to schedule.
    assert _resolve_lr_scheduler(None, opt) is None
    assert _resolve_lr_scheduler(_StubOptimizerWithGroups(), None) is None


def test_mlite_lr_scheduler_step_advances_lr_instead_of_returning_zero() -> None:
    """``lr_scheduler_step`` must drive the real scheduler, not silently no-op."""
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts.handle import ModelHandle

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    optimizer = _StubOptimizerWithGroups()
    scheduler = _FakeParamScheduler(
        optimizer, init_lr=0.0, max_lr=1.0e-4, min_lr=1.0e-5, lr_warmup_steps=2
    )
    handle = ModelHandle(model=None, optimizer=optimizer, lr_scheduler=scheduler)

    lr = runtime.lr_scheduler_step(handle)
    assert lr == optimizer.param_groups[0]["lr"]
    assert lr > 0.0

    # Without a scheduler the arm would train frozen; the contract forbids that,
    # but the method itself still degrades to 0.0 (and the gate/tests catch the
    # missing scheduler upstream).
    frozen_handle = ModelHandle(model=None, optimizer=optimizer, lr_scheduler=None)
    assert runtime.lr_scheduler_step(frozen_handle) == 0.0


def test_gate_probe_flags_a_frozen_lr_arm() -> None:
    """The gate probe must fail loudly when an arm would drop its scheduler."""
    from examples.bench.full_parallel_contract import _probe_lr_scheduler_would_build
    from megatron.lite.runtime.contracts.config import OptimizerConfig

    frozen = OptimizerConfig(optimizer_algorithm="muon", total_training_steps=-1)
    with pytest.raises(AssertionError, match="frozen LR"):
        _probe_lr_scheduler_would_build(frozen)

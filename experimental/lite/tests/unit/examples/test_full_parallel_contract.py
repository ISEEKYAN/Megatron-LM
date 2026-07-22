# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU-only gate for the shared 8-GPU comparison configuration."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_LITE_ROOT = str(Path(__file__).resolve().parents[3])
sys.path = [path for path in sys.path if path != _LITE_ROOT]
sys.path.insert(0, _LITE_ROOT)


def test_full_parallel_contract_really_builds_all_three_arm_optimizers(monkeypatch) -> None:
    """The gate must *construct* each arm's optimizer + scheduler, not diff config.

    Megatron-Core is absent on a bare CPU host, so the builder surfaces
    (``OptimizerConfig`` + ``OptimizerParamScheduler``) are injected. The dist_opt
    arms then build a real ``muon`` Megatron-Core config, the FSDP2 arm builds a
    real ``FP32Muon`` child, and every arm builds a non-``None`` LR scheduler --
    the four false-greens the moe repeatedly caught.
    """
    _inject_fake_megatron_core(monkeypatch)

    from examples.bench.full_parallel_contract import (
        FullParallelContract,
        config_only_gate,
    )

    artifact = config_only_gate(FullParallelContract(hf_path="/tmp/Qwen3.5-35B-A3B"))

    assert artifact["topology"] == "TP2 x PP2 x EP2 x CP1 = 8 ranks (DP=1)"
    assert set(artifact["arms"]) == {"megatron", "distopt_mlite", "fsdp2_mlite"}

    builds = artifact["arm_builds"]
    assert builds["megatron"] == {"optimizer": "megatron_core:muon", "scheduler": "built"}
    assert builds["distopt_mlite"] == {
        "optimizer": "megatron_core:muon",
        "scheduler": "built",
    }
    assert builds["fsdp2_mlite"] == {"optimizer": "fsdp2_muon", "scheduler": "built"}

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


@dataclass
class _FakeCoreOptimizerConfig:
    """Stand-in for megatron.core's OptimizerConfig.

    ``build_dist_opt_optimizer_config`` inspects ``fields(...)`` to decide which
    Muon knobs it may forward, so this must be a dataclass carrying the base and
    Muon fields the builder constructs with.
    """

    optimizer: str = "adam"
    lr: float = 0.0
    min_lr: float = 0.0
    weight_decay: float = 0.0
    clip_grad: float = 0.0
    use_distributed_optimizer: bool = False
    bf16: bool = False
    params_dtype: object = None
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    adam_eps: float | None = None
    use_precision_aware_optimizer: bool | None = None
    decoupled_weight_decay: bool | None = None
    optimizer_offload_fraction: float | None = None
    overlap_cpu_optimizer_d2h_h2d: bool | None = None
    optimizer_cpu_offload: bool | None = None
    muon_momentum: float | None = None
    muon_nesterov: bool | None = None
    muon_scale_mode: str | None = None
    muon_fp32_matmul_prec: str | None = None
    muon_coefficient_type: str | None = None
    muon_num_ns_steps: int | None = None
    muon_tp_mode: str | None = None
    muon_extra_scale_factor: float | None = None
    muon_scalar_optimizer: str | None = None
    muon_split_qkv: bool | None = None


def _inject_fake_core_optimizer_config(monkeypatch) -> None:
    import types

    core = sys.modules.get("megatron.core")
    if core is None:
        core = types.ModuleType("megatron.core")
        monkeypatch.setitem(sys.modules, "megatron.core", core)
    optimizer_pkg = sys.modules.get("megatron.core.optimizer")
    if optimizer_pkg is None:
        optimizer_pkg = types.ModuleType("megatron.core.optimizer")
        monkeypatch.setitem(sys.modules, "megatron.core.optimizer", optimizer_pkg)
    config_module = types.ModuleType("megatron.core.optimizer.optimizer_config")
    config_module.OptimizerConfig = _FakeCoreOptimizerConfig
    monkeypatch.setitem(
        sys.modules, "megatron.core.optimizer.optimizer_config", config_module
    )


def _inject_fake_megatron_core(monkeypatch) -> None:
    """Inject both Megatron-Core builder surfaces the real gate build depends on."""
    _inject_fake_scheduler(monkeypatch)
    _inject_fake_core_optimizer_config(monkeypatch)


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


def test_gate_flags_a_frozen_lr_arm() -> None:
    """The gate build must fail loudly when an arm would drop its scheduler.

    A ``total_training_steps<=0`` horizon makes ``build_lr_scheduler`` return
    ``None`` *before* it touches Megatron-Core, so no fake injection is needed.
    """
    from examples.bench.full_parallel_contract import (
        _StubOptimizer,
        _build_scheduler_and_assert_advances,
    )
    from megatron.lite.runtime.contracts.config import OptimizerConfig

    frozen = OptimizerConfig(optimizer_algorithm="muon", total_training_steps=-1)
    with pytest.raises(AssertionError, match="frozen LR"):
        _build_scheduler_and_assert_advances(_StubOptimizer(), frozen, name="megatron")


def test_gate_fails_loud_when_megatron_core_is_absent(monkeypatch) -> None:
    """Without the real builder surface the gate must raise, not pass config-only.

    This is the root-cause guard: the old gate swallowed ``ImportError`` and fell
    back to config diffing, so any path whose config disagreed with its real
    builder passed green. Forcing the Megatron-Core optimizer-config import to
    fail proves the gate now fails loud instead.
    """
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer.optimizer_config", None)

    from examples.bench.full_parallel_contract import (
        FullParallelContract,
        build_all_arms,
    )

    with pytest.raises((ImportError, ModuleNotFoundError)):
        build_all_arms(FullParallelContract(hf_path="/tmp/Qwen3.5-35B-A3B"))

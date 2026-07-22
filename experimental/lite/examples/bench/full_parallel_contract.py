# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared 8-GPU TP/PP/EP/CP training contract for the Muon comparisons.

This module deliberately builds *configuration only*.  The GPU executions and
their precision/performance assertions live in the follow-up tasks; keeping the
contract here means those runs cannot quietly drift in model, input stream,
initialization, scheduler, or Muon settings.

The primary topology is ``TP2 x PP2 x EP2 x CP1 = 8``.  CP is explicit even
though it is one: making it an ordinary contract field prevents a later arm
from relying on an implicit CP default.  Eight ranks cannot also use CP2 with
TP2/PP2/EP2 (that would require 16 ranks).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from examples.bench.bench import (
    BenchCliConfig,
    build_dry_run_plan,
    build_runtime_config,
)


@dataclass(frozen=True)
class FullParallelContract:
    """Fields that must be identical for every comparison arm."""

    model_name: str = "qwen3_5"
    hf_path: str = "/models/Qwen3.5-35B-A3B"
    dtype: str = "bfloat16"
    tp: int = 2
    pp: int = 2
    ep: int = 2
    cp: int = 1
    world_size: int = 8
    seq_len: int = 4096
    num_microbatches: int = 1
    steps: int = 12
    warmup: int = 2
    model_init_seed: int = 1234
    data_seed: int = 1234
    seed: int = 1234
    same_data_across_dp: bool = True
    optimizer_algorithm: str = "muon"
    lr: float = 1.0e-4
    min_lr: float = 1.0e-5
    total_training_steps: int = 12
    lr_warmup_steps: int = 2
    lr_decay_steps: int = 12
    lr_decay_style: str = "linear"
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    muon_momentum: float = 0.95
    muon_split_qkv: bool = True
    muon_nesterov: bool = False
    muon_scale_mode: str = "spectral"
    muon_fp32_matmul_prec: str = "medium"
    muon_coefficient_type: str = "quintic"
    muon_num_ns_steps: int = 5
    muon_tp_mode: str = "blockwise"
    muon_extra_scale_factor: float = 1.0
    muon_scalar_optimizer: str = "adam"

    def validate(self) -> None:
        if self.tp * self.pp * self.ep * self.cp != self.world_size:
            raise ValueError("TP * PP * EP * CP must equal world_size.")
        if (self.tp, self.pp, self.ep, self.cp) != (2, 2, 2, 1):
            raise ValueError("The 8-GPU comparison contract is TP2 x PP2 x EP2 x CP1.")
        if self.model_name != "qwen3_5":
            raise ValueError(
                "The comparison contract requires the real Qwen3.5 MoE model."
            )
        if self.dtype != "bfloat16":
            raise ValueError("The comparison contract requires bfloat16.")
        if self.optimizer_algorithm != "muon":
            raise ValueError("The comparison contract requires Muon.")
        if self.seed != self.data_seed or self.model_init_seed != self.seed:
            raise ValueError(
                "Model initialization and data stream must use the fixed seed."
            )
        if self.steps != self.total_training_steps or self.steps != self.lr_decay_steps:
            raise ValueError(
                "The fixed token budget and scheduler horizon must match steps."
            )

    def optimizer_overrides(self) -> dict[str, Any]:
        return {
            "optimizer_algorithm": self.optimizer_algorithm,
            "lr": self.lr,
            "min_lr": self.min_lr,
            "total_training_steps": self.total_training_steps,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_decay_steps": self.lr_decay_steps,
            "lr_decay_style": self.lr_decay_style,
            "weight_decay": self.weight_decay,
            "clip_grad": self.clip_grad,
            "muon_momentum": self.muon_momentum,
            "muon_split_qkv": self.muon_split_qkv,
            "muon_nesterov": self.muon_nesterov,
            "muon_scale_mode": self.muon_scale_mode,
            "muon_fp32_matmul_prec": self.muon_fp32_matmul_prec,
            "muon_coefficient_type": self.muon_coefficient_type,
            "muon_num_ns_steps": self.muon_num_ns_steps,
            "muon_tp_mode": self.muon_tp_mode,
            "muon_extra_scale_factor": self.muon_extra_scale_factor,
            "muon_scalar_optimizer": self.muon_scalar_optimizer,
        }


def build_arm_configs(contract: FullParallelContract) -> dict[str, BenchCliConfig]:
    """Return the three runs, differing only in the declared implementation arm."""

    contract.validate()
    common = dict(
        hf_path=contract.hf_path,
        model_name=contract.model_name,
        tp=contract.tp,
        pp=contract.pp,
        ep=contract.ep,
        cp=contract.cp,
        steps=contract.steps,
        warmup=contract.warmup,
        num_microbatches=contract.num_microbatches,
        seq_len=contract.seq_len,
        seed=contract.seed,
        same_data_across_dp=contract.same_data_across_dp,
        optimizer_lr=contract.lr,
        optimizer_weight_decay=contract.weight_decay,
        optimizer_clip_grad=contract.clip_grad,
        override_optimizer_json=json.dumps(
            contract.optimizer_overrides(), sort_keys=True
        ),
    )
    return {
        # mbridge is the existing Megatron-Core distopt reference adapter.
        "megatron": BenchCliConfig(backend="mbridge", **common),
        "distopt_mlite": BenchCliConfig(
            backend="mlite", impl_cfg_json='{"optimizer":"dist_opt"}', **common
        ),
        "fsdp2_mlite": BenchCliConfig(
            backend="mlite", impl_cfg_json='{"optimizer":"fsdp2"}', **common
        ),
    }


def _probe_megatron_optimizer_build(backend_cfg) -> None:
    """Invoke the real Megatron-Core optimizer-config builder when it is importable.

    On a bare CPU host Megatron-Core is unavailable and this is a no-op -- the
    base-config assertions above already guard against the false-green. Wherever
    Megatron-Core *is* importable (the GPU arms in the follow-up tasks), this
    proves the megatron arm's config actually constructs instead of raising a
    base-vs-override algorithm conflict.
    """
    try:
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_optimizer_config,
        )

        build_dist_opt_optimizer_config(
            backend_cfg.optimizer,
            override_optimizer_config=backend_cfg.override_optimizer_config,
        )
    except (ImportError, ModuleNotFoundError):
        # Megatron-Core not present (bare CPU gate); base-config checks stand.
        return


class _StubOptimizer:
    """Minimal optimizer stand-in for the scheduler-build probe.

    ``build_lr_scheduler`` only touches ``param_groups`` inside
    ``OptimizerParamScheduler``; the None-vs-build decision happens before that,
    so a single dummy group is enough to exercise the branch the runtime takes.
    """

    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.0, "weight_decay": 0.0}]


def _probe_lr_scheduler_would_build(optimizer_config) -> None:
    """Prove the arm builds a real LR scheduler instead of freezing the LR.

    Every arm's runtime lowers this same ``OptimizerConfig`` into the shared
    ``build_lr_scheduler``. Checking only ``total_training_steps > 0`` is
    false-green: the mlite arms used to hard-code ``lr_scheduler=None`` and drop
    the schedule regardless. Here we call the exact builder the runtimes call.

    * Returns a scheduler object -> the arm schedules its LR. Good.
    * Returns ``None`` -> the arm would train with a frozen LR. Fail loudly.
    * Raises ``ImportError`` -> Megatron-Core is absent (bare CPU gate), but the
      code already passed the ``total_training_steps > 0`` guard and reached the
      ``OptimizerParamScheduler`` import, i.e. it *would* build on the GPU arms.
      The executable proof that the object constructs and steps the LR lives in
      the unit tests (which inject a stub scheduler module).
    """
    from megatron.lite.primitive.optimizers.megatron_wrap import build_lr_scheduler

    try:
        scheduler = build_lr_scheduler(_StubOptimizer(), optimizer_config)
    except (ImportError, ModuleNotFoundError):
        return
    if scheduler is None:
        raise AssertionError(
            "arm would train with a frozen LR: build_lr_scheduler returned None "
            "for a contract with a real training horizon."
        )


def config_only_gate(contract: FullParallelContract) -> dict[str, Any]:
    """Exercise all three real config constructors without distributed/GPU init."""

    arms = build_arm_configs(contract)
    plans = {name: build_dry_run_plan(cfg) for name, cfg in arms.items()}
    runtime_configs = {name: build_runtime_config(cfg) for name, cfg in arms.items()}
    expected_parallel = {
        "tp": contract.tp,
        "pp": contract.pp,
        "ep": contract.ep,
        "cp": contract.cp,
    }
    for name, plan in plans.items():
        backend_cfg = plan["runtime"]["backend_cfg"]
        actual_parallel = {
            key: backend_cfg["parallel"][key] for key in expected_parallel
        }
        if actual_parallel != expected_parallel:
            raise AssertionError(
                f"{name} parallel contract drifted: {actual_parallel!r}"
            )
        # Every arm must carry the contract on the *base* ``OptimizerConfig`` --
        # the object the real optimizer/scheduler builders read. Checking a
        # backend override dict instead would pass a config that raises (base
        # ``adam`` vs override ``muon``) or silently skips the LR scheduler
        # (``total_training_steps`` left at ``-1``) at real build time.
        runtime_backend_cfg = runtime_configs[name].backend_cfg
        optimizer = runtime_backend_cfg.optimizer
        expected_optimizer = contract.optimizer_overrides()
        actual_optimizer = {key: getattr(optimizer, key) for key in expected_optimizer}
        if actual_optimizer != expected_optimizer:
            raise AssertionError(f"{name} optimizer contract drifted.")
        # The scheduler horizon must be lowered onto the optimizer *and* the arm
        # must actually build a scheduler from it. ``build_lr_scheduler`` returns
        # ``None`` when the horizon is <= 0, so the arm would train with a frozen
        # LR while claiming the contract's schedule; the probe calls the exact
        # builder each runtime uses (see ``_probe_lr_scheduler_would_build``).
        if optimizer.total_training_steps <= 0:
            raise AssertionError(
                f"{name} scheduler horizon was not lowered onto the optimizer."
            )
        try:
            _probe_lr_scheduler_would_build(optimizer)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc
        if name == "megatron":
            override = runtime_backend_cfg.override_optimizer_config
            if override:
                raise AssertionError(
                    "megatron arm must lower Muon onto the base optimizer, not a "
                    "conflicting Megatron-Core override dict."
                )
            _probe_megatron_optimizer_build(runtime_backend_cfg)

    return {
        "contract": asdict(contract),
        "topology": "TP2 x PP2 x EP2 x CP1 = 8 ranks (DP=1)",
        "arms": {
            name: {
                "backend": cfg.backend,
                "impl_cfg_json": cfg.impl_cfg_json,
                "dry_run": plans[name],
            }
            for name, cfg in arms.items()
        },
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=FullParallelContract.hf_path)
    args = parser.parse_args(argv)
    artifact = config_only_gate(FullParallelContract(hf_path=args.hf_path))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return artifact


if __name__ == "__main__":
    main()

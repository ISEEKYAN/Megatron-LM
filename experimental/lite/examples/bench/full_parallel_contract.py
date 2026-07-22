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


class _StubOptimizer:
    """Minimal optimizer stand-in whose ``param_groups`` a scheduler can drive.

    ``build_lr_scheduler`` reads and writes ``param_groups[*]["lr"]``; a single
    dummy group is enough to build the real ``OptimizerParamScheduler`` and prove
    a step advances the LR for the dist_opt arms (whose real optimizer needs a
    GPU/distributed init the CPU gate cannot perform).
    """

    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.0, "weight_decay": 0.0}]


def _tiny_muon_model():
    """A minimal module with one Muon-managed matrix and an Adam-fallback bias.

    ``build_fsdp2_muon`` routes the tagged matrix to an :class:`FP32Muon` child.
    Building the FSDP2 optimizer on this tiny module runs the *actual* builder on
    CPU (no GPU/distributed init), so the gate proves the FSDP2 arm constructs a
    real Muon optimizer instead of silently falling back to AdamW.
    """
    import torch

    class _TinyMatrixModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.randn(4, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

    model = _TinyMatrixModel()
    model.matrix.is_managed_by_layer_wise_optimizer = True
    return model


def _build_scheduler_and_assert_advances(optimizer, opt_config, *, name: str):
    """Build the shared LR scheduler and prove stepping it advances the LR.

    Every arm's runtime lowers this same ``OptimizerConfig`` into the shared
    ``build_lr_scheduler``.  It returns ``None`` when the training horizon is
    unset, which would silently freeze the LR (the ``lr_scheduler=None``
    regression).  This builds the real scheduler and steps it, failing loud if it
    is absent or inert.  ``ImportError`` is *not* swallowed: on a host without
    Megatron-Core the gate must fail rather than pass config-only.
    """
    from megatron.lite.primitive.optimizers.megatron_wrap import build_lr_scheduler

    scheduler = build_lr_scheduler(optimizer, opt_config)
    if scheduler is None:
        raise AssertionError(
            f"{name}: build_lr_scheduler returned None -- the arm would train "
            "with a frozen LR despite a real training horizon."
        )
    lr_before = optimizer.param_groups[0]["lr"]
    scheduler.step(1)
    lr_after = optimizer.param_groups[0]["lr"]
    if opt_config.lr_warmup_steps > 0 and not lr_after > lr_before:
        raise AssertionError(
            f"{name}: LR scheduler did not advance the LR during warmup "
            f"({lr_before} -> {lr_after})."
        )
    return scheduler


def _build_megatron_core_optimizer(opt_config, override, *, name: str):
    """Really build the Megatron-Core ``OptimizerConfig`` for the dist_opt arms.

    Both the ``megatron`` (mbridge) and ``distopt_mlite`` arms lower onto this
    exact builder at real build time.  Constructing it proves the *base* config
    carries ``optimizer="muon"`` (not ``adam``) and forwards the Muon
    hyperparameters, instead of only asserting config fields agree.
    """
    from megatron.lite.primitive.optimizers.megatron_wrap import (
        build_dist_opt_optimizer_config,
    )

    core_cfg = build_dist_opt_optimizer_config(
        opt_config, override_optimizer_config=override
    )
    if str(core_cfg.optimizer).lower() != "muon":
        raise AssertionError(
            f"{name}: Megatron-Core optimizer built as {core_cfg.optimizer!r}, "
            "not muon -- the base config lost the Muon algorithm."
        )
    if getattr(core_cfg, "muon_momentum", None) != opt_config.muon_momentum:
        raise AssertionError(
            f"{name}: Muon momentum was not forwarded to the Megatron-Core "
            f"optimizer ({getattr(core_cfg, 'muon_momentum', None)!r})."
        )
    return core_cfg


def _build_fsdp2_muon_optimizer(opt_config, *, name: str):
    """Really build the FSDP2 Muon optimizer and assert it is not an AdamW fallback."""
    from megatron.lite.primitive.optimizers.fsdp2.muon import FP32Muon
    from megatron.lite.primitive.optimizers.fsdp2.optimizer import build_fsdp2_muon

    optimizer = build_fsdp2_muon([_tiny_muon_model()], opt_config, ps=None)
    if not any(isinstance(child, FP32Muon) for child in optimizer.optimizer.optimizers):
        raise AssertionError(
            f"{name}: FSDP2 optimizer built without an FP32Muon child -- it fell "
            "back to AdamW despite the Muon contract."
        )
    return optimizer


def build_all_arms(contract: FullParallelContract) -> dict[str, Any]:
    """Really construct every arm's optimizer + LR scheduler; fail loud on any gap.

    This is the decisive gate.  Diffing config fields lets any path whose config
    disagrees with its real builder pass green (the repeated false-greens: base
    ``adam`` under a Muon override, an unwritten scheduler horizon, an FSDP2 Muon
    that falls back to AdamW, a ``None`` LR scheduler).  Here we call the exact
    builders each runtime uses:

    * ``megatron`` / ``distopt_mlite`` -> ``build_dist_opt_optimizer_config``
      (asserts ``optimizer="muon"`` on the base config) + the shared scheduler.
    * ``fsdp2_mlite`` -> ``build_fsdp2_muon`` (asserts a real ``FP32Muon`` child)
      + the shared scheduler driven by that optimizer's own ``param_groups``.

    A missing Megatron-Core raises here rather than being swallowed, so the CPU
    proof injects the real builder surface (see the unit tests) and the GPU arms
    use the genuine packages.
    """
    receipts: dict[str, dict[str, Any]] = {}
    for name, cli_cfg in build_arm_configs(contract).items():
        runtime_backend_cfg = build_runtime_config(cli_cfg).backend_cfg
        opt_config = runtime_backend_cfg.optimizer
        if opt_config.total_training_steps <= 0:
            raise AssertionError(
                f"{name}: scheduler horizon was not lowered onto the optimizer."
            )
        if opt_config.optimizer_algorithm != "muon":
            raise AssertionError(
                f"{name}: base optimizer_algorithm is "
                f"{opt_config.optimizer_algorithm!r}, not muon."
            )
        if name == "fsdp2_mlite":
            optimizer = _build_fsdp2_muon_optimizer(opt_config, name=name)
            _build_scheduler_and_assert_advances(optimizer, opt_config, name=name)
            receipts[name] = {"optimizer": "fsdp2_muon", "scheduler": "built"}
        else:
            override = getattr(runtime_backend_cfg, "override_optimizer_config", None)
            if name == "megatron" and override:
                raise AssertionError(
                    "megatron arm must lower Muon onto the base optimizer, not a "
                    "conflicting Megatron-Core override dict."
                )
            core_cfg = _build_megatron_core_optimizer(opt_config, override, name=name)
            _build_scheduler_and_assert_advances(_StubOptimizer(), opt_config, name=name)
            receipts[name] = {
                "optimizer": f"megatron_core:{core_cfg.optimizer}",
                "scheduler": "built",
            }
    return receipts


def config_only_gate(contract: FullParallelContract) -> dict[str, Any]:
    """Assemble the shared config and *really build* all three arms' optimizers.

    The topology/config assembly is CPU-only, but the acceptance check is the
    real build in :func:`build_all_arms` -- not a config diff.  ``build_all_arms``
    fails loud if any arm cannot construct its optimizer or LR scheduler.
    """

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
        # the object the real optimizer/scheduler builders read.
        optimizer = runtime_configs[name].backend_cfg.optimizer
        expected_optimizer = contract.optimizer_overrides()
        actual_optimizer = {key: getattr(optimizer, key) for key in expected_optimizer}
        if actual_optimizer != expected_optimizer:
            raise AssertionError(f"{name} optimizer contract drifted.")

    # Decisive check: construct each arm's real optimizer + LR scheduler.
    arm_builds = build_all_arms(contract)

    return {
        "contract": asdict(contract),
        "topology": "TP2 x PP2 x EP2 x CP1 = 8 ranks (DP=1)",
        "arm_builds": arm_builds,
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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Controlled two-arm configuration gate for Qwen3 EP overlap."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from examples.bench import bench as bench_module
from examples.bench import cycle_memory_probe

BenchCliConfig = bench_module.BenchCliConfig
build_dry_run_plan = bench_module.build_dry_run_plan
build_runtime_config = bench_module.build_runtime_config
current_snapshot = cycle_memory_probe.current_snapshot
dump_snapshot = cycle_memory_probe.dump_snapshot
live_allocation_stacks = cycle_memory_probe.live_allocation_stacks
per_cycle_retention = cycle_memory_probe.per_cycle_retention
record_memory_history = cycle_memory_probe.record_memory_history
sample_cuda_memory = cycle_memory_probe.sample_cuda_memory

_OVERLAP_KEY = "overlap_moe_expert_parallel_comm"
_SHARED_IMPL_CFG: dict[str, Any] = {"use_deepep": False, "recompute": []}


def build_arm_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = deepcopy(_SHARED_IMPL_CFG)
    overlap = deepcopy(_SHARED_IMPL_CFG)
    baseline[_OVERLAP_KEY] = False
    overlap[_OVERLAP_KEY] = True
    assert_only_overlap_diff(baseline, overlap)
    return baseline, overlap


def select_arm_config(arm: str) -> dict[str, Any]:
    baseline, overlap = build_arm_configs()
    return {"baseline": baseline, "overlap": overlap}[arm]


def assert_only_overlap_diff(baseline: dict[str, Any], overlap: dict[str, Any]) -> None:
    if set(baseline) != set(overlap):
        raise ValueError(
            f"impl_cfg keys differ: baseline={sorted(baseline)}, overlap={sorted(overlap)}"
        )
    for key in sorted(baseline):
        if key == _OVERLAP_KEY:
            if baseline[key] is not False or overlap[key] is not True:
                raise ValueError(f"{_OVERLAP_KEY} must be False/True")
        elif baseline[key] != overlap[key]:
            raise ValueError(f"controlled A/B differs outside {_OVERLAP_KEY}: {key}")


def build_config_only_plans(hf_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline, overlap = build_arm_configs()
    common = dict(
        backend="mlite",
        hf_path=hf_path,
        model_name="qwen3_moe",
        tp=1,
        etp=1,
        ep=8,
        pp=1,
        cp=1,
        steps=4,
        warmup=1,
        num_microbatches=4,
        seq_len=256,
        truncate_layers=2,
        keep_experts=8,
        disable_mtp=True,
        same_data_across_dp=True,
        skip_load_hf_weights=True,
        dry_run=True,
    )
    baseline_plan = build_dry_run_plan(
        BenchCliConfig(**common, impl_cfg_json=json.dumps(baseline))
    )
    overlap_plan = build_dry_run_plan(
        BenchCliConfig(**common, impl_cfg_json=json.dumps(overlap))
    )
    assert_only_overlap_diff(
        baseline_plan["runtime"]["backend_cfg"]["impl_cfg"],
        overlap_plan["runtime"]["backend_cfg"]["impl_cfg"],
    )
    return baseline_plan, overlap_plan


def require_probe_artifacts(
    out_dir: Path, arm: str, rank: int, cycles: int, warmup: int
) -> list[Path]:
    expected = [
        out_dir / f"{arm}-rank{rank}.csv",
        out_dir / f"{arm}-rank{rank}-summary.json",
        *[
            out_dir / f"{arm}-rank{rank}-cycle{cycle}.pickle"
            for cycle in range(warmup, cycles + warmup)
        ],
    ]
    missing = [
        path for path in expected if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "probe evidence missing or empty: "
            + ", ".join(str(path) for path in missing)
        )
    return expected


def _gpu_arm(
    name: str, cfg: BenchCliConfig, out_dir: Path, cycles: int, warmup: int
) -> None:
    """One arm of the copied per-cycle CSV/snapshot lifecycle."""
    import torch
    from examples.bench.session import PretrainSessionConfig, _make_data_iter
    from megatron.lite.runtime import create_runtime

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to emit CPU proxy evidence")
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    record_memory_history()
    runtime = create_runtime(build_runtime_config(cfg))
    handle = runtime.build_model()
    session = PretrainSessionConfig(
        steps=cycles + warmup,
        warmup=warmup,
        num_microbatches=cfg.num_microbatches,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        device="cuda",
        same_data_across_dp=True,
    )
    rows: list[dict[str, Any]] = []
    path = out_dir / f"{name}-rank{rank}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh, runtime.train_mode(handle):
        fields = [
            "cycle",
            "phase",
            "allocated_bytes",
            "reserved_bytes",
            "reserved_minus_allocated_bytes",
            "inactive_split_bytes",
            "step_ms",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        data = _make_data_iter(handle, session)
        for cycle in range(cycles + warmup):
            runtime.zero_grad(handle)
            torch.cuda.synchronize()
            before = {
                "cycle": cycle,
                "phase": "before",
                **sample_cuda_memory(),
                "step_ms": 0.0,
            }
            writer.writerow(before)
            rows.append(before)
            started = time.perf_counter()
            runtime.forward_backward(
                handle, data, loss_fn=None, num_microbatches=cfg.num_microbatches
            )
            runtime.optimizer_step(handle)
            runtime.lr_scheduler_step(handle)
            torch.cuda.synchronize()
            after = {
                "cycle": cycle,
                "phase": "after",
                **sample_cuda_memory(),
                "step_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            writer.writerow(after)
            rows.append(after)
            if cycle >= warmup:
                dump_snapshot(out_dir / f"{name}-rank{rank}-cycle{cycle}.pickle")
    stacks = live_allocation_stacks(current_snapshot(), top_n=15)
    if not stacks:
        raise RuntimeError(
            "snapshot yielded no live allocation stacks; attribution evidence missing"
        )
    (out_dir / f"{name}-rank{rank}-summary.json").write_text(
        json.dumps(
            {
                "arm": name,
                "impl_cfg": json.loads(cfg.impl_cfg_json),
                "rows": rows,
                "retention": {
                    metric: per_cycle_retention(
                        rows, phase="after", metric=metric, warmup_cycles=warmup
                    )
                    for metric in (
                        "reserved_minus_allocated_bytes",
                        "inactive_split_bytes",
                    )
                },
                "live_stacks": stacks,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--arm", choices=("baseline", "overlap"))
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args(argv)
    baseline, overlap = build_config_only_plans(args.hf_path)
    if args.config_only:
        print(
            json.dumps(
                {
                    "baseline": baseline["runtime"]["backend_cfg"]["impl_cfg"],
                    "overlap": overlap["runtime"]["backend_cfg"]["impl_cfg"],
                },
                sort_keys=True,
            )
        )
        print("QWEN3_EP_OVERLAP_CONFIG_ONLY_OK", flush=True)
        return 0
    if args.arm is None:
        parser.error("--arm is required unless --config-only")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = BenchCliConfig(
        backend="mlite",
        hf_path=args.hf_path,
        model_name="qwen3_moe",
        tp=1,
        etp=1,
        ep=8,
        pp=1,
        cp=1,
        steps=args.cycles + args.warmup,
        warmup=args.warmup,
        num_microbatches=4,
        seq_len=256,
        truncate_layers=2,
        keep_experts=8,
        disable_mtp=True,
        same_data_across_dp=True,
        skip_load_hf_weights=True,
        impl_cfg_json=json.dumps(select_arm_config(args.arm), sort_keys=True),
    )
    _gpu_arm(args.arm, cfg, out_dir, args.cycles, args.warmup)
    require_probe_artifacts(
        out_dir=out_dir,
        arm=args.arm,
        rank=int(os.environ.get("RANK", "0")),
        cycles=args.cycles,
        warmup=args.warmup,
    )
    print("QWEN3_EP_OVERLAP_PROBE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

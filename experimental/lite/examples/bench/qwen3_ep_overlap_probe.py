# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Controlled two-arm configuration gate for Qwen3 EP overlap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
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
reset_peak_memory = cycle_memory_probe.reset_peak_memory
sample_cuda_memory = cycle_memory_probe.sample_cuda_memory

_OVERLAP_KEY = "num_chunks_ep_a2a_overlap"
_SHARED_IMPL_CFG: dict[str, Any] = {"use_deepep": True, "recompute": ["moe"]}
_PARALLEL_EVIDENCE = {"ep": 8, "tp": 1, "pp": 1, "cp": 1}


def _config_root(config: Any) -> Any:
    if isinstance(config, dict):
        return config.get("text_config", config)
    return getattr(config, "text_config", config)


def load_source_model_evidence(hf_path: str) -> dict[str, int | bool]:
    config_path = Path(hf_path) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"model evidence requires {config_path}")
    source = _config_root(json.loads(config_path.read_text(encoding="utf-8")))
    if isinstance(source, dict):
        layers = source.get("num_hidden_layers")
        experts = source.get("num_experts")
    else:
        layers = getattr(source, "num_hidden_layers", None)
        experts = getattr(source, "num_experts", None)
    if not isinstance(layers, int) or layers <= 2:
        raise ValueError(
            f"full multi-layer evidence requires >2 layers, got {layers!r}"
        )
    if not isinstance(experts, int) or experts <= 0:
        raise ValueError(f"MoE evidence requires num_experts > 0, got {experts!r}")
    return {"num_hidden_layers": layers, "truncated": False, "num_experts": experts}


def build_model_evidence(cfg: BenchCliConfig, model_cfg: Any) -> dict[str, int | bool]:
    if cfg.truncate_layers is not None or cfg.keep_experts is not None:
        raise ValueError("probe evidence forbids truncated layers or experts")
    root = _config_root(model_cfg)
    evidence = {
        "num_hidden_layers": int(getattr(root, "num_hidden_layers")),
        "truncated": False,
        "num_experts": int(getattr(root, "num_experts")),
    }
    source = load_source_model_evidence(cfg.hf_path)
    if evidence != source:
        raise ValueError(
            f"built model differs from source model: built={evidence}, source={source}"
        )
    return evidence


def build_parallel_evidence(cfg: BenchCliConfig) -> dict[str, int]:
    evidence = {"ep": cfg.ep, "tp": cfg.tp, "pp": cfg.pp, "cp": cfg.cp}
    if evidence != _PARALLEL_EVIDENCE:
        raise ValueError(
            f"probe requires {_PARALLEL_EVIDENCE}, got parallel topology {evidence}"
        )
    return evidence


def _performance_summary(rows: list[dict[str, Any]], warmup: int) -> dict[str, Any]:
    measured = [
        row for row in rows if row["phase"] == "after" and int(row["cycle"]) >= warmup
    ]
    step_ms = [float(row["step_ms"]) for row in measured]
    if len(step_ms) < 2:
        raise ValueError(
            f"performance evidence requires >=2 repeats, got {len(step_ms)}"
        )
    mean = statistics.fmean(step_ms)
    sem = statistics.stdev(step_ms) / math.sqrt(len(step_ms))
    return {
        "repeat_count": len(step_ms),
        "step_ms": {
            "mean": mean,
            "median": statistics.median(step_ms),
            "ci95_low": mean - 1.96 * sem,
            "ci95_high": mean + 1.96 * sem,
            "samples": step_ms,
        },
        "throughput_tokens_per_second": {
            "mean": statistics.fmean(
                float(row["throughput_tokens_per_second"]) for row in measured
            ),
            "samples": [float(row["throughput_tokens_per_second"]) for row in measured],
        },
    }


def validate_probe_summary(summary: dict[str, Any], *, warmup: int) -> None:
    model = summary.get("model") or {}
    if model.get("truncated") is not False:
        raise ValueError("probe evidence must declare truncated=false")
    if int(model.get("num_hidden_layers", 0)) <= 2:
        raise ValueError("probe evidence requires full multi-layer model")
    if int(model.get("num_experts", 0)) <= 0:
        raise ValueError("probe evidence requires num_experts")
    if summary.get("parallel") != _PARALLEL_EVIDENCE:
        raise ValueError(f"unexpected parallel topology: {summary.get('parallel')!r}")
    measured = [
        row
        for row in summary.get("rows", [])
        if row.get("phase") == "after" and int(row.get("cycle", -1)) >= warmup
    ]
    peak_fields = ("peak_allocated_bytes", "peak_reserved_bytes")
    if len(measured) < 2 or any(
        not isinstance(row.get(field), int) or int(row[field]) <= 0
        for row in measured
        for field in peak_fields
    ):
        raise ValueError("peak memory evidence missing or invalid")
    expected_peak = {
        "max_memory_allocated_bytes": max(
            int(row["peak_allocated_bytes"]) for row in measured
        ),
        "max_memory_reserved_bytes": max(
            int(row["peak_reserved_bytes"]) for row in measured
        ),
    }
    if summary.get("peak_memory") != expected_peak:
        raise ValueError("peak memory aggregate missing or inconsistent")
    if summary.get("performance", {}).get("repeat_count") != len(measured):
        raise ValueError("performance repeat evidence missing or inconsistent")


def build_arm_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = deepcopy(_SHARED_IMPL_CFG)
    overlap = deepcopy(_SHARED_IMPL_CFG)
    baseline[_OVERLAP_KEY] = 1
    overlap[_OVERLAP_KEY] = 2
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
            if baseline[key] != 1 or overlap[key] != 2:
                raise ValueError(f"{_OVERLAP_KEY} must be 1/2")
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
        steps=13,
        warmup=3,
        num_microbatches=4,
        seq_len=256,
        disable_mtp=True,
        same_data_across_dp=True,
        skip_load_hf_weights=True,
        dry_run=True,
    )
    baseline_cfg = BenchCliConfig(**common, impl_cfg_json=json.dumps(baseline))
    overlap_cfg = BenchCliConfig(**common, impl_cfg_json=json.dumps(overlap))
    baseline_plan = build_dry_run_plan(baseline_cfg)
    overlap_plan = build_dry_run_plan(overlap_cfg)
    assert_only_overlap_diff(
        baseline_plan["runtime"]["backend_cfg"]["impl_cfg"],
        overlap_plan["runtime"]["backend_cfg"]["impl_cfg"],
    )
    evidence_contract = {
        "model": load_source_model_evidence(hf_path),
        "parallel": build_parallel_evidence(baseline_cfg),
        "required_metrics": [
            "step_ms",
            "throughput_tokens_per_second",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ],
    }
    baseline_plan["evidence_contract"] = deepcopy(evidence_contract)
    overlap_plan["evidence_contract"] = deepcopy(evidence_contract)
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
    model_evidence = build_model_evidence(cfg, handle._extras.get("model_cfg"))
    parallel_evidence = build_parallel_evidence(cfg)
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
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "reserved_minus_allocated_bytes",
            "inactive_split_bytes",
            "step_ms",
            "throughput_tokens_per_second",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        data = _make_data_iter(handle, session)
        for cycle in range(cycles + warmup):
            runtime.zero_grad(handle)
            torch.cuda.synchronize()
            reset_peak_memory()
            before = {
                "cycle": cycle,
                "phase": "before",
                **sample_cuda_memory(),
                "step_ms": 0.0,
                "throughput_tokens_per_second": 0.0,
            }
            writer.writerow(before)
            rows.append(before)
            started = time.perf_counter()
            torch.cuda.nvtx.range_push(f"qwen3_ep_{name}_step_{cycle}")
            try:
                runtime.forward_backward(
                    handle, data, loss_fn=None, num_microbatches=cfg.num_microbatches
                )
                runtime.optimizer_step(handle)
                runtime.lr_scheduler_step(handle)
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            elapsed_ms = (time.perf_counter() - started) * 1000
            step_memory = sample_cuda_memory()
            if torch.distributed.is_initialized():
                elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device="cuda")
                torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
                elapsed_ms = float(elapsed.item())
            tokens_per_step = cfg.num_microbatches * cfg.seq_len * handle.dp_size
            after = {
                "cycle": cycle,
                "phase": "after",
                **step_memory,
                "step_ms": round(elapsed_ms, 3),
                "throughput_tokens_per_second": tokens_per_step / (elapsed_ms / 1000),
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
    summary = {
        "arm": name,
        "impl_cfg": json.loads(cfg.impl_cfg_json),
        "model": model_evidence,
        "parallel": parallel_evidence,
        "rows": rows,
        "performance": _performance_summary(rows, warmup),
        "peak_memory": {
            "max_memory_allocated_bytes": max(
                int(row["peak_allocated_bytes"])
                for row in rows
                if row["phase"] == "after" and int(row["cycle"]) >= warmup
            ),
            "max_memory_reserved_bytes": max(
                int(row["peak_reserved_bytes"])
                for row in rows
                if row["phase"] == "after" and int(row["cycle"]) >= warmup
            ),
        },
        "retention": {
            metric: per_cycle_retention(
                rows, phase="after", metric=metric, warmup_cycles=warmup
            )
            for metric in ("reserved_minus_allocated_bytes", "inactive_split_bytes")
        },
        "live_stacks": stacks,
    }
    validate_probe_summary(summary, warmup=warmup)
    (out_dir / f"{name}-rank{rank}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--arm", choices=("baseline", "overlap"))
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args(argv)
    baseline, overlap = build_config_only_plans(args.hf_path)
    if args.config_only:
        print(
            json.dumps(
                {
                    "baseline": baseline["runtime"]["backend_cfg"]["impl_cfg"],
                    "overlap": overlap["runtime"]["backend_cfg"]["impl_cfg"],
                    "evidence_contract": baseline["evidence_contract"],
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

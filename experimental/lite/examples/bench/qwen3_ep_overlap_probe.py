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

from examples.bench.bench import (
    BenchCliConfig,
    build_dry_run_plan,
    build_runtime_config,
)

_OVERLAP_KEY = "overlap_moe_expert_parallel_comm"
_SHARED_IMPL_CFG: dict[str, Any] = {"use_deepep": False, "recompute": []}


def build_arm_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = deepcopy(_SHARED_IMPL_CFG)
    overlap = deepcopy(_SHARED_IMPL_CFG)
    baseline[_OVERLAP_KEY] = False
    overlap[_OVERLAP_KEY] = True
    assert_only_overlap_diff(baseline, overlap)
    return baseline, overlap


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


def _sample_cuda_memory() -> dict[str, int]:
    """Reuse the established probe's reserved/allocated/inactive-split evidence."""
    import torch

    stats = torch.cuda.memory_stats()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "reserved_minus_allocated_bytes": reserved - allocated,
        "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
    }


def _live_stacks(snapshot: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    """Existing probe's live allocation-stack aggregation, unchanged in criterion."""
    grouped: dict[tuple[str, ...], list[int]] = {}
    for segment in snapshot.get("segments", []) or []:
        for block in segment.get("blocks", []) or []:
            if block.get("state") not in {"active_allocated", "allocated"}:
                continue
            frames = block.get("frames") or []
            if not frames and (history := block.get("history") or []):
                frames = history[0].get("frames") or []
            key = tuple(
                f"{frame.get('filename', '?')}:{frame.get('line', '?')}:{frame.get('name', '?')}"
                for frame in frames
            )[:6]
            if not key:
                continue
            entry = grouped.setdefault(key, [0, 0])
            entry[0] += int(block.get("size", block.get("requested_size", 0)) or 0)
            entry[1] += 1
    return [
        {"frames": list(key), "retained_bytes": value[0], "num_blocks": value[1]}
        for key, value in sorted(
            grouped.items(), key=lambda item: item[1][0], reverse=True
        )[:top_n]
    ]


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
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", stacks="all", max_entries=200000
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"_record_memory_history failed: {exc}") from exc
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
                **_sample_cuda_memory(),
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
                **_sample_cuda_memory(),
                "step_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            writer.writerow(after)
            rows.append(after)
            if cycle >= warmup:
                try:
                    torch.cuda.memory._dump_snapshot(
                        out_dir / f"{name}-rank{rank}-cycle{cycle}.pickle"
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"_dump_snapshot cycle {cycle} failed: {exc}"
                    ) from exc
    stacks = _live_stacks(torch.cuda.memory._snapshot(), top_n=15)
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_configs = build_arm_configs()
    for name, impl_cfg in zip(("baseline", "overlap"), arm_configs, strict=True):
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
            impl_cfg_json=json.dumps(impl_cfg, sort_keys=True),
        )
        _gpu_arm(name, cfg, out_dir, args.cycles, args.warmup)
    print("QWEN3_EP_OVERLAP_PROBE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

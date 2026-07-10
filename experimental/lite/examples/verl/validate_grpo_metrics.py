# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate that a bounded GRPO run produced finite, non-skip RL evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


_FINITE_KEYS = (
    "critic/score/mean",
    "critic/score/min",
    "critic/score/max",
    "actor/pg_loss",
    "actor/grad_norm",
    "perf/time_per_step",
    "perf/throughput",
)


def _finite_float(data: dict, key: str, *, path: Path, line_number: int) -> float:
    if key not in data:
        raise ValueError(f"{path}:{line_number} is missing required metric {key}")
    value = float(data[key])
    if not math.isfinite(value):
        raise ValueError(f"{path}:{line_number} has non-finite {key}={value}")
    return value


def validate_metrics(paths: list[Path], *, expected_steps: int) -> dict:
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    by_step: dict[int, dict[str, float]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                data = payload.get("data")
                if not isinstance(data, dict):
                    raise ValueError(f"{path}:{line_number} has no metric data object")
                step = int(data.get("training/global_step", payload.get("step", -1)))
                values = {
                    key: _finite_float(data, key, path=path, line_number=line_number)
                    for key in _FINITE_KEYS
                }
                if step in by_step:
                    raise ValueError(f"duplicate GRPO metric step {step} across phase logs")
                by_step[step] = values

    expected = list(range(1, expected_steps + 1))
    steps = sorted(by_step)
    if steps != expected:
        raise ValueError(f"expected metric steps {expected}, got {steps}")
    grad_norms = [by_step[step]["actor/grad_norm"] for step in steps]
    throughputs = [by_step[step]["perf/throughput"] for step in steps]
    step_times = [by_step[step]["perf/time_per_step"] for step in steps]
    if max(grad_norms) <= 0:
        raise ValueError("all actor gradient norms are zero; no RL update was demonstrated")
    if min(throughputs) <= 0 or min(step_times) <= 0:
        raise ValueError("throughput and step time must be positive on every RL step")
    if not any(
        by_step[step]["critic/score/max"] > by_step[step]["critic/score/min"]
        for step in steps
    ):
        raise ValueError("no step has reward variation; GRPO advantages are degenerate")

    return {
        "steps": steps,
        "score_mean": [by_step[step]["critic/score/mean"] for step in steps],
        "policy_loss": [by_step[step]["actor/pg_loss"] for step in steps],
        "grad_norm": grad_norms,
        "throughput": throughputs,
        "time_per_step": step_times,
        "max_grad_norm": max(grad_norms),
        "min_throughput": min(throughputs),
        "reward_variation_steps": [
            step
            for step in steps
            if by_step[step]["critic/score/max"] > by_step[step]["critic/score/min"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_metrics(args.metrics, expected_steps=args.expected_steps)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"DS4_GRPO_METRICS_VALID steps={len(report['steps'])} "
        f"max_grad_norm={report['max_grad_norm']:.6e} "
        f"min_throughput={report['min_throughput']:.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract and compare per-step RL metrics from two VERL console logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_STEP = re.compile(r"step:(\d+) -")
_METRICS = (
    "actor/grad_norm",
    "actor/pg_loss",
    "actor/ppo_kl",
    "critic/score/mean",
    "val-core/math_dapo/acc/mean@1",
)


def _records(path: Path) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _ANSI.sub("", raw_line)
        step_match = _STEP.search(line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        values = records.setdefault(step, {})
        for metric in _METRICS:
            match = re.search(rf"(?:^| - ){re.escape(metric)}:([^ ]+)", line)
            if match:
                values[metric] = float(match.group(1))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    candidate = _records(args.candidate)
    baseline = _records(args.baseline)
    comparisons = []
    for step in sorted(set(candidate) & set(baseline)):
        metrics = {}
        for metric in _METRICS:
            cand = candidate[step].get(metric)
            base = baseline[step].get(metric)
            if cand is None or base is None:
                continue
            metrics[metric] = {
                "candidate": cand,
                "baseline": base,
                "ratio": cand / base if base != 0 else None,
                "absolute_difference": cand - base,
            }
        comparisons.append({"step": step, "metrics": metrics})
    print(
        json.dumps(
            {
                "candidate_steps": candidate,
                "baseline_steps": baseline,
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract comparable DAPO reward curves from VERL JSONL/file logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REWARD_KEYS = (
    "critic/rewards/mean",
    "critic/score/mean",
    "reward/mean",
    "train/reward",
)
CONSOLE_RE = re.compile(
    r"\bstep:(?P<step>\d+)\b.*?\bcritic/rewards/mean:"
    r"(?P<reward>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def reward_curve(path: Path) -> list[tuple[int, float]]:
    curve: list[tuple[int, float]] = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            match = CONSOLE_RE.search(line)
            if match:
                curve.append((int(match["step"]), float(match["reward"])))
            continue
        data = row.get("data", row)
        for key in REWARD_KEYS:
            value = data.get(key)
            if isinstance(value, (int, float)):
                curve.append((int(row.get("step", len(curve))), float(value)))
                break
    if len(curve) < 2:
        raise ValueError(f"{path}: found {len(curve)} reward points; keys={REWARD_KEYS}")
    return curve


def summarize(curve: list[tuple[int, float]], window_size: int) -> dict[str, object]:
    if window_size < 1 or 2 * window_size > len(curve):
        raise ValueError(
            f"window_size={window_size} requires at least {2 * window_size} points; "
            f"found {len(curve)}"
        )
    steps = [step for step, _ in curve]
    rewards = [reward for _, reward in curve]
    first_window_mean = sum(rewards[:window_size]) / window_size
    last_window_mean = sum(rewards[-window_size:]) / window_size
    step_mean = sum(steps) / len(steps)
    reward_mean = sum(rewards) / len(rewards)
    denominator = sum((step - step_mean) ** 2 for step in steps)
    if denominator == 0:
        raise ValueError("reward curve steps must not all be identical")
    linear_slope = (
        sum(
            (step - step_mean) * (reward - reward_mean)
            for step, reward in curve
        )
        / denominator
    )
    return {
        "points": [{"step": step, "reward": reward} for step, reward in curve],
        "first": rewards[0],
        "last": rewards[-1],
        "gain": rewards[-1] - rewards[0],
        "window_size": window_size,
        "first_window_mean": first_window_mean,
        "last_window_mean": last_window_mean,
        "window_gain": last_window_mean - first_window_mean,
        "linear_slope": linear_slope,
    }


def summarize_curves(
    curves: dict[str, list[tuple[int, float]]], window_size: int | None = None
) -> dict[str, object]:
    minimum_points = min(len(curve) for curve in curves.values())
    if window_size is None:
        window_size = max(1, min(10, minimum_points // 3))
    summary: dict[str, object] = {
        name: summarize(curve, window_size) for name, curve in curves.items()
    }
    muon = summary["muon"]
    adam = summary["adam"]
    assert isinstance(muon, dict)
    assert isinstance(adam, dict)
    muon_reward_increased = (
        muon["window_gain"] > 0 and muon["linear_slope"] > 0
    )
    muon_gain_not_below_adam = muon["window_gain"] >= adam["window_gain"]
    muon_last_window_not_below_adam = (
        muon["last_window_mean"] >= adam["last_window_mean"]
    )
    summary["verdict"] = {
        "muon_reward_increased": muon_reward_increased,
        "muon_gain_not_below_adam": muon_gain_not_below_adam,
        "muon_last_window_not_below_adam": muon_last_window_not_below_adam,
        "hard_gate_passed": (
            muon_reward_increased
            and muon_gain_not_below_adam
            and muon_last_window_not_below_adam
        ),
    }
    return summary


def svg(curves: dict[str, list[tuple[int, float]]]) -> str:
    width, height, pad = 800, 480, 60
    xs = [x for curve in curves.values() for x, _ in curve]
    ys = [y for curve in curves.values() for _, y in curve]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1

    def px(x: float) -> float:
        return pad + (x - xmin) * (width - 2 * pad) / (xmax - xmin)

    def py(y: float) -> float:
        return height - pad - (y - ymin) * (height - 2 * pad) / (ymax - ymin)

    colors = {"muon": "#0b84f3", "adam": "#f97316"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#222"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle">DAPO training step</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">reward mean</text>',
        f'<text x="{pad}" y="{pad-18}">reward range {ymin:.6g}..{ymax:.6g}</text>',
    ]
    for index, (name, curve) in enumerate(curves.items()):
        points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in curve)
        color = colors[name]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{width-pad-100}" y="{pad+index*24}" fill="{color}">{name}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--muon", type=Path, required=True)
    parser.add_argument("--adam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--window-size",
        type=int,
        help="points in the disjoint leading/trailing trend windows (default: auto)",
    )
    args = parser.parse_args()

    curves = {"muon": reward_curve(args.muon), "adam": reward_curve(args.adam)}
    summary = summarize_curves(curves, args.window_size)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reward_curves.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "reward_curves.svg").write_text(svg(curves))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

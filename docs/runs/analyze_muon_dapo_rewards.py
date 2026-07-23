#!/usr/bin/env python3
"""Extract comparable DAPO reward curves from VERL JSONL/file logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REWARD_KEYS = (
    "critic/rewards/mean",
    "critic/score/mean",
    "reward/mean",
    "train/reward",
)


def reward_curve(path: Path) -> list[tuple[int, float]]:
    curve: list[tuple[int, float]] = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
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
    args = parser.parse_args()

    curves = {"muon": reward_curve(args.muon), "adam": reward_curve(args.adam)}
    summary = {
        name: {
            "points": [{"step": step, "reward": reward} for step, reward in curve],
            "first": curve[0][1],
            "last": curve[-1][1],
            "gain": curve[-1][1] - curve[0][1],
        }
        for name, curve in curves.items()
    }
    summary["verdict"] = {
        "muon_reward_increased": summary["muon"]["gain"] > 0,
        "muon_gain_not_below_adam": summary["muon"]["gain"] >= summary["adam"]["gain"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reward_curves.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "reward_curves.svg").write_text(svg(curves))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

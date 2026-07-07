#!/usr/bin/env python3
"""Summarize raw per-parameter gradient square sums from correctness artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _family(name: str) -> str:
    if "self_attention" in name or "linear_attn" in name:
        return "gdn"
    if ".mlp." in name or ".moe." in name or "pre_mlp_layernorm" in name:
        return "moe"
    if "embedding" in name:
        return "embedding"
    if (
        "output_layer" in name
        or "final_layernorm" in name
        or ".head." in name
        or name.endswith("module.norm.weight")
    ):
        return "head"
    return "other"


def _summarize(path: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    details = artifact["steps"][0]["grad_fingerprint"]["details"]
    families: dict[str, float] = defaultdict(float)
    parameters = []
    for detail in details:
        square_sum = float(detail["summary"]["square_sum"])
        family = _family(detail["name"])
        families[family] += square_sum
        parameters.append(
            {
                "name": detail["name"],
                "family": family,
                "square_sum": square_sum,
                "norm": math.sqrt(square_sum),
            }
        )
    return {
        "path": str(path),
        "loss": artifact["steps"][0]["loss"]["value"],
        "families": dict(sorted(families.items())),
        "parameters": sorted(parameters, key=lambda item: item["square_sum"], reverse=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([_summarize(path) for path in args.artifacts], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

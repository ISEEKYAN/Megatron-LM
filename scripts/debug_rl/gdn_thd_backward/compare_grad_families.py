#!/usr/bin/env python3
"""Compare parameter-family squared gradient norms from replay logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _records(path: Path, marker: str) -> list[dict[str, float]]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        records.append({key: float(value) for key, value in json.loads(payload).items()})
    if not records:
        raise ValueError(f"{path} has no {marker} records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mlite_log", type=Path)
    parser.add_argument("mcore_log", type=Path)
    args = parser.parse_args()

    mlite = _records(args.mlite_log, "MLITE_GRAD_FAMILY_SQ ")
    mcore = _records(args.mcore_log, "MCORE_GRAD_FAMILY_SQ ")
    count = min(len(mlite), len(mcore))
    families = sorted(set(mlite[0]) & set(mcore[0]) - {"family_total", "reported_total"})
    comparisons = []
    for index in range(count):
        ml_family_total = sum(mlite[index][family] for family in families)
        mc_family_total = sum(mcore[index][family] for family in families)
        ml_reported_total = mlite[index].get("reported_total", ml_family_total)
        mc_reported_total = mcore[index].get("reported_total", mc_family_total)
        row = {
            "mini_step": index + 1,
            "closure": {
                "mlite_family_total": ml_family_total,
                "mlite_reported_total": ml_reported_total,
                "mlite_relative_error": abs(ml_family_total - ml_reported_total)
                / ml_reported_total,
                "mcore_family_total": mc_family_total,
                "mcore_reported_total": mc_reported_total,
                "mcore_relative_error": abs(mc_family_total - mc_reported_total)
                / mc_reported_total,
            },
            "families": {},
        }
        for family in families:
            ml_sq = mlite[index][family]
            mc_sq = mcore[index][family]
            row["families"][family] = {
                "mlite_sq": ml_sq,
                "mcore_sq": mc_sq,
                "norm_ratio": math.sqrt(ml_sq / mc_sq) if mc_sq > 0 else None,
            }
        comparisons.append(row)
    print(
        json.dumps(
            {
                "paired_mini_steps": count,
                "mlite_record_count": len(mlite),
                "mcore_record_count": len(mcore),
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

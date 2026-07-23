#!/usr/bin/env python3
"""Summarize AC#4 three-arm speed+memory JSONL outputs into a markdown table."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _summarize_arm(name: str, path: Path) -> dict:
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError(f"{name}: empty jsonl {path}")
    losses = [float(r["data"]["train/loss"]) for r in rows]
    peak_mem = max(float(r["data"]["perf/max_memory_allocated_gb"]) for r in rows)
    peak_reserved = max(float(r["data"]["perf/max_memory_reserved_gb"]) for r in rows)
    mfus = [float(r["data"].get("train/mfu", 0.0)) for r in rows]
    # Skip warmup step 1 for steady-state speed proxy.
    steady_mfus = mfus[1:] if len(mfus) > 1 else mfus
    return {
        "arm": name,
        "steps": len(rows),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean_steady": statistics.mean(losses[1:]) if len(losses) > 1 else losses[0],
        "peak_mem_gb": peak_mem,
        "peak_reserved_gb": peak_reserved,
        "mfu_mean_steady": statistics.mean(steady_mfus) if steady_mfus else 0.0,
        "path": str(path),
    }


def _md_table(rows: list[dict]) -> str:
    header = (
        "| Arm | Steps | Loss (step1→last) | Mean loss (steady) | Peak alloc GB | Peak reserved GB | Mean MFU (steady) |"
    )
    sep = "| --- | ---: | --- | ---: | ---: | ---: | ---: |"
    body = []
    for r in rows:
        body.append(
            f"| {r['arm']} | {r['steps']} | {r['loss_first']:.4f}→{r['loss_last']:.4f} | "
            f"{r['loss_mean_steady']:.4f} | {r['peak_mem_gb']:.2f} | {r['peak_reserved_gb']:.2f} | "
            f"{r['mfu_mean_steady']:.6f} |"
        )
    return "\n".join([header, sep, *body])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    root = Path(args.run_root)
    arms = {
        "adamw": root / "adamw/q3moe_speed_mem_adamw.jsonl",
        "muon_distopt": root / "muon/q3moe_speed_mem_muon.jsonl",
        "muon_fsdp2": root / "muon_fsdp2/q3moe_speed_mem_muon_fsdp2.jsonl",
    }
    summaries = [_summarize_arm(name, path) for name, path in arms.items()]
    # parity checks
    dist = next(s for s in summaries if s["arm"] == "muon_distopt")
    fsdp = next(s for s in summaries if s["arm"] == "muon_fsdp2")
    adam = next(s for s in summaries if s["arm"] == "adamw")
    loss_gap = abs(dist["loss_mean_steady"] - fsdp["loss_mean_steady"])
    mem_ok = dist["peak_mem_gb"] < adam["peak_mem_gb"] and fsdp["peak_mem_gb"] < adam["peak_mem_gb"]
    text = [
        "# Three-arm speed+memory summary (AC#4)",
        "",
        _md_table(summaries),
        "",
        "## Acceptance checks",
        f"- DistOpt muon vs FSDP2 muon steady mean loss gap: **{loss_gap:.4f}**",
        f"- Muon peak memory < AdamW (both arms): **{mem_ok}** "
        f"(adamw={adam['peak_mem_gb']:.2f}GB, distopt={dist['peak_mem_gb']:.2f}GB, fsdp2={fsdp['peak_mem_gb']:.2f}GB)",
        "",
    ]
    out = "\n".join(text)
    if args.output == "-":
        sys.stdout.write(out)
    else:
        Path(args.output).write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

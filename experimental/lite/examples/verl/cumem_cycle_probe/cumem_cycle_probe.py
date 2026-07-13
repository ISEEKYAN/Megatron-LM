# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Standalone vLLM sleep/wake per-cycle VRAM probe (Branch C, TASK-1.13.8.1).

Isolates the vLLM cumem allocator × PyTorch ``expandable_segments`` interaction
that is the suspected root cause of the colocated-RL per-cycle VRAM net-growth
(see ``docs/cumem-expandable-segments-cycle-leak-analysis.md``). It deliberately
uses **vLLM only** — no verl / mfsdp / actor — so any monotonic device-residency
climb it reproduces is attributable to the sleep/wake + cumem + allocator layer
alone, with zero training-stack confounders.

The decisive experiment is the A0/A1 pair (same script, two env values):
  A0: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   (Branch A's lever)
  A1: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
If the per-cycle device MiB climbs under A0 and is flat under A1, the leak is the
documented expandable×cumem incompatibility (fix: do NOT set expandable globally
for the colocated run). gmu is held at 0.7 (bayan 铁律) so we measure the real
leak, not a KV-shrink mask.

Emits one CSV row per (phase, cycle) to stdout and to ``--csv``; the phases are
measured AWAKE (after wake_up + a short generate) and ASLEEP (after sleep), so a
creep in the *asleep* residency isolates untracked-allocation escape (Issue
#47654 family) from awake-peak growth.

Run via ``run_cumem_cycle_probe.sbatch`` inside the vllm023 container. This file
is stdlib+vLLM only; it does not import torch at module load so ``--help`` works
without a CUDA runtime.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time


def _query_device_memory():
    """Return list of (gpu_index, used_MiB, total_MiB) from nvidia-smi."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, used, total = (x.strip() for x in line.split(","))
        rows.append((int(idx), int(used), int(total)))
    return rows


def _query_proc_memory():
    """Return {pid: used_MiB} for compute apps (per-process device residency)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        return {}
    procs = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, used = (x.strip() for x in line.split(","))
        try:
            procs[int(pid)] = int(used)
        except ValueError:
            continue
    return procs


def _torch_stats():
    import torch

    return {
        "torch_reserved_MiB": torch.cuda.memory_reserved() // (1024**2),
        "torch_allocated_MiB": torch.cuda.memory_allocated() // (1024**2),
        "torch_max_reserved_MiB": torch.cuda.max_memory_reserved() // (1024**2),
        "torch_num_segments": torch.cuda.memory_stats().get(
            "segment.all.current", -1
        ),
        # reserved-minus-allocated gap = fragmentation proxy
        "torch_frag_MiB": (
            torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        )
        // (1024**2),
    }


def _sample(writer, csv_fh, cycle, phase, gpu0_only=True):
    dev = _query_device_memory()
    if gpu0_only:
        dev = [d for d in dev if d[0] == 0]
    tstats = _torch_stats()
    for idx, used, total in dev:
        row = {
            "ts": round(time.time(), 2),
            "cycle": cycle,
            "phase": phase,
            "gpu": idx,
            "device_used_MiB": used,
            "device_total_MiB": total,
            "device_free_MiB": total - used,
            **tstats,
        }
        writer.writerow(row)
    csv_fh.flush()
    d0 = dev[0]
    print(
        f"[cycle {cycle:>2} {phase:>6}] gpu0 used={d0[1]}MiB "
        f"free={d0[2]-d0[1]}MiB torch_reserved={tstats['torch_reserved_MiB']}MiB "
        f"frag={tstats['torch_frag_MiB']}MiB segs={tstats['torch_num_segments']}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to a small HF model (0.5B~4B)")
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--sleep-level", type=int, default=1, choices=(1, 2))
    ap.add_argument("--gpu-mem-util", type=float, default=0.7)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument(
        "--reload-weights",
        action="store_true",
        help="simulate update_weights: reload model weights each cycle "
        "(escalation arm if idle sleep/wake is flat)",
    )
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--csv", default="cumem_cycle_probe.csv")
    args = ap.parse_args()

    # Freshness datum: record the vLLM version and the allocator conf in effect.
    import vllm

    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
    print(f"vllm_version={vllm.__version__}", flush=True)
    print(f"PYTORCH_CUDA_ALLOC_CONF={conf}", flush=True)
    print(
        f"probe_config: model={args.model} cycles={args.cycles} "
        f"sleep_level={args.sleep_level} gmu={args.gpu_mem_util} "
        f"reload_weights={args.reload_weights}",
        flush=True,
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        enable_sleep_mode=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=False,  # keep CUDA graphs ON — they are a suspected accumulator (A4)
    )
    sp = SamplingParams(max_tokens=args.gen_tokens, temperature=0.0)
    prompt = "Explain colocated RL memory management in one paragraph."

    fieldnames = [
        "ts", "cycle", "phase", "gpu",
        "device_used_MiB", "device_total_MiB", "device_free_MiB",
        "torch_reserved_MiB", "torch_allocated_MiB", "torch_max_reserved_MiB",
        "torch_num_segments", "torch_frag_MiB",
    ]
    csv_fh = open(args.csv, "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()

    _sample(writer, csv_fh, 0, "init")

    for cycle in range(1, args.cycles + 1):
        # AWAKE phase: generate a little, then measure the awake residency/peak.
        llm.generate([prompt], sp)
        _sample(writer, csv_fh, cycle, "awake")

        # SLEEP: this is where cumem unmaps tracked handles; untracked allocations
        # (Issue #47654 family) stay resident → asleep creep isolates that escape.
        llm.sleep(level=args.sleep_level)
        _sample(writer, csv_fh, cycle, "asleep")

        # WAKE: cuMemCreate needs fresh physical pages; the OOM site in the real run.
        llm.wake_up()

        if args.reload_weights:
            # Escalation arm: reload weights each cycle to approximate the
            # actor→vLLM update_weights path without pulling in verl/mfsdp.
            try:
                llm.llm_engine.model_executor.collective_rpc(
                    "reload_weights"  # best-effort; not all versions expose this
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] reload_weights rpc unavailable: {exc}", flush=True)

        _sample(writer, csv_fh, cycle, "woke")

    csv_fh.close()
    print(f"done; csv={args.csv}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

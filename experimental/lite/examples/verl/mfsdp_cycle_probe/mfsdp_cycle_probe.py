# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M-FSDP training-side per-cycle retention probe (Branch B, TASK-1.13.8.2).

Branch C (TASK-1.13.8.1) proved the colocated-RL per-cycle VRAM net-growth is
NOT the vLLM sleep/wake path (idle sleep/wake ×20 was 0 MiB/cycle, both
``expandable_segments`` arms flat). bayan's ruling: the residue is on the
**M-FSDP training side** — each resync export cycle leaves something the
allocator does not get back. This probe isolates that, with the vLLM confounder
removed entirely.

It drives the Megatron-Lite runtime directly (no verl trainer, no Ray, no
colocated vLLM) on a cheap proxy and, per cycle, performs the exact training-rank
lifecycle of one RL step:

    wake   = runtime.to(handle, "cuda")          # onload actor params+optim
    export = drain runtime.export_weights(...)   # the resync / update_weights all-gather
    sleep  = runtime.to(handle, "cpu")           # offload actor

drained through the production ``stream_export_with_empty_cache`` so the export
buffer is flushed to the driver exactly as in the real colocated run. It samples
``torch.cuda`` memory at each phase every cycle and, via
``torch.cuda.memory._record_memory_history`` / ``_dump_snapshot``, captures the
allocation call-stacks so the retained bytes can be attributed to a source line.

Host-RAM axis (TASK-1.13.8.6): the real 32-card DAPO run's GPU export peak is
already bounded (memcurve flat at 22.148 GiB across cycles), yet it is SIGKILLed
at ~37 min by host RSS climbing monotonically 157→282 GiB per resync cycle — a
leak the device-side ``torch.cuda`` metrics cannot see. So each sample also
records process ``rss_MiB`` (the SIGKILL quantity, ``VmRSS``) and a live CPU-
tensor census ``host_tensor_MiB`` (distinct storages, deduped by ``data_ptr``,
via the production ``summarize_host_storages``). The decisive read is the
``asleep``-phase per-cycle slope: mfsdp rss climbing while fsdp2 plateaus
reproduces the leak cheaply (2 GPU, <1 GPU-h), and ``host_tensor_MiB`` rising
with rss point-names a retained tensor whereas rss rising with a flat tensor
total point-names non-tensor host memory (pinned pool / malloc fragmentation).

The decisive comparison is the fsdp2 gold standard A/B: run the identical vehicle
with ``--optimizer mfsdp`` and ``--optimizer fsdp2``; ``mfsdp − fsdp2`` per-cycle
retention slope, and the diff of the live allocation stacks, is the answer. The
secondary axis is ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:{True,False}``.

Heavy imports (torch, megatron.lite, bench) are deferred so ``--help`` and
``py_compile`` work without a CUDA runtime. Launch via
``run_mfsdp_cycle_probe.sbatch`` under torchrun inside the verl.vllm023 image.
See ``docs/mfsdp-training-side-cycle-retention-probe.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time


# ── memory sampling (torch imported lazily inside) ──────────────────────────


def _device_used_mib(local_rank: int) -> int:
    """This rank's device residency in MiB from nvidia-smi (best-effort)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:  # noqa: BLE001 - nvidia-smi absent → fall back to torch view
        return -1
    for line in out.strip().splitlines():
        idx, used = (x.strip() for x in line.split(","))
        if int(idx) == local_rank:
            return int(used)
    return -1


def _torch_mem(local_rank: int):
    import torch

    stats = torch.cuda.memory_stats()
    return {
        "torch_alloc_MiB": torch.cuda.memory_allocated() // (1024**2),
        "torch_reserved_MiB": torch.cuda.memory_reserved() // (1024**2),
        "torch_max_reserved_MiB": torch.cuda.max_memory_reserved() // (1024**2),
        "torch_max_alloc_MiB": torch.cuda.max_memory_allocated() // (1024**2),
        "torch_num_segments": stats.get("segment.all.current", -1),
        # reserved-not-allocated = fragmentation / cached-not-freed proxy
        "torch_frag_MiB": (
            torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        )
        // (1024**2),
        "device_used_MiB": _device_used_mib(local_rank),
    }


_MIB = 1024 * 1024


def _rss_mib() -> float:
    """Whole-process resident set in MiB (``VmRSS`` from ``/proc/self/status``).

    RSS is the quantity the OOM-killer / SIGKILL acts on. The per-cycle RSS slope
    at the offloaded ``asleep`` phase is the host-leak signal that the device-side
    ``torch.cuda`` metrics cannot see: the real 32-card DAPO run climbs host RAM
    157→282 GiB across resync cycles with no GPU-side growth (memcurve flat at
    22.148 GiB export peak), so the SIGKILL is pure host residency.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB → MiB
    except OSError:
        pass
    return -1.0


def _cpu_storage_entries():
    """Raw ``(data_ptr, nbytes, shape)`` for every live CPU torch tensor.

    Walks ``gc`` for host-resident tensors so a growing distinct-storage total
    point-names a retained CPU tensor (offload buffer / optimizer state / export
    residue). Deduplication by ``data_ptr`` — so aliased views over one offload
    buffer count once — is deferred to the pure ``dedup_host_storage_entries``
    helper so it is CPU unit-testable without torch.
    """
    import gc

    import torch

    out = []
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type != "cpu":
                continue
            storage = obj.untyped_storage()
            out.append((storage.data_ptr(), storage.nbytes(), tuple(obj.shape)))
        except Exception:  # noqa: BLE001 - some gc objects raise on attribute access
            continue
    return out


def _host_mem() -> dict:
    """Host-RAM census: process RSS plus the distinct live CPU-tensor total.

    ``rss_MiB`` climbing while ``host_tensor_MiB`` stays flat means the residue is
    non-tensor host memory (pinned-host pool / malloc-arena fragmentation) rather
    than a retained tensor — the two fixes differ, so the probe must distinguish
    them. Reuses the production census aggregation (``summarize_host_storages``)
    so the proxy and the real resync report the same quantity.
    """
    from mfsdp_cycle_analysis import dedup_host_storage_entries
    from verl_mlite.resync_export import summarize_host_storages

    deduped = dedup_host_storage_entries(_cpu_storage_entries())
    census = summarize_host_storages(deduped, top_n=8)
    top = census.get("top", [])
    top_str = ";".join(
        f"{entry['nbytes'] / _MIB:.0f}MiB{tuple(entry['shape'])}" for entry in top[:3]
    )
    return {
        "rss_MiB": round(_rss_mib(), 1),
        "host_tensor_MiB": round(census["total_gib"] * 1024.0, 1),
        "host_tensor_count": census["count"],
        "host_top": top_str,
    }


def _sample(writer, fh, rank, local_rank, cycle, phase):
    row = {"ts": round(time.time(), 2), "rank": rank, "cycle": cycle, "phase": phase}
    row.update(_torch_mem(local_rank))
    row.update(_host_mem())
    writer.writerow(row)
    fh.flush()
    if rank == 0:
        print(
            f"[cycle {cycle:>2} {phase:>9}] alloc={row['torch_alloc_MiB']}MiB "
            f"reserved={row['torch_reserved_MiB']}MiB frag={row['torch_frag_MiB']}MiB "
            f"device={row['device_used_MiB']}MiB "
            f"rss={row['rss_MiB']}MiB host_tensor={row['host_tensor_MiB']}MiB"
            f"(n={row['host_tensor_count']}) top={row['host_top']}",
            flush=True,
        )
    return row


# ── vehicle build (reuses the validated bench config builder) ───────────────


def _bench_cli_config(args):
    from examples.bench.bench import BenchCliConfig

    impl_cfg = {"optimizer": args.optimizer, "use_thd": True}
    # The bench default sets use_precision_aware_optimizer=True, but the mfsdp
    # (megatron_fsdp) path in this image guards against it — precision-aware
    # segfaults in transformer_engine::multi_tensor_scale_cuda
    # (validate_precision_aware_disabled). Disable it for BOTH backends so the
    # gold-standard A/B differs only in the optimizer backend (mfsdp vs fsdp2),
    # not also in the optimizer precision mode — an aligned baseline. Applied via
    # the bench override_optimizer_json channel (setattr onto the OptimizerConfig,
    # honored by both the guard and megatron_wrap's real optimizer args).
    optimizer_overrides = {"use_precision_aware_optimizer": False}
    return BenchCliConfig(
        backend="mlite",
        hf_path=args.hf_path,
        model_name=args.model_name,
        impl="lite",
        override_optimizer_json=json.dumps(optimizer_overrides),
        tp=args.tp,
        etp=args.etp,
        ep=args.ep,
        pp=args.pp,
        cp=args.cp,
        seq_len=args.seq_len,
        num_microbatches=args.num_microbatches,
        use_thd=True,
        skip_load_hf_weights=args.skip_load_hf_weights,
        keep_experts=args.keep_experts,
        truncate_layers=args.truncate_layers,
        disable_mtp=True,
        impl_cfg_json=json.dumps(impl_cfg),
    )


def _build_runtime(args):
    """Build the mlite runtime+model with the chosen optimizer backend."""
    from examples.bench.bench import build_runtime_config
    from megatron.lite.runtime import create_runtime

    rt_cfg = build_runtime_config(_bench_cli_config(args))
    rt = create_runtime(rt_cfg)
    handle = rt.build_model()
    return rt, handle


def _config_only(args) -> int:
    """Zero-GPU init-chain gate: build the runtime config for the chosen backend
    and confirm the module import + config-construction chain is intact, without
    touching CUDA. Mirrors the mandated CONFIG_ONLY gate for the GPU matrix."""
    from examples.bench.bench import build_runtime_config

    # Confirm the resync-export and analysis modules import (probe deps).
    import mfsdp_cycle_analysis  # noqa: F401
    from verl_mlite.resync_export import stream_export_with_empty_cache  # noqa: F401

    rt_cfg = build_runtime_config(_bench_cli_config(args))
    backend_cfg = rt_cfg.backend_cfg
    print(
        f"CONFIG_ONLY_OK optimizer={args.optimizer} model={args.model_name} "
        f"impl_cfg.optimizer={backend_cfg.impl_cfg.get('optimizer')} "
        f"truncate_layers={args.truncate_layers} keep_experts={args.keep_experts}",
        flush=True,
    )
    return 0


def _parse_expandable(conf: str):
    """Extract the ``expandable_segments`` bool from a PYTORCH_CUDA_ALLOC_CONF string."""
    for part in str(conf).split(","):
        key, _, val = part.partition(":")
        if key.strip() == "expandable_segments":
            return val.strip().lower() in ("true", "1", "yes", "on")
    return None


def _combine(args) -> int:
    """Zero-GPU: fold the per-arm summaries into the mfsdp-fsdp2 gold-standard A/B.

    Reads every ``*-summary.json`` in ``--out-dir`` (each written by one probe
    arm), pairs mfsdp vs fsdp2 within each ``expandable_segments`` value, and
    writes ``gold-standard-AB.json`` with the ``mfsdp − fsdp2`` per-cycle
    retention delta and the live-stack diff. No CUDA — pure arithmetic on the
    summaries, so it runs on the login node / CPU."""
    import glob

    from mfsdp_cycle_analysis import combine_gold_standard

    paths = sorted(glob.glob(os.path.join(args.out_dir, "*-summary.json")))
    summaries = []
    for p in paths:
        if p.endswith("gold-standard-AB.json"):
            continue
        with open(p) as fh:
            summaries.append(json.load(fh))
    if not summaries:
        print(f"[FATAL] --combine: no *-summary.json in {args.out_dir}", flush=True)
        return 5

    combined = combine_gold_standard(summaries)
    out_path = os.path.join(args.out_dir, "gold-standard-AB.json")
    with open(out_path, "w") as fh:
        json.dump(combined, fh, indent=2, default=str)

    pairs = combined["gold_standard_AB"]
    if not pairs:
        print(
            f"[FATAL] --combine: no mfsdp/fsdp2 arm pair shares an expandable_segments "
            f"value; got {[s.get('tag') for s in summaries]} → {out_path}",
            flush=True,
        )
        return 6
    for pr in pairs:
        asleep = pr["delta"]["asleep"]["device_used_MiB"]
        top = pr["stack_diff"][0] if pr["stack_diff"] else ("<none>", 0.0)
        print(
            f"[GOLD-STANDARD expandable={pr['expandable_segments']}] "
            f"mfsdp−fsdp2 asleep device MiB/cycle={asleep['delta_mib_per_cycle']:.2f} "
            f"(mfsdp={asleep['mfsdp_slope_mib_per_cycle']:.2f} "
            f"fsdp2={asleep['fsdp2_slope_mib_per_cycle']:.2f}); "
            f"top mfsdp-only stack={top[0]} (+{top[1]:.1f} MiB)",
            flush=True,
        )
    if combined["unpaired_arms"]:
        print(f"[warn] --combine: unpaired arms {combined['unpaired_arms']}", flush=True)
    print(f"[COMBINE_OK] {out_path}", flush=True)
    return 0


def _data_iter(handle, args, device):
    """A minimal packed-batch stream (mirrors bench/session bench data)."""
    import torch
    from megatron.lite.runtime.contracts.data import PackedBatch

    model_cfg = handle._extras.get("model_cfg")
    vocab = int(getattr(model_cfg, "vocab_size", 151936)) if model_cfg is not None else 151936
    seed = args.seed + handle.dp_rank
    g = torch.Generator(device=device).manual_seed(seed)
    seq_lens = torch.tensor([args.seq_len], dtype=torch.int64, device=device)
    while True:
        yield PackedBatch(
            input_ids=torch.randint(0, vocab, (args.seq_len,), device=device, generator=g),
            labels=torch.randint(0, vocab, (args.seq_len,), device=device, generator=g),
            seq_lens=seq_lens.clone(),
        )


def _warmup_train(rt, handle, args, device):
    """A few real train steps so Adam moments + grads exist before cycling.

    Without populated optimizer state the export all-gather and the offload/
    onload move near-nothing, which would understate the per-cycle residue.
    """
    if args.warmup_steps <= 0:
        return
    it = _data_iter(handle, args, device)
    with rt.train_mode(handle):
        for _ in range(args.warmup_steps):
            rt.zero_grad(handle)
            rt.forward_backward(handle, it, loss_fn=None, num_microbatches=args.num_microbatches)
            rt.optimizer_step(handle)
            rt.lr_scheduler_step(handle)


def _export_kwargs(args):
    kw = {}
    if args.model_name == "qwen3_5":
        kw["target"] = "vllm"
    if args.export_dtype:
        kw["export_dtype"] = args.export_dtype
    return kw


def _drain_export(rt, handle, args):
    """Run one resync export exactly as the verl integration layer does."""
    import torch
    from verl_mlite.resync_export import stream_export_with_empty_cache

    threshold = int(args.export_empty_cache_gib * (1024**3))
    generator = rt.export_weights(handle, **_export_kwargs(args))
    streamed = stream_export_with_empty_cache(generator, threshold, torch.cuda.empty_cache)
    n_tensors = 0
    for _name, _tensor in streamed:
        n_tensors += 1
    return n_tensors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--optimizer",
        choices=("mfsdp", "fsdp2"),
        help="required for a probe/config-only run; omit only with --combine",
    )
    ap.add_argument("--hf-path", help="HF model dir (config read; weights random if --skip-load-hf-weights); required unless --combine")
    ap.add_argument("--model-name", default="qwen3_5", choices=("qwen3_5", "qwen3_moe"))
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--etp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--num-microbatches", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--truncate-layers", type=int, default=None)
    ap.add_argument("--keep-experts", type=int, default=None)
    ap.add_argument("--skip-load-hf-weights", action="store_true")
    ap.add_argument("--export-dtype", default=None)
    ap.add_argument("--export-empty-cache-gib", type=float, default=4.0)
    ap.add_argument("--record-history-entries", type=int, default=200000)
    ap.add_argument(
        "--top-stacks",
        type=int,
        default=15,
        help="top-N live allocation call-stacks to attribute retained bytes to",
    )
    ap.add_argument(
        "--snapshot-cycles",
        default="4,19",
        help="comma-separated cycle indices at which to dump a memory snapshot for attribution",
    )
    ap.add_argument("--out-dir", default=".", help="directory for per-rank csv/json/pickle")
    ap.add_argument("--tag", default="probe", help="arm tag, e.g. mfsdp-expTrue")
    ap.add_argument(
        "--config-only",
        action="store_true",
        help="zero-GPU gate: build the runtime config + import deps, then exit (no CUDA)",
    )
    ap.add_argument(
        "--combine",
        action="store_true",
        help="zero-GPU: read per-arm *-summary.json in --out-dir and emit the "
        "mfsdp-fsdp2 gold-standard A/B (no CUDA, no --optimizer)",
    )
    args = ap.parse_args()

    if args.combine:
        return _combine(args)
    if not args.optimizer:
        ap.error("--optimizer is required unless --combine")
    if not args.hf_path:
        ap.error("--hf-path is required unless --combine")
    if args.config_only:
        return _config_only(args)

    # Heavy imports now (after --help / arg parsing is safely past).
    import torch

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # A CPU run cannot produce the memory-history / snapshot evidence this probe
    # exists to capture; refuse it rather than emit successful-looking empty
    # output (moe pre-GPU gate BLOCKER: no valid-looking runs without evidence).
    if not torch.cuda.is_available():
        print("[FATAL] CUDA unavailable; probe requires a GPU for valid evidence", flush=True)
        return 3
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")
    snapshot_cycles = {int(x) for x in str(args.snapshot_cycles).split(",") if x.strip()}
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{args.tag}-rank{rank}.csv")
    if rank == 0:
        print(f"probe tag={args.tag} optimizer={args.optimizer} model={args.model_name}", flush=True)
        print(f"PYTORCH_CUDA_ALLOC_CONF={conf} cycles={args.cycles} export_empty_cache_gib={args.export_empty_cache_gib}", flush=True)

    # Record allocation history so live blocks carry their stacks for attribution.
    # Hard failure: without this the end-of-run snapshot has no frames and the
    # attribution (the probe's required evidence) is invalid, not merely absent.
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", stacks="all", max_entries=args.record_history_entries
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] _record_memory_history failed: {exc}", flush=True)
        return 4

    rt, handle = _build_runtime(args)
    _warmup_train(rt, handle, args, device)

    fieldnames = [
        "ts", "rank", "cycle", "phase",
        "torch_alloc_MiB", "torch_reserved_MiB", "torch_max_reserved_MiB",
        "torch_max_alloc_MiB", "torch_num_segments", "torch_frag_MiB", "device_used_MiB",
        "rss_MiB", "host_tensor_MiB", "host_tensor_count", "host_top",
    ]
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()

    rows = []
    torch.cuda.synchronize()
    rows.append(_sample(writer, fh, rank, local_rank, 0, "init"))

    def _dump(cycle):
        # Hard failure: a snapshot we asked for but could not write means the
        # attribution evidence for that cycle is silently missing (moe BLOCKER).
        path = os.path.join(args.out_dir, f"{args.tag}-rank{rank}-cycle{cycle}.pickle")
        try:
            torch.cuda.memory._dump_snapshot(path)
        except Exception as exc:
            raise RuntimeError(f"_dump_snapshot cycle {cycle} failed: {exc}") from exc

    for cycle in range(1, args.cycles + 1):
        # wake = onload actor params + optimizer
        rt.to(handle, "cuda", model=True, optimizer=True, grad=True)
        torch.cuda.synchronize()
        rows.append(_sample(writer, fh, rank, local_rank, cycle, "woke"))

        # update_weights = resync export all-gather, drained to driver
        rt.to(handle, "cuda", model=True, optimizer=False, grad=False)
        n_tensors = _drain_export(rt, handle, args)
        torch.cuda.synchronize()
        r = _sample(writer, fh, rank, local_rank, cycle, "exported")
        r["n_export_tensors"] = n_tensors
        rows.append(r)  # exported-phase retention is computed from these rows

        # sleep = offload actor; residue that survives offload is the leak
        rt.to(handle, "cpu", model=True, optimizer=True, grad=True)
        torch.cuda.synchronize()
        rows.append(_sample(writer, fh, rank, local_rank, cycle, "asleep"))

        if cycle in snapshot_cycles:
            _dump(cycle)

    fh.close()

    # End-of-run pickle (offline memory-viz) for every arm; the residue that
    # survived the last offload is what these snapshots hold.
    _dump("final")

    # ── on-node analysis (rank 0): slope + attribution ──────────────────────
    if rank == 0:
        from mfsdp_cycle_analysis import (
            per_cycle_retention_mib,
            top_live_allocation_stacks,
        )

        # Attribution: turn the end-of-run live snapshot into the top-N retained
        # call-stacks. Empty here means the required evidence was NOT produced
        # (bad _record_memory_history, wrong torch snapshot shape) — fail loudly
        # rather than emit a summary that looks fine but attributes nothing.
        try:
            snap = torch.cuda.memory._snapshot()
        except Exception as exc:  # noqa: BLE001
            print(f"[FATAL] _snapshot failed; no attribution evidence: {exc}", flush=True)
            return 4
        stacks = top_live_allocation_stacks(snap, top_n=args.top_stacks)
        if not stacks:
            print(
                "[FATAL] snapshot yielded no live allocation stacks; attribution "
                "evidence missing (check _record_memory_history / torch version)",
                flush=True,
            )
            return 4
        live_stacks = [
            {
                "frames": list(s.frames),
                "top_frame": s.top_frame(),
                "retained_bytes": s.retained_bytes,
                "retained_mib": round(s.retained_mib, 3),
                "num_blocks": s.num_blocks,
            }
            for s in stacks
        ]

        summary = {
            "tag": args.tag,
            "optimizer": args.optimizer,
            "model_name": args.model_name,
            "cycles": args.cycles,
            "pytorch_cuda_alloc_conf": conf,
            "expandable_segments": _parse_expandable(conf),
            "retention": {
                phase: {
                    metric: per_cycle_retention_mib(rows, phase=phase, metric=metric)
                    for metric in (
                        "device_used_MiB",
                        "torch_reserved_MiB",
                        "torch_alloc_MiB",
                        # host-RAM axis: rss_MiB is the SIGKILL quantity, and
                        # host_tensor_MiB isolates retained-tensor vs non-tensor
                        # (pinned pool / fragmentation) host growth.
                        "rss_MiB",
                        "host_tensor_MiB",
                    )
                }
                for phase in ("woke", "exported", "asleep")
            },
            "live_stacks": live_stacks,
        }
        summary_path = os.path.join(args.out_dir, f"{args.tag}-summary.json")
        with open(summary_path, "w") as sf:
            json.dump(summary, sf, indent=2, default=str)
        asleep = summary["retention"]["asleep"]["device_used_MiB"]["slope_mib_per_cycle"]
        woke = summary["retention"]["woke"]["torch_reserved_MiB"]["slope_mib_per_cycle"]
        asleep_rss = summary["retention"]["asleep"]["rss_MiB"]["slope_mib_per_cycle"]
        asleep_htensor = summary["retention"]["asleep"]["host_tensor_MiB"]["slope_mib_per_cycle"]
        top = live_stacks[0]
        print(
            f"[RESULT tag={args.tag}] asleep device MiB/cycle={asleep:.2f} "
            f"woke torch_reserved MiB/cycle={woke:.2f} "
            f"asleep rss MiB/cycle={asleep_rss:.2f} "
            f"asleep host_tensor MiB/cycle={asleep_htensor:.2f} "
            f"top-live-stack={top['top_frame']} ({top['retained_mib']:.1f} MiB) "
            f"summary={summary_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

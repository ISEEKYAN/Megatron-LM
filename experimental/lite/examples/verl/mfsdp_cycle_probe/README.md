# M-FSDP training-side per-cycle retention probe (Branch B)

Isolates the colocated-RL per-cycle VRAM net-growth to the **M-FSDP training
side**. Branch C (`../cumem_cycle_probe/`) already proved the vLLM sleep/wake
path is flat (0 MiB/cycle, both `expandable_segments` arms), so this probe
removes vLLM entirely and drives the Megatron-Lite runtime directly on a cheap
proxy, replaying the training-rank lifecycle of one RL step per cycle:

```
wake   = runtime.to(handle, "cuda")           # onload actor params + optimizer
export = drain runtime.export_weights(...)    # the resync / update_weights all-gather
sleep  = runtime.to(handle, "cpu")            # offload actor
```

The export is drained through the production
`verl_mlite.resync_export.stream_export_with_empty_cache`, so the M-FSDP
all-gather buffer is flushed to the driver exactly as in the real run. Each phase
is sampled every cycle (`torch.cuda` stats + nvidia-smi device MiB) and
`torch.cuda.memory._record_memory_history` / `_dump_snapshot` capture the
allocation call-stacks for attribution.

## The experiment

Gold-standard A/B (main axis) + `expandable_segments` (secondary axis) = 2x2:

| arm | optimizer | `PYTORCH_CUDA_ALLOC_CONF` |
|-----|-----------|---------------------------|
| `mfsdp-expTrue`  | mfsdp | expandable_segments:True  |
| `mfsdp-expFalse` | mfsdp | expandable_segments:False |
| `fsdp2-expTrue`  | fsdp2 | expandable_segments:True  |
| `fsdp2-expFalse` | fsdp2 | expandable_segments:False |

The answer = `mfsdp − fsdp2` per-cycle retention slope (MiB/cycle) at the
`asleep` phase (residue that survives offload) plus the diff of the live
allocation stacks (`mfsdp_cycle_analysis.diff_live_stacks`) — which call-stack
mfsdp holds that fsdp2 does not.

Each arm writes `<tag>-summary.json` carrying its per-phase retention slopes AND
its top-N live allocation stacks (attribution generated in-process from
`torch.cuda.memory._snapshot()`, not merely dumped). Missing evidence is a hard
failure (non-zero rc), never a silent warning. After the 4 arms, a zero-GPU
`--combine` pass folds them into `gold-standard-AB.json`: `mfsdp − fsdp2`
per-cycle retention + live-stack diff, paired by `expandable_segments`.

## Files

- `mfsdp_cycle_probe.py` — the harness (heavy imports lazy; `--help`/`py_compile`
  work without CUDA). Drives one arm; `--optimizer {mfsdp,fsdp2}`.
- `mfsdp_cycle_analysis.py` — pure, CPU-testable analysis: per-cycle retention
  slope, `mfsdp − fsdp2` delta, top-N live allocation stacks, stack diff.
- `test_mfsdp_cycle_analysis.py` — CPU unit tests for the analysis core.
- `run_mfsdp_cycle_probe.sbatch` + `run_mfsdp_cycle_probe_inner.sh` — fire the
  2x2 matrix on cw (verl.vllm023 image, staged `mlite-133413497` tree with the
  DoubleBuffer fix, qwen3_moe `Qwen3-30B-A3B` truncated to ~0.7B, random init).

## Run

CPU (no GPU): `python -m pytest test_mfsdp_cycle_analysis.py -q` and
`python mfsdp_cycle_probe.py --help`.

GPU (cw): stage this dir to `$BASE/branchb-mfsdp/mfsdp_cycle_probe` on lustre,
then `sbatch --export=ALL,NGPUS=2 run_mfsdp_cycle_probe.sbatch`. Budget: 2 GPU x
<=30 min = <=1 GPU-h (AC#4 ceiling 2 GPU-h). Evidence (per-arm CSV, summary
JSON with live-stack attribution, memory-snapshot pickles, and the combined
`gold-standard-AB.json`) lands in `$BASE/branchb-mfsdp/evidence/$JOBID`.

The FSDP DoubleBuffer export retention is per-rank, so `dp=2` reproduces it;
`NGPUS=8` matches the validated proxy topology for a wider check.

Both backends run with `use_precision_aware_optimizer=False` (passed via bench
`override_optimizer_json`): the mfsdp path guards against precision-aware in the
`verl.vllm023` image (it segfaults in `transformer_engine::multi_tensor_scale_cuda`),
so matching fsdp2 keeps the A/B differing only in the optimizer backend.

## Results (job 13910983)

`sacct` COMPLETED, ExitCode 0:0, elapsed 06:05 → ~0.20 GPU-h. Evidence committed
under `evidence/13910983/`. Full analysis + interpretation in
`../../../docs/mfsdp-training-side-cycle-retention-probe.md`.

Per-arm per-cycle slope (cycles 4–20, warmup dropped):

| arm | asleep device MiB/cycle | woke reserved MiB/cycle | end-of-run live MiB | rc |
|-----|--|--|--|--|
| mfsdp-expTrue  | 0.00 | 0.00 | 224.5  | OK |
| mfsdp-expFalse | 0.00 | 0.00 | 224.5  | OK |
| fsdp2-expTrue  | 0.00 | 0.00 | 1555.6 | OK |
| fsdp2-expFalse | 0.00 | 0.00 | 1558.6 | OK |

Gold standard `mfsdp − fsdp2` (paired by `expandable_segments`): **0.00 MiB/cycle**
at every phase, both arms. mfsdp holds ~1.33 GiB *less* steady residue than fsdp2;
no mfsdp-only stack survives offload. **The M-FSDP training side is flat per
cycle** — with Branch C's flat vLLM side, neither isolated subsystem reproduces
the real colocated per-cycle growth on this proxy.

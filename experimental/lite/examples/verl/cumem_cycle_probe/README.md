# Branch C — cumem × expandable_segments per-cycle VRAM probe

Cheap (<1 GPU-h) discriminating experiment for TASK-1.13.8.1. Isolates the
suspected root cause of the colocated-RL per-cycle VRAM net-growth — the vLLM
CuMemAllocator (sleep/wake) × PyTorch `expandable_segments` incompatibility —
with a **vLLM-only** sleep/wake loop (no verl/mfsdp/actor confounders).

Full analysis + hypotheses + decision tree:
`../../../docs/cumem-expandable-segments-cycle-leak-analysis.md`.

## What it does
`cumem_cycle_probe.py` loads a small model in vLLM with `enable_sleep_mode=True`,
then loops `generate → sleep → wake_up` for N cycles, sampling device residency
(`nvidia-smi`) and torch allocator stats at `awake` / `asleep` / `woke` phases
each cycle, writing one CSV row per sample. gmu is held at **0.7** (bayan 铁律).

## The decisive run (A0/A1)
`run_cumem_cycle_probe.sbatch` runs two arms back-to-back on one GPU, the only
variable being the allocator conf:
- **A0**: `expandable_segments:True` (Branch A's global lever)
- **A1**: `expandable_segments:False`

Read the verdict off `device_used_MiB` vs `cycle`:
- A0 climbs monotonically, A1 flat ⇒ **expandable×cumem is the leak** (H1); fix =
  do NOT set `expandable_segments:True` globally for the colocated收口 run
  (reverses Branch A's "primary lever" — needs moe gate + bayan).
- Both flat (mean), fragmentation (`torch_frag_MiB` / `device_free` largest gap)
  worsens ⇒ **F1 fragmentation**, not a true leak.
- A1 still climbs ⇒ leak is not expandable-driven; escalate to `--reload-weights`
  (approximate update_weights) then to a full colocated actor+vLLM proxy (H2/H3).
- `asleep`-phase residency climbing ⇒ untracked-allocation escape (vLLM Issue
  #47654 family, H2).

## Run
```sh
# from a cw/oci login node (Slurm; GPU 铁律 — never on a login node directly)
sbatch --export=ALL,MODEL=/lustre/.../Qwen3-0.6B \
  experimental/lite/examples/verl/cumem_cycle_probe/run_cumem_cycle_probe.sbatch
```
Escalation arm: add `,RELOAD_WEIGHTS=1`. Larger model if 0.6B doesn't stress the
cliff: `,MODEL=/lustre/.../Qwen3-4B`.

## Prereqs (ignition-prep)
- A small HF model on lustre (`$MODEL/config.json` must exist inside the mount).
- First line of stdout records `vllm_version=` and `PYTORCH_CUDA_ALLOC_CONF=` —
  the freshness datum that decides whether the container's vLLM has the hard
  `assert` (old) or the pool-context toggle (new) for expandable_segments.

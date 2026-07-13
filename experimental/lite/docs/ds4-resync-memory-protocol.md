# DeepSeek-V4 rollout resync memory protocol

The rollout weight sync materialises full-precision parameters on the GPU: the
actor's `get_per_tensor_param` reloads its parameter shard and the model adapter
streams a per-tensor all-gather across TP/EP/PP into the colocated vLLM worker.
When the actor's optimizer moment tensors and gradient buffers are still
resident, that export peak stacks on top of training-only memory and can OOM the
colocated ranks (observed on the 128-GPU PP4/EP8/CP4 run: GPU0 OOM at the first
`update_weights` all-gather).

## Protocol

`MegatronLiteEngine.get_per_tensor_param` wraps the export with an
evict → export → restore protocol, with per-tensor residency control on the
export stream itself:

1. **Evict** (eager, before the gather): offload optimizer moment tensors to CPU
   and release DDP gradient-buffer GPU storage, recording their entry state.
2. **Export**: reload the parameter shard and stream the per-tensor export. The
   generator body runs lazily in the consumer (vLLM `update_weights`). After
   **each** exported tensor is consumed, the actor drops its reference and calls
   `torch.cuda.empty_cache()` so the next tensor's export all-gather does not
   stack on top of the previous full tensor's allocation; CUDA peak stats are
   reset per tensor so the worst single-tensor peak is captured.
3. **Restore** (in the generator's `finally`, after the stream drains): re-load
   gradient buffers and optimizer state to whatever device they were on at
   entry, keeping VERL's offload bookkeeping consistent.

The run also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (sbatch,
propagated into the Ray actor env) so the allocator can grow/shrink segments
instead of fragmenting across the resync peak.

**Note on the evict lever (empirically null for the DS4 128-GPU config).** On
the 128-GPU PP4/EP8/CP4 run, VERL already offloads optimizer state and DDP grad
buffers for the rollout phase, so `optimizer_states_on_gpu` /
`model_grads_resident` both report false and the evict step is a no-op
(`memory_allocated()==0` at `resync/enter`). The evict path is kept as a
correctness-preserving safety net for configs that *do* keep training state
resident; for DS4-128 the load-bearing lever is the **per-tensor
`empty_cache` + `expandable_segments`** residency control that caps the actor's
export-time footprint, not the optimizer/grad eviction.

Primitives live in `megatron/lite/runtime/megatron_utils.py`
(`optimizer_states_on_gpu`, `model_grads_resident`, `free_grad_buffers`,
`cuda_mem_snapshot`).

## Memory curve evidence

Every resync emits a per-rank line to stdout:

```
MLITE_RESYNC_MEMCURVE rank=<r> opt_offloaded=<bool> grad_freed=<bool> \
  resync/enter=<GiB> resync/after_optimizer_offload=<GiB> \
  resync/after_grad_free=<GiB> resync/export_begin=<GiB> \
  resync/export_end=<GiB> resync/restore=<GiB> \
  worst_tensor=<name> worst_tensor_peak_gib=<GiB> \
  export_peak_max_alloc_gib=<GiB>
```

Set `MLITE_RESYNC_MEMLOG_PATH=/path/to/curve.jsonl` to also append the full
snapshot list (with `max_allocated_gib`) plus `worst_tensor` /
`worst_tensor_peak_gib` as one JSON record per resync per rank.
Because peak stats are reset per exported tensor, the coarse `resync/*`
snapshots understate the true peak; `worst_tensor_peak_gib` is the largest
single-tensor export peak (attributed to the tensor whose materialisation
caused it) and `export_peak_max_alloc_gib` is the max of that and the curve
snapshots. On an OOM this pins the single-tensor lower bound — the minimum
headroom any resync-capable geometry must leave on the colocated GPU.

## Fail-fast smoke resync

Set `MLITE_RESYNC_SMOKE_EXIT_AFTER=<n>` so the actor exits(0) once the colocated
resync peak has survived `n` times (barrier-synchronized across ranks, emitting
`MLITE_RESYNC_SMOKE_COMPLETE`). This lets a proxy report a verdict at the first
resync instead of running a full training step. Off by default; an export
failure (OOM) still propagates as a hard failure and is never masked.

## Why there is no 8-GPU proxy — the smoke gate runs at full scale

The production geometry (TP1/PP4/CP4/EP8 = 128 ranks) **cannot shrink to a
faithful single-node proxy**: PP4×CP4 is what slices the layers 16× so each GPU
holds a fraction of the model at init. Any 8-GPU geometry puts more of the model
on GPU0 than the 128-GPU layout does, so a proxy dies at actor `init` (before it
ever reaches a resync) rather than at the resync peak we want to test. There is
no faithful sub-128 reproduction of the colocated resync footprint.

The gate therefore runs at full 128-GPU scale but is bounded by the fail-fast
smoke resync: set `MLITE_RESYNC_SMOKE_EXIT_AFTER=1` and
`MLITE_RESYNC_MEMLOG_PATH` in the container env so the actor triggers one real
colocated resync immediately after init and exits(0) if the export peak
survives. The verdict lands in the first ~10 minutes and caps the failure cost
at ~20 GPU-h. Watch for `MLITE_RESYNC_MEMCURVE` on the first resync:

- **green** — every rank prints the curve and `MLITE_RESYNC_SMOKE_COMPLETE`;
  clear the smoke-exit var and relaunch for the real training run.
- **red (OOM)** — the export still fails; `worst_tensor` /
  `worst_tensor_peak_gib` on the surviving ranks pin the single-tensor lower
  bound, which is the input to the next mitigation decision.

Production runs leave `MLITE_RESYNC_SMOKE_EXIT_AFTER` unset (the smoke-exit and
per-resync JSONL are opt-in; per-tensor `empty_cache` runs unconditionally
during resync export, which is outside any hot loop).

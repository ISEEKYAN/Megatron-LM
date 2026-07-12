# DeepSeek-V4 rollout resync memory protocol

The rollout weight sync materialises full-precision parameters on the GPU: the
actor's `get_per_tensor_param` reloads its parameter shard and the model adapter
streams a per-tensor all-gather across TP/EP/PP into the colocated vLLM worker.
When the actor's optimizer moment tensors and gradient buffers are still
resident, that export peak stacks on top of training-only memory and can OOM the
colocated ranks (observed on the 128-GPU PP4/EP8/CP4 run: GPU0 OOM at the first
`update_weights` all-gather).

## Protocol

`MegatronLiteEngine.get_per_tensor_param` now wraps the export with an
evict → export → restore protocol:

1. **Evict** (eager, before the gather): offload optimizer moment tensors to CPU
   and release DDP gradient-buffer GPU storage, recording their entry state.
2. **Export**: reload the parameter shard and stream the per-tensor export. The
   generator body runs lazily in the consumer (vLLM `update_weights`), so the
   all-gather peak now runs without optimizer/grad resident.
3. **Restore** (in the generator's `finally`, after the stream drains): re-load
   gradient buffers and optimizer state to whatever device they were on at
   entry, keeping VERL's offload bookkeeping consistent.

The eviction is transparent: if VERL already offloaded optimizer/grad for the
rollout phase, `optimizer_states_on_gpu` / `model_grads_resident` report false
and the protocol is a no-op (pure instrumentation cost).

Primitives live in `megatron/lite/runtime/megatron_utils.py`
(`optimizer_states_on_gpu`, `model_grads_resident`, `free_grad_buffers`,
`cuda_mem_snapshot`).

## Memory curve evidence

Every resync emits a per-rank line to stdout:

```
MLITE_RESYNC_MEMCURVE rank=<r> opt_offloaded=<bool> grad_freed=<bool> \
  resync/enter=<GiB> resync/after_optimizer_offload=<GiB> \
  resync/after_grad_free=<GiB> resync/export_begin=<GiB> \
  resync/export_end=<GiB> resync/restore=<GiB> export_peak_max_alloc_gib=<GiB>
```

Set `MLITE_RESYNC_MEMLOG_PATH=/path/to/curve.jsonl` to also append the full
snapshot list (with `max_allocated_gib`) as one JSON record per resync per rank.
`export_peak_max_alloc_gib` is the CUDA peak measured over the export window
(peak stats are reset at `export_begin`).

## Fail-fast smoke resync

Set `MLITE_RESYNC_SMOKE_EXIT_AFTER=<n>` so the actor exits(0) once the colocated
resync peak has survived `n` times (barrier-synchronized across ranks, emitting
`MLITE_RESYNC_SMOKE_COMPLETE`). This lets a proxy report a verdict at the first
resync instead of running a full training step. Off by default; an export
failure (OOM) still propagates as a hard failure and is never masked.

## 8-GPU colocated proxy

The production geometry (TP1/PP4/CP4/EP8 = 128 ranks) cannot shrink to 8 GPUs,
but the memory protocol operates on the local model chunks and optimizer, so it
is geometry-independent. A single-node colocated proxy that exercises a real
actor + colocated vLLM resync (unlike the `VLLM_LOAD_ONLY` gate, which builds no
actor and transfers no real tensors) validates the protocol and captures the
curve:

```bash
sbatch --partition=batch --nodes=1 --gres=gpu:8 --time=00:40:00 \
  --export=ALL,ACTOR_TP=1,ACTOR_PP=1,ACTOR_CP=1,ACTOR_EP=8,ROLLOUT_TP=8,\
PHASE1_STEPS=1,TOTAL_STEPS=2 \
  experimental/lite/examples/verl/slurm/run_ds4_gsm8k_grpo.sbatch
```

with the fail-closed path contracts set for the run (`BASE_IMAGE`, `MLITE_SRC`
at this commit, `MLITE_COMMIT`, `MEGATRON_ROOT`, `VERL_ROOT`, `MLITE_SM90_SITE`,
`DS4_VLLM_SITE`, `DS4_VLLM_SHIM`, `CHECKPOINT_DIR`, fresh `RUN_ROOT`), plus
`MLITE_RESYNC_SMOKE_EXIT_AFTER=1` and `MLITE_RESYNC_MEMLOG_PATH` inside the
container env. Watch for `MLITE_RESYNC_MEMCURVE` on the first resync: a bounded
`export_peak_max_alloc_gib` with `opt_offloaded`/`grad_freed` reflecting what was
resident is the green signal. Only after the proxy is green does the 128-GPU
PP4/EP8/CP4 run go, keeping the smoke-exit disabled for the production run.

# M-FSDP DAPO E2E resync OOM — port plan (TASK-1.13.8)

Status: **prepared, gated on DS4 (TASK-1.1.12) validating the resync-memory recipe.**
Do NOT burn GPU on this until DS4 confirms which branch below actually holds.

## Where we are

- CP4 dense-gather deadlock root cause is **solved** — fix `c17a05eff`
  (`0ed992990` buffer.py overlap-gate) verified decisively on real 32-card /
  4-node E2E (job 13838501): mfsdp param-gather recompute ran to completion at
  92–97% util, no deadlock, empty FR dump. Milestone accounted.
- The 32-card run then died at a **different** site: vLLM `wake_up` CUDA OOM
  (`cumem_allocator.cpp:139`) during actor→rollout weight resync
  (`ray_trainer.py:1672 update_weights`). This is colocated memory contention,
  same family as DS4 128-card resync OOM (`[[ds4-128card-resync-oom]]`), NOT the
  CP4 deadlock.

## The mfsdp materialization peak (concrete site)

Export path for resync:

- `mlite_engine.get_per_tensor_param` (engine/mlite_engine.py:370)
  → `runtime.export_weights` (backends/mlite/runtime.py:348)
  → enters `chunk.full_parameter_context()` for **every** model chunk inside one
    `ExitStack`, up front, holding them all for the whole generator lifetime.
- `full_parameter_context` (primitive/optimizers/mfsdp/wrapper.py:191-199) calls
  `param_sync.materialize_all()` → all-gathers the full dense params
  (≈34.6B BF16 ≈ **69 GiB/rank**) and keeps them resident until `release_all()`
  in the generator's `finally`.

So during resync the actor holds ~69–77 GiB of materialized full params resident
while colocated vLLM tries to `wake_up` (peak >80 GiB) → OOM. `materialize_all`
was already flagged OOM-prone in `[[mfsdp-cp4-deadlock-rootcause]]`.

Note: mfsdp DAPO config currently ships offload OFF
(`config/engine/mlite.yaml`: param_offload/optimizer_offload/grad_offload=false),
so the actor is fully resident even before export materialization.

## DS4 dependency — what NOT to port

- DS4's implemented free-grad evict/restore (`31cfecbdf`) was **proven NULL** on
  DS4 config (job 13840020 FAILED, first resync still OOM by ~128 MiB; MEMCURVE
  showed grads not GPU-resident → free-grad lever empirically no-op). Do NOT port
  the free-grad protocol.
- bayan's live candidates for DS4 (deciding tonight, 2026-07-12): (a)
  `expandable_segments`, (b) per-tensor `empty_cache` during export. Port only the
  branch DS4 validates green.

## Port plan — two branches

### Branch A — `expandable_segments` suffices (zero mfsdp code)

If DS4 shows `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` alone lets the
transient export/wake peak fit (it attacks fragmentation, not absolute residency):

- The "port" is a single launch-env line in the cw DAPO harness
  (`qwen35_dapo_mfsdp_*`), matching the `CUDA_DEVICE_MAX_CONNECTIONS="2"` env
  pattern. No repo primitive change.
- Cheapest, most likely-first-to-try. Verify on the 32-card收口 run itself.

### Branch B — per-tensor `empty_cache` needed (mfsdp primitive change)

If Branch A is insufficient, the mfsdp analogue of DS4's "逐tensor empty_cache"
is to stop materializing **all** params at once:

- Restructure the export so each chunk/bucket is materialized, yielded to the
  vLLM consumer, released, and `empty_cache`'d **before** the next — instead of
  `materialize_all()` up front in `full_parameter_context`.
- Touch points: `param_sync.materialize_all` / a new streaming
  `materialize_iter` in mfsdp `param_sync`, consumed by
  `runtime.export_weights` and `full_parameter_context`.
- Correctness constraint to check first: `export_hf_weights` for qwen3_5
  (model/qwen3_5/lite/protocol.py:306) must not require all chunks materialized
  simultaneously (e.g. cross-chunk fusion for vLLM TP sharding). If it does,
  per-chunk release breaks it — gate any change behind an opt-in env
  (default-off, like DS4's `MLITE_RESYNC_*`) and **GPU-verify before enabling**.
- This is a correctness-sensitive primitive edit → follow MLite skills
  (primitive/perf), keep invariants, and it CANNOT ship without a real GPU run.

#### Branch B correctness pre-check — DONE (static, 2026-07-12, no GPU)

Traced the real export generator
`primitive/ckpt/hf_weights.py::export_hf_weights` (protocol.py:306 →
checkpoint.py:752 → this). DAPO mfsdp config is **PP1 + MoE (qwen3_5,
`num_experts`, ran EP8)**, so the PP≤1 branch (hf_weights.py:419-466) is the live
path. Two distinct retention behaviours:

- **Dense params** (attn/router/norms): iterated per chunk via
  `base_chunk.named_parameters()`, gathered one-at-a-time by `_gather_dense`
  (`allgather_concat` → a *new* `torch.cat` tensor, independent of the mfsdp
  buffer) and **yielded immediately**. No cross-chunk retention → per-chunk
  materialize/iterate/release/`empty_cache` is SAFE for dense weights.
- **Expert params** (MoE, active here): when `limit is None` they are
  **accumulated across ALL chunks** into `expert_groups` (hf_weights.py:421-440)
  and gathered only at the end (458-465). Each accumulated entry is
  `_materialize_dtensor(param.data.detach())`. For **mfsdp** params
  `_materialize_dtensor` (hf_weights.py:32-46) is a **pass-through** — it only
  `full_tensor()`s a *DTensor* (FSDP2); mfsdp's `materialize_all()` populates
  `param.data` in place, so the stored tensor is a **view into the mfsdp
  materialized buffer**, NOT an independent copy. `full_parameter_context`'s
  `finally` (wrapper.py:198) calls `release_all()` +
  `discard_full_parameter_views()`, which frees that buffer.

**Conclusion:** naive per-chunk release is UNSAFE for the expert path — releasing
chunk *i*'s mfsdp buffer before the final `expert_groups` gather would invalidate
the retained `param.data.detach()` views (use-after-free / wrong data). This IS
the "cross-chunk fusion" hazard the plan warned about, and it is real for MoE.

Viable Branch-B shapes (whichever DS4 evidence justifies):
1. **Clone-on-extract**: `.clone()` (or `_gather_expert` immediately) each expert
   tensor at accumulation so it survives its chunk's release. Adds a transient
   expert-sized copy — smaller than holding the full mfsdp all-gather buffer, but
   verify it actually lowers the peak vs. just holding one chunk.
2. **Incremental per-group expert gather**: gather+yield each expert group as soon
   as it is complete instead of deferring all groups to the end.
   Dense weights get per-chunk release for free in either shape.

Architectural note: current `runtime.export_weights` (runtime.py:354-361) opens
**every** chunk's `full_parameter_context` up front in one `ExitStack`, holding
all chunks resident for the whole generator. Streaming requires moving context
management **into** the export generator's per-chunk loop — a refactor crossing
the runtime → primitive → model boundary. Keep it behind a default-off env and
GPU-verify the peak actually drops before enabling.

This pre-check strengthens the "try Branch A (`expandable_segments`, zero code)
first" ordering: Branch B is a genuine multi-file primitive change with a live
MoE use-after-free pitfall, not a mechanical edit.

## Execution sequence (once DS4 green)

1. Read DS4's validated recipe + evidence; pick Branch A or B.
2. Apply the minimal port (A: harness env; B: opt-in streaming export + GPU verify).
3. Re-run the 32-card DAPO E2E收口 (fix `c17a05eff` + FR + watchdog, budget ≤16
   GPU-h) for curves/throughput vs existing DAPO. Register via
   `vicky work execute remote-job` to avoid auto-submit/reap.

# M-FSDP DAPO E2E resync OOM — port plan (TASK-1.13.8)

Status: **DS4 gate LIFTED (bayan 2026-07-13 00:30) — Branch B implemented in the
verl layer, CPU/unit green; ready for the 32-card E2E收口 run.**

bayan 00:30 解禁: the "wait for DS4" gate is void — our OOM is mfsdp's own
(actor won't yield to vLLM wake), fixable entirely in our own code. Branch B is
implemented (see §"Implemented (2026-07-13)"): threshold-batched `empty_cache`
draining the export so the released all-gather buffer is returned to the driver
**before vLLM wakes**, plus the `expandable_segments` launch env. The deeper
per-bucket streaming refactor (the MoE use-after-free hazard below) stays
DEFERRED — the observed OOM is at wake_up *after* export drains, which the
empty_cache-before-wake fix targets directly; the per-bucket peak reduction only
matters if the 32-card residency probe shows the transient export peak (not the
handoff) is the OOM.

Historical (pre-00:30) gating context retained below for provenance.

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

## Implemented (2026-07-13) — Branch B, verl layer

Fix landed in the **verl integration layer** (not the megatron_lite export
primitive) — the empty-cache-before-vLLM-wake concern is RL-colocation-specific,
so putting it in `runtime.export_weights` would be a layering violation. Files:

- `examples/verl/verl_mlite/resync_export.py` (new, stdlib-only so it is CPU
  unit-testable without a verl/CUDA runtime):
  - `stream_export_with_empty_cache(gen, threshold_bytes, empty_cache_fn)` —
    drains the export generator, firing `empty_cache_fn` once per
    `threshold_bytes` of cumulative exported material **and once more after the
    generator drains** (its `finally`, i.e. the runtime `ExitStack` →
    `release_all()`, has by then freed the M-FSDP all-gather buffer). That final
    flush returns the freed segments to the driver so the colocated vLLM cumem
    allocator can reclaim them on `wake_up` — closing the observed OOM.
  - `resync_export_empty_cache_threshold_bytes()` — ≥4 GiB default (DS4 recipe),
    env `MLITE_RESYNC_EXPORT_EMPTY_CACHE_GIB` (`0` disables).
- `examples/verl/verl_mlite/engine/mlite_engine.py::get_per_tensor_param` wraps
  the returned export generator with the above, `empty_cache_fn =
  aggressive_empty_cache(force_sync=True)`.
- `tests/unit/verl/test_mlite_engine_resync_export.py` — 10 CPU tests: export
  transparency (correctness), flush-per-threshold + final-drain flush counts,
  disabled pass-through, early-abort still flushes, env parsing. `10 passed`.

**Why empty_cache is the fix, not per-bucket streaming:** the 32-card run
completed the export (transient peak sharded≈8.6 + full≈69 ≈ 77.6 GiB fit) and
OOMed at vLLM `wake_up` *afterward*. `release_all()` returns memory only to
torch's caching allocator; vLLM's separate cumem allocator cannot see it, so
without an `empty_cache` between export-drain and wake the 69 GiB stays pinned.
`get_per_tensor_param` only ran `aggressive_empty_cache` once (guarded, at the
*start* of the first sync) — never after export. This wrapper fills that gap.

`expandable_segments`: a launch-env line
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) applied on the cw harness at
the收口 run (no repo patch — the harness lives cw-side, per Branch A). Defense in
depth for fragmentation.

## The mfsdp materialization peak (concrete site)

Export path for resync:

- `mlite_engine.get_per_tensor_param` (engine/mlite_engine.py:370)
  → `runtime.export_weights` (runtime/backends/mlite/runtime.py:348)
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

### DS4 recipe as SETTLED by bayan 2026-07-12 20:48 (supersedes earlier candidates)

Per-tensor `empty_cache` was **rejected** — each call carries a device sync +
allocator compaction, and thousands of params would blow up resync wall time.
The DS4 relaunch recipe bayan approved is:

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` = the PRIMARY lever**
   (it is the proper fix for fragmentation / materialization peak; pure env, not a
   config knob). Run this + instrumentation FIRST, with no per-tensor cleanup, as
   the baseline.
2. **Threshold-batched `empty_cache` only** — call once per **≥4 GiB of
   accumulated dropped references**; small tensors never trigger it. (Not
   per-tensor.)
3. **`resync` wall time is an acceptance gate**: instrument resync wall time;
   >20% overhead vs. the no-(2) baseline = **fail**. Memory AND speed both must
   hold — no robbing Peter to pay Paul.

Port only what DS4 validates green, and carry the same wall-time gate.

### DS4 20:54 correction (supersedes 20:48 on two points)

bayan scancelled DS4 job 13870979 (止损) and corrected the recipe again at
2026-07-12 20:54. Two deltas that reshape this plan:

1. **Per-tensor `empty_cache` is hard-OFF** — reconfirmed (thousands of tensors
   would 等死). `expandable_segments` + instrumentation stay. Our Branch B is
   already threshold-batched (≥4 GiB), not per-tensor, so this only *strengthens*
   the existing ordering — do NOT resurrect per-tensor cleanup.
2. **NEW pre-ignition memory-budget gate** — before burning ANY card, itemize the
   per-card residency and prove `< 80 GiB with headroom`. bayan's algorithm:
   - rollout side: vLLM per-card weight (e.g. FP8+TP16 ⇒ ≈300GB/16 ≈ 19 GiB);
   - training side: actor weight after parallel sharding, **+ optim state proven
     *empirically* CPU-offloaded (not by reading a flag — verify it is not GPU
     resident; last time an `opt_offloaded=False` field's semantics were unclear),
     + activations + NCCL buffers, each itemized;
   - sum `< 80 GiB` with headroom = the only green-light to ignite; log the table.

### DS4 21:05 diagnostic — sibling-rank NCCL/CUDA buffer residency (HYPOTHESIS, not yet confirmed)

While assembling the 20:54 budget table, DS4 (job r2) observed **7 sibling ranks
each holding ≈2.51 GiB resident (≈17.5 GiB/card of "stray" occupancy)**. bayan's
2026-07-12 21:05 targeted diagnosis (top priority on DS4): this is **almost
certainly an NCCL-buffer / CUDA-context problem, prime suspect a
`CUDA_VISIBLE_DEVICES` / Ray `num_gpus` misconfig** — if each actor process sees
all 8 local GPUs instead of only its own, NCCL/CUDA builds a context + P2P buffers
on *every* visible device, so exactly 7 siblings each pin a buffer. If confirmed
and fixed, that reclaims ~17.5 GiB/card and **flips the budget table outright**.

**Why this is in OUR plan (transfer, not copy):** our 32-card收口 run uses the
**same colocated verl/Ray actor↔vLLM architecture** (`ray_trainer.py:1672
update_weights`, vLLM `wake_up` — see §"Where we are"), so this visibility hazard
is a generic property of that colocation, not a DS4-only path. It therefore
belongs in OUR pre-ignition gate as a **watch item**: before believing any
per-card residency row, first verify each actor's `CUDA_VISIBLE_DEVICES` /
`torch.cuda.device_count()` is 1 (its own card), and check whether the colocated
vLLM↔actor CUDA-IPC weight sync *intentionally* needs cross-process visibility
(if shared-on-purpose, the lever is `NCCL_P2P` / cumem, not visibility). Do NOT
port a "fix" yet — it is an unconfirmed DS4 hypothesis; treat it as the first row
to audit when the budget table is assembled cw-side at ignition-prep time.

Status note (2026-07-12): DS4 (TASK-1.1.12) is **still In Progress, NOT green** —
20:48 and 20:54 were both "must-change-before-ignition" corrections, the last job
(13870979) was scancelled, and 21:05 opened a fresh sibling-residency diagnostic
that may再 reshape the budget analysis. This task stays gated until DS4 validates a
recipe green.

## Port plan — two branches

### Branch A — `expandable_segments` suffices (zero mfsdp code) — LEADING

This is now DS4's **primary** lever (20:48), so it is also the leading — and most
likely sole — mfsdp port. It attacks fragmentation / materialization peak, not
absolute residency.

- The "port" is a single launch-env line
  (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) in the cw DAPO mfsdp
  harness (`qwen35_dapo_mfsdp_*`), matching the `CUDA_DEVICE_MAX_CONNECTIONS="2"`
  env pattern. **These harness scripts live on the cw cluster, NOT in this repo**,
  so there is no repo-side patch to stage for Branch A — it is applied at launch
  time on the收口 run. No mfsdp primitive change.
- Cheapest, most likely-first-to-try. Verify on the 32-card收口 run itself, and
  record resync wall time (the DS4 ≤20% overhead gate applies here too).

### Branch B — threshold-batched `empty_cache` needed (mfsdp primitive change)

Only if Branch A is insufficient. NOTE the DS4 20:48 correction reshapes this:
the analogue is **NOT** per-tensor `empty_cache` (rejected as too slow) but
**threshold-batched** release — stop materializing **all** params at once, and
call `empty_cache` at most once per ≥4 GiB of released material, never per small
tensor. Any Branch-B change must clear the same resync wall-time gate (≤20%
overhead vs. Branch-A baseline).

- Restructure the export so chunks/buckets are materialized, yielded to the
  vLLM consumer, and released as we go — with a single `empty_cache` fired only
  after cumulative released bytes cross the ≥4 GiB threshold — instead of
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

Architectural note: current `runtime.export_weights`
(runtime/backends/mlite/runtime.py:354-361) opens
**every** chunk's `full_parameter_context` up front in one `ExitStack`, holding
all chunks resident for the whole generator. Streaming requires moving context
management **into** the export generator's per-chunk loop — a refactor crossing
the runtime → primitive → model boundary. Keep it behind a default-off env and
GPU-verify the peak actually drops before enabling.

This pre-check strengthens the "try Branch A (`expandable_segments`, zero code)
first" ordering: Branch B is a genuine multi-file primitive change with a live
MoE use-after-free pitfall, not a mechanical edit.

## Execution sequence (once DS4 green)

1. Wait for DS4 (TASK-1.1.12) to go green, then read its validated recipe +
   evidence (incl. measured resync wall-time overhead). Expect Branch A
   (`expandable_segments`) to be the answer since it is DS4's primary lever.
2. **Pre-ignition memory-budget gate (mandatory, per DS4 20:54)** — before
   launching the 32-card收口 run, produce and log the per-card residency table for
   OUR colocated site and prove `< 80 GiB with headroom`:
   - vLLM per-card rollout weight (at the收口 run's actual TP/quant);
   - actor weight after mfsdp sharding at TP1·PP1·CP4·EP8·DP2;
   - **⚠ offload tension**: mfsdp DAPO config currently ships param/optim/grad
     offload **OFF** (`config/engine/mlite.yaml`, see §"materialization peak"). The
     gate demands optim state be *empirically* off-GPU — so either flip offload ON
     for the收口 run and verify residency, or budget the full resident optim state
     and confirm the sum still clears 80 GiB. Do NOT trust the flag; measure.
   - export materialization peak (≈69–77 GiB if `materialize_all` stays) +
     activations + NCCL buffers, itemized. **Audit the NCCL-buffer row FIRST per
     the DS4 21:05 diagnostic** (§"DS4 21:05 diagnostic"): confirm each actor sees
     only its own GPU (`CUDA_VISIBLE_DEVICES` / `device_count()==1`) so stray
     sibling-rank P2P buffers (~2.5 GiB each) aren't silently inflating the table.
   Sum < 80 GiB with headroom is the only green-light. If the table itself shows
   Branch A (`expandable_segments`, a fragmentation fix, not a residency fix)
   cannot close an *absolute*-residency overflow, that is the signal Branch B
   (streaming export to cut the materialization peak) is required — decide from the
   table BEFORE burning the card, not after another OOM.
   - **Data-availability boundary (verified 2026-07-12, repo-only recon):** this
     table CANNOT be faithfully pre-computed from this worktree. The repo default
     `engine/mlite.yaml` ships `tp/pp/cp/ep=1` and offload OFF; the收口 run's real
     parallel sizing AND the vLLM rollout TP/quant are set by the **cw harness at
     launch time** (not in this repo, per Branch A), and bayan's gate requires the
     optim-state residency be measured *empirically on the target config*, not read
     from a flag. So two of the three rows (rollout weight, empirical optim
     residency) need cw-side + GPU data that is unavailable pre-ignition here.
     Assemble the table at ignition-prep time on the cw side — do NOT fabricate
     placeholder numbers to "pass" the gate. The only rows knowable now are the
     static ones already in this doc (export materialization peak ≈69–77 GiB;
     ≈34.6B BF16 dense actor weight before sharding).
3. Apply the minimal port:
   - Branch A: add the env line to the cw DAPO mfsdp harness at launch time
     (no repo patch). This is the default plan.
   - Branch B (only if the budget table or A's result proves it needed): opt-in,
     default-off, threshold-batched streaming export + GPU verify the peak drops
     AND wall-time overhead ≤20%. Do NOT pre-write this speculatively — its shape
     depends on DS4's threshold/wall-time evidence and it carries the live MoE
     use-after-free hazard documented above.
4. Re-run the 32-card DAPO E2E收口 (fix `c17a05eff` + FR + watchdog, budget ≤16
   GPU-h) for curves/throughput vs existing DAPO. Register via
   `vicky work execute remote-job` to avoid auto-submit/reap.

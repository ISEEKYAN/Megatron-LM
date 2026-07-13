# DS4 128-GPU resync OOM — GPU0 sibling-residency root cause + memory budget

> **⚠️ CORRECTION (2026-07-12, jobs `13874307`/`13874308`): the §2 root-cause
> hypothesis below is REFUTED by direct measurement.** An 8-GPU load-only A/B
> experiment shows the vLLM TP8 rollout, on its own, leaves **exactly one process
> per GPU and zero sibling residency on GPU 0** — both at baseline and with
> `NCCL_P2P_DISABLE=1`. If the siblings were a static vLLM-TP-init /
> shared-visibility CUDA context (§2), the baseline would already show 7 of them.
> It does not. The 7×2.51 GiB siblings are therefore **triggered by the real
> colocated resync weight-gather traffic**, which load-only does not exercise.
> Read §7 first; treat §2/§4/§5 as the (now-superseded) initial hypothesis.

Scope: TASK-1.1.12. Directed diagnosis of the 7×2.51 GiB sibling residency on GPU 0
that OOMs the first colocated weight resync of the DS4 128-GPU run.
Evidence: cw job `13840020` (`smoke128-b67cd7684-r2`, ROLLOUT_TP8 / util 0.60 /
SMOKE_EXIT_AFTER=1), OOM at `vllm_rollout.py update_weights`.

## 1. The OOM, decoded

```
CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a total capacity of
79.11 GiB of which 125.44 MiB is free.
  Process 3536516..3536522  has 2.51 GiB memory in use   (x7)
  Process 3536515           has 20.12 GiB memory in use   (this process, incl. non-PyTorch)
  Process 3559059           has 41.25 GiB memory in use
```

Per-GPU accounting on a colocated node (each of the 8 GPUs carries the same shape;
GPU 0 simply OOMs first because rank-0 exports first):

| Component | Process(es) | GiB |
|---|---|---|
| FSDP2 actor (Megatron train, PP4·EP8·CP4·TP1) at resync export peak | 3559059 | 41.25 |
| vLLM **own** TP rank (weights@util0.60 + KV + CUDA context) | 3536515 | 20.12 |
| **7× sibling vLLM TP-rank CUDA context on GPU 0 — LEAK** | 3536516–3536522 | **17.57** (7×2.51) |
| **Total** | | **78.94** |
| Free | | 0.125 (128 MiB) → OOM on the next 128 MiB alloc |

The 7 siblings are the vLLM tensor-parallel workers whose **home** device is GPU 1–7,
each holding a ~2.51 GiB CUDA context / P2P residency **on GPU 0**. Pure waste.

Note: the OOM lists **seven** vLLM siblings and **zero** actor siblings. That asymmetry
is the tell — see §2.

## 2. Root cause (confirmed in code)

vLLM V1 `MultiprocExecutor` (`VLLM_USE_V1=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`)
spawns the 8 TP-worker subprocesses from one parent (`EngineCore`). The workers
**inherit the parent's `CUDA_VISIBLE_DEVICES` = all 8 node GPUs** and only call
`current_platform.set_device(local_rank)`
(`vllm/v1/executor/multiproc_executor.py:938`). There is **no per-worker
`CUDA_VISIBLE_DEVICES` isolation** in the multiproc path (contrast the Ray executor
path, `ray_executor.py:309`, which comments it deliberately exposes all node GPUs too).

Because every vLLM worker can *see* all 8 GPUs, the TP-group NCCL init
(`NCCL_NVLS_ENABLE=0`, `disable_custom_all_reduce=True`, so this is **not** vLLM
custom-all-reduce IPC) establishes a CUDA context + P2P residency on the peer
devices — each of the 7 non-local ranks lands ~2.51 GiB on GPU 0.

The FSDP actors do **not** leak: they are separate Ray actors with `num_gpus=1`, so Ray
isolates each actor's `CUDA_VISIBLE_DEVICES` to a single physical GPU. That is exactly
why the OOM shows 7 vLLM siblings and no actor siblings — the discriminating evidence
that pins the leak on the vLLM multiproc path specifically.

### Gate blind spot
The preflight `DeviceProbe` (`run_ds4_gsm8k_grpo.sbatch:645`) asserts
`physical_id in visible_ids` but **not** `len(visible_ids) == 1`, and records
`device_count` without asserting on it. It is therefore structurally blind to
full-node visibility. Tightening it to assert single-device visibility per rank would
have caught this before burning 128 GPUs.

## 3. Memory budget if the leak is reclaimed

| Component | current (GiB) | fixed (GiB) |
|---|---|---|
| FSDP2 actor @ resync peak | 41.25 | 41.25 |
| vLLM own rank | 20.12 | 20.12 |
| 7× sibling leak | 17.57 | 0 |
| **Total** | **78.94** | **61.37** |
| **Headroom (79.11 cap)** | **−0.13 (OOM)** | **+17.7** |

Reclaiming the siblings flips the budget from −0.13 GiB (OOM by 128 MiB) to +17.7 GiB.
This is the single dominant lever; the previously-hypothesised free-grad / optim-evict
levers were empirically NULL for this config (grads/optim already CPU-resident;
`memory_allocated()==0` at resync entry — see [ds4-resync-memory-protocol.md]).

## 4. Fix options

**Option A — per-vLLM-worker visibility isolation (cleanest, but touches the
colocated contract).** Set each MultiprocExecutor worker's `CUDA_VISIBLE_DEVICES` to
its single physical device before CUDA init, so it physically cannot allocate on GPU 0.
Risk: the colocated resync IPC handoff relies on physical-id visibility
(`verl_mlite/compat.py:_normalize_vllm_visible_device_id`, "leaked physical CUDA id").
Isolating to one device remaps the visible index (device→0) and could break the
actor↔vLLM device-UUID matching used during weight sync. Requires code + careful
validation of the handoff.

**Option B — NCCL P2P suppression (zero-code env, test first).**
`NCCL_P2P_DISABLE=1` (and/or `NCCL_CUMEM_ENABLE=0`). If the 2.51 GiB is NCCL P2P
peer-access context/buffers, disabling P2P prevents the sibling residency on GPU 0.
Cost: intra-node TP8 all-reduce falls back to SHM staging → continuous generation
throughput hit. Acceptable only if the throughput cost is tolerable.

**Open question that decides A vs B:** whether the 2.51 GiB is NCCL-P2P-tunable or an
irreducible per-process CUDA context. That is an empirical question — resolve it before
any 128-GPU spend (§5).

## 5. Cheap verification (structurally viable — unlike the FSDP 8-GPU proxy)

The FSDP actor cannot fit in 8 GPUs (dies at init — established, not re-litigated).
**But the vLLM-side sibling leak is reproducible with `VLLM_LOAD_ONLY=1` on 1 node /
8 GPUs** (vLLM TP8, no actor — this path already passes, job `13826680`). Add an
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` dump right after vLLM
engine init and read GPU-0 per-process residency. A/B/C:

| Arm | Expectation if leak = NCCL P2P | Expectation if leak = CUDA context |
|---|---|---|
| Baseline (current) | 7 siblings × ~2.51 GiB on GPU 0 | same |
| `NCCL_P2P_DISABLE=1` | siblings → ~0 | siblings persist (~context) |
| per-worker visibility isolation | siblings → 0 | siblings → 0 |

Cost ~0.1–0.2 GPU-h, faithful to the exact vLLM mechanism, and decisively selects
Option A vs B before any 128-GPU relaunch.

## 6. Recommendation

1. Run the §5 8-GPU load-only A/B/C (pre-authorised by bayan 2026-07-12 21:05; needs the
   pre-GPU moe gate).
2. If `NCCL_P2P_DISABLE=1` reclaims the siblings and generation throughput is acceptable
   → ship Option B (env-only, no colocated-contract risk).
3. Else → implement Option A with explicit validation that the actor↔vLLM UUID handoff
   still resolves, plus tighten the `DeviceProbe` to assert single-device visibility.

## 7. Empirical result — the §5 experiment REFUTES §2 (2026-07-12)

Ran the §5 8-GPU load-only diagnostic at HEAD `0b00a4028` (`MLITE_VLLM_RESIDENCY_PROBE=1`,
`VLLM_LOAD_ONLY=1`, 1 node × 8 GPU, ROLLOUT_TP8, util 0.60), two arms:

| Arm | Job | Result: per-GPU processes | GPU-0 siblings |
|---|---|---|---|
| A — baseline (full-node visibility) | `13874307` | **1 process per GPU** (own TP worker, ~50226 MiB) | **0** |
| B — `NCCL_P2P_DISABLE=1` | `13874308` | **1 process per GPU** (~49352 MiB) | **0** |

Both arms `DS4_VLLM_LOAD_ONLY_PASSED`. Confirmed by two independent methods (the
`post_init` residency probe **and** a live `nvidia-smi --overlap` snapshot taken both
mid-init and after the KV-cache profiling forward pass). Every GPU carries only its own
rank; nothing lands on a peer GPU.

### What this means

- **§2 is refuted.** The 7×2.51 GiB siblings are **not** a static artifact of vLLM
  TP-init under shared `CUDA_VISIBLE_DEVICES`. If they were, arm A (identical vLLM TP8,
  full visibility) would already show 7 siblings on GPU 0. It shows zero. The
  profiling forward pass runs a real TP all-reduce and still leaves no peer residency.
- **The leak is resync-traffic-triggered.** The only thing the real 128-GPU run does
  that load-only does not is the actual colocated weight resync — `vllm_rollout.py
  update_weights(..., base_sync_done=True) → torch.distributed.all_gather` over **real**
  weight tensors across the TP group (plus the colocated actor's CUDA-IPC handoff).
  The load-only `VLLM_CHECKPOINT_SYNC_PROBE` sends only empty buckets (`tensors=0`), so
  it never gathers real weights and never opens the peer buffers.
- **Load-only is structurally blind to this leak** — the same blind spot flagged before
  the first 128-GPU burn (`VLLM_LOAD_ONLY` marker `tensors=0`). The A/B/C harness is
  cheap and faithful to vLLM *init*, but the failure is at *resync*.
- **Arm C (per-worker visibility isolation) was not built.** It targets the refuted
  init-visibility mechanism; there is nothing for it to isolate in load-only. There is
  also no built-in per-TP-worker isolation switch in vLLM 0.20.2 — `CUDA_VISIBLE_DEVICES`
  is set per **DP** rank only (`v1/engine/core.py:1934`, `v1/engine/utils.py:261`); TP
  workers deliberately share full-node visibility so the TP NCCL group can form.
- Minor corroborating detail: arm B used ~874 MiB **less** per GPU than arm A. NCCL P2P
  buffers do exist, but in the load-only forward path they are allocated on the rank's
  **own** GPU, not as cross-GPU residency on GPU 0.

### Next faithful step (needs a decision — see task escalation)

To observe *where* the siblings form we need a path that runs the real resync
all_gather over real weights. Candidates, in rough cost order:

1. **Instrument the real 128-GPU run**: snapshot GPU-0 per-process residency
   immediately before and after the first resync all_gather (SMOKE_EXIT_AFTER=1), to
   confirm the buffers appear *at* the gather and measure their exact size — and, in the
   same run, A/B `NCCL_P2P_DISABLE=1` on the actual failing path.
2. **A resync-exercising colocated proxy** that actually gathers real weights across a
   TP group (the FSDP-actor 8-GPU proxy does not fit — established — but a reduced
   colocated config that reaches a real resync might).
3. If the buffers are confirmed to be NCCL all_gather P2P buffers, the levers are
   `NCCL_P2P_DISABLE` / `NCCL_BUFFSIZE` / bucketing the gather — none of which is the
   visibility-isolation of the old Option A.

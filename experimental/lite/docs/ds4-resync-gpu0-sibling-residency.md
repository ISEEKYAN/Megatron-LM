# DS4 128-GPU resync OOM — GPU0 sibling-residency root cause + memory budget

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

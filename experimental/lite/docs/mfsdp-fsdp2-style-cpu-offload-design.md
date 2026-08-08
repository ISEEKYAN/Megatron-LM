# M-FSDP CPU Offload by Reusing the FSDP2 FP32 AdamW Kernel

Status: research recommendation; no implementation or production-readiness claim.

## Decision

Route A is technically viable, but it is not ready to ship. The recommended
implementation is to extract the existing FSDP2 `FP32AdamW` update kernel into
a backend-neutral optimizer module and let both FSDP2 and M-FSDP call it.
M-FSDP must retain a thin adapter that owns gradient selection, communication
completion, full-parameter-view release, runtime residency, and compatibility
with existing M-FSDP checkpoints.

This is reuse of one optimizer kernel, not a third optimizer and not a direct
reuse of the FSDP2 wrapper. In particular:

- do not insert MCore `HybridDeviceOptimizer` into M-FSDP;
- do not make M-FSDP import a sibling FSDP2 backend module;
- do not copy `FP32AdamW` into the M-FSDP directory;
- do not move M-FSDP process-group, clipping, or bucket knowledge into the
  shared update kernel.

Implementation should be authorized only as a follow-up after this design is
accepted. Production readiness then requires the four-arm 8-GPU experiment
defined below. Existing results are useful baselines but do not satisfy that
gate.

## Audited references

The source was inspected without modifying either implementation:

- NVIDIA/Megatron-LM `nv/main@ccf686edd41c79ff876239d0d31f1ad24fc81116`,
  fetched 2026-08-08;
- ISEEKYAN/Megatron-LM `is/main@9d2686765fdd37bd498d01a71061db38382065b2`;
- the measured M-FSDP performance tree
  `rebase/pr148-on-pr89@c613d9d169cb4e0191049c951b06cd6b9b6a1db5`;
- the latest local M-FSDP memory-lifecycle tree
  `work/mfsdp-pr148-memory@223a10ef798b12b4778c0d0998ed265fb2f893d9`.

Line references below refer to those immutable revisions. The FSDP2 CPU-update
baseline used in the existing 8-GPU comparison is the implementation at
`is/main`; the M-FSDP lifecycle references use `work/mfsdp-pr148-memory`.

## File-by-file comparison

### Reusable FSDP2 surface

`experimental/lite/megatron/lite/primitive/optimizers/fsdp2/adamw.py` at
`is/main` contains the reusable algorithm:

- `FP32AdamW.__init__` (lines 117-161) normalizes parameter groups and creates
  one authoritative FP32 master plus FP32 first and second moments;
- `_init_master_param` (lines 163-173) converts only the local tensor to CPU
  when `cpu_update=True`;
- `_step_param_groups` (lines 192-219) implements AdamW decay, bias correction,
  moment updates, and master update in a deterministic per-parameter loop;
- `_prepare_grad` (lines 221-225) copies one local gradient to CPU on demand;
- `_copy_master_to_param` (lines 227-234) refreshes the local model shard;
- `state_dict` and `load_state_dict` (lines 236-293) own master, moments,
  per-parameter steps, and weight-decay metadata;
- `build_adamw_optimizer` (lines 296-362) selects this kernel when an FP32
  master is requested.

The FSDP2-specific parts must remain outside the shared kernel: DTensor
unwrapping and uneven-shard copy rules at lines 35-73, FSDP2 replicated/expert
gradient synchronization and norm calculation in `fsdp2/optimizer.py`, and
DTensor residency metadata in `fsdp2/state.py` lines 15-99.

### M-FSDP surface to replace

`experimental/lite/megatron/lite/primitive/optimizers/mfsdp/cpu_offload.py` at
`work/mfsdp-pr148-memory` currently implements the same AdamW algorithm through
`CpuAdamGroup`:

- lines 85-109 create a pinned FP32 CPU master for every offloaded local shard;
- lines 111-125 create one `torch.optim.AdamW(foreach=False)` per parameter;
- lines 134-184 copy all gradients into persistent per-parameter pinned buffers,
  run the CPU optimizers, and refresh GPU shard mirrors;
- lines 186-216 serialize the torch optimizers and a second explicit list of
  master parameters, including compatibility with the former aggregate AdamW
  format at lines 47-63.

The persistent `_cpu_grad_bufs` list at lines 89, 104, and 150-157 is the
specific mechanism to remove. Its size is proportional to every offloaded
local element even though a scalar AdamW update only needs the gradient slice
currently being updated.

### M-FSDP lifecycle that must not move

`mfsdp/optimizer.py` must continue to own all backend semantics:

- `_StandaloneOptimizer.step` lines 124-133 performs TP synchronization,
  expert scaling, global clipping, GPU-subset update, then CPU-subset update;
- gradient norm and scaling read `main_grad` before `.grad` at lines 135-189
  and 246-270;
- the existing checkpoint envelope is selected at lines 191-225;
- cross-stage optimizer residency stays at lines 227-244;
- `MFSdpOptimizer.step` lines 341-347 releases communication buffers and stale
  full-parameter views only after the update completes;
- construction, fail-loud algorithm checks, and stable fraction splitting stay
  at lines 362-474 and 571-620.

`mfsdp/buffer.py` on the same revision owns the parameter/gradient storage:

- full offload keeps a BF16 local shard mirror while the authoritative master
  is on CPU (lines 311-345);
- bounded FP32 fused-wgrad views are attached at lines 422-450;
- model offload first drains full-param and full-grad scratch and preserves
  shard aliases (lines 541-591);
- colocated wake releases scratch but retains shard weights (lines 593-612);
- the non-TE fallback accumulates `.grad` into FP32 `main_grad` before clearing
  `.grad` (lines 769-789).

`mfsdp/wrapper.py` lines 179-205 and 207-244 own model residency and bounded
bucket streaming for export. None of these concepts belongs in the shared
AdamW kernel.

## The thinnest adapter

The implementation should have three layers:

1. **Shared FP32 AdamW kernel.** Move the current algorithm, parameter-group
   normalization, master/moment state, and step math from `fsdp2/adamw.py` to a
   backend-neutral module under `primitive/optimizers`. Its inputs are ordinary
   local tensors. It must not import FSDP2, M-FSDP, DTensor, process groups, or
   model code.
2. **FSDP2 adapter.** Keep DTensor `to_local` and copy-back callbacks, the
   current FSDP2 state envelope, and all FSDP2 synchronization in the FSDP2
   directory. Behavior is unchanged.
3. **M-FSDP adapter.** Replace `CpuAdamGroup`'s per-parameter torch optimizers
   with the shared kernel. The adapter supplies `main_grad` as the gradient,
   waits for M-FSDP grad reduction before stepping, provides the bounded
   transfer pipeline below, preserves the current M-FSDP checkpoint envelope,
   and returns only after the GPU shard mirrors are current.

The shared kernel needs only these explicit backend hooks:

```python
local_param(param) -> Tensor
local_grad(param) -> Tensor | None
copy_master_slice_to_param(param, master, start, length) -> None
transfer_policy.copy_grad_slice(grad, start, length) -> CPU Tensor
```

The default hooks are plain tensor identity, `param.grad`, plain `copy_`, and a
synchronous `.to("cpu", dtype=torch.float32)`. FSDP2 supplies DTensor hooks;
M-FSDP supplies `getattr(param, "main_grad", param.grad)` and the bounded
pinned pipeline. A hook is warranted only where the two existing backends
already differ. Adding a generic process-group or lifecycle callback would be
over-design and would violate primitive layering.

The M-FSDP adapter remains production-reachable through
`build_mfsdp_stack -> _build_cpu_adam_group -> _StandaloneOptimizer.step`.
After extraction, the old torch-AdamW update loop and `_CpuOptimizerCollection`
must be removed; retaining them for tests or legacy checkpoints would leave a
second, dead algorithm path.

## Bounded gradient-transfer design

The full persistent per-parameter gradient staging set must be deleted. Use a
two-slot pinned ring whose slot capacity is bounded by the existing M-FSDP
communication `bucket_size`. A parameter larger than one slot is updated in
contiguous slices. No new user-visible tuning knob is needed for the first
implementation.

For slot capacity `C` elements and pipeline depth `P=2`:

1. enqueue D2H of slice `i` from the already reduced FP32 `main_grad` into slot
   `i mod P`;
2. wait only for that slot's D2H event;
3. update the matching FP32 master/moment slices using one per-parameter step
   value and the same bias corrections for every slice;
4. enqueue H2D copy/cast of the updated master slice into the BF16 shard mirror;
5. do not reuse a slot until both its consumer and transfer event are complete;
6. synchronize the final H2D event before `MFSdpOptimizer.step` releases views
   or returns.

This preserves exact AdamW semantics because the update is elementwise and the
step counter advances once per parameter, not once per slice. Zero-sized local
shards are skipped without advancing their state.

The hard staging bounds per rank are:

- pinned gradient staging: `4 * min(N_offload, P*C)` bytes;
- concurrently staged gradient elements: at most `min(N_offload, P*C)`;
- concurrently updated parameter: one parameter slice of at most `C` elements;
- extra allocated CUDA staging: zero, because D2H reads existing `main_grad`
  and H2D `copy_` writes directly into the existing shard mirror. CUDA memory
  instrumentation must verify this instead of assuming it.

`C` is the effective positive M-FSDP bucket size, currently defaulting to
40,000,000 elements in `mfsdp/config.py` lines 197-202. If the model's largest
local offloaded shard is smaller, the allocated slot may be smaller. Pipeline
depth is fixed at two because only one producer/consumer overlap is required;
increasing it is a separate performance proposal requiring evidence.

## Byte accounting

Let `N` be padded optimizer-owned local shard elements on one rank, `O` the
offloaded subset, `G=N-O` the GPU-updated subset, `C` the bounded slice capacity,
and `P=2`. BF16 uses 2 bytes and FP32 uses 4 bytes. Scalar step tensors and
Python metadata are listed separately because they are `O(number_of_params)`,
not `O(number_of_elements)`.

### M-FSDP offload=1 steady state after the change

| Device | Storage | Bytes |
|---|---|---:|
| GPU | BF16 local shard mirror | `2N` |
| GPU | FP32 sharded `main_grad` | `4N` |
| GPU | optimizer master/moments | `0` |
| CPU | authoritative FP32 master | `4N` |
| CPU | FP32 Adam first and second moments | `8N` |
| CPU pinned | bounded gradient ring | at most `4*min(N, PC)` |

Thus element-proportional steady storage is `6N` GPU bytes and
`12N + 4*min(N, 2C)` CPU bytes, upper-bounded by `12N + 8C`. The current
implementation is `6N` GPU bytes and approximately `16N` CPU bytes because
`_cpu_grad_bufs` contributes another `4N`. The proposal removes
`4N - 8C` host bytes when `N > 2C` and, more importantly, removes the
per-parameter pinned-allocation count. When `N <= 2C`, only slices that can hold
distinct gradient elements are allocated, so the ring never exceeds the
current `4N` staging footprint.

### Partial and zero offload

For fraction `0 < f < 1`, CPU optimizer storage is
`12O + 4*min(O, 2C)`. Partial offload does not enable the full-offload BF16
mirror optimization: M-FSDP retains an FP32 local parameter buffer (`4N`), an
FP32 main-gradient buffer (`4N`), and two FP32 Adam moments for the GPU subset
(`8G`) on GPU. The GPU parameter itself is authoritative for that subset, so
adding another `4G` master would double count it. For `offload=0`, steady
element storage is therefore `16N` GPU bytes (`4N` parameter, `4N` gradient,
`8N` moments), and no CPU master, CPU moments, pinned ring, transfer streams, or
transfer events may be constructed. This zero-allocation property is a
contract test, not an inferred property.

### Transient phases

The optimizer phase may add at most the bounded host ring; it must add no CUDA
allocation above the existing shard mirror and `main_grad`. Full-step peak also
includes M-FSDP forward/backward all-gather and FP32 fused-wgrad scratch, which
are owned and bounded by `mfsdp/buffer.py`, not by the optimizer adapter. At the
optimizer boundary, all full-parameter and full-gradient leases must already be
released. Formal validation therefore records three different CUDA peaks:

1. steady state after at least two completed optimizer steps;
2. full step, from forward entry through optimizer return;
3. optimizer only, after gradient synchronization and before optimizer entry
   through optimizer return.

Allocated and reserved bytes must both be recorded. A post-build snapshot is
not a steady-state measurement because Adam moments are lazily materialized in
some arms.

## Checkpoint, export, and wake contracts

### Checkpoint and resume

- `state_dict()` first drains D2H/H2D events. It serializes the authoritative
  CPU master, both moments, and one step per parameter; the pinned ring and
  CUDA events are scratch and are never serialized.
- The external M-FSDP envelope remains `{"gpu": ..., "cpu": ...}` plus
  `_mfsdp_param_values`. The adapter may use the shared kernel internally, but
  must keep writing a versioned M-FSDP format and must read the current
  per-parameter torch-AdamW format.
- Loading validates parameter count, order, shape, dtype, group hyperparameters,
  and offload partition. It restores CPU authority, refreshes every GPU shard
  mirror, and only then exposes the optimizer to training.
- A checkpoint saved with `offload=1` must resume with `offload=1` and produce
  the same next update as an uninterrupted run. Changing the fraction at load
  is out of scope unless a separately designed reshard/migration contract is
  added.
- DCP still saves optimizer state per rank as implemented in `primitive/ckpt/dcp.py`
  lines 228-255. The shared kernel does not acquire global resharding ownership.

### Export

Export consumes current GPU shard mirrors through M-FSDP's existing
`stream_full_parameters` bucket stream. The CPU master is not exported directly:
doing so would bypass model dtype conversion, TP/EP placement, and the bounded
gather contract. Optimizer return guarantees all H2D refreshes are visible
before export begins.

### Colocated rollout wake and training reload

- Before CPU offload or rollout wake, all optimizer transfer events are drained.
- M-FSDP model state then releases full-param/full-grad scratch and moves only
  persistent shard storage according to `move_model_state`.
- CPU-authoritative master/moments remain on CPU for `offload=1`; the bounded
  pinned ring is released at the training/rollout boundary and is never treated
  as optimizer state.
- `release_export_scratch` continues to release only M-FSDP gather scratch; it
  must not learn about AdamW internals.
- On training reload, model shard mirrors return to CUDA before the next step.
  CPU optimizer state stays on CPU and the ring is lazily recreated. Partial
  offload continues to use the existing optimizer state residency hooks for the
  GPU subset.

## Correctness validation before GPU performance work

The implementation task should start with failing contract tests and the
smallest implementation that turns them green:

1. shared-kernel CPU parity against `torch.optim.AdamW(foreach=False)` for
   decay/no-decay groups, skipped gradients, per-parameter steps, and sliced
   versus unsliced updates;
2. legacy M-FSDP checkpoint load followed by next-step parity, plus a new-format
   save/load round trip;
3. `offload=0` constructs no CPU/pinned/stream state;
4. bounded staging asserts total slot capacity `<= 2C`, including one parameter
   larger than `C` and zero-sized local shards;
5. M-FSDP composition preserves TP/expert synchronization, global grad clipping,
   and release-after-step ordering;
6. native FP32 accumulation remains genuine: representative TE-fused and
   non-TE gradients must be FP32 and at least one must satisfy
   `torch.equal(g, g.to(torch.bfloat16).to(torch.float32)) == False`;
7. checkpoint continuity follows the existing single-node distributed smoke
   path and verifies the update after resume, not merely successful loading.

These tests are necessary but cannot produce performance or memory conclusions.

## Formal 8xH100 readiness experiments

No 2-GPU, 2-layer, or sequence-128 result may enter the decision. All GPU arms
run through Slurm on one node with exactly 8 H100 GPUs and this fixed workload:

- Qwen3.5 truncated to 8 transformer layers and 8 experts;
- DP=8, TP=EP=CP=PP=1;
- 4 microbatches per optimizer step;
- sequence length 1024;
- AdamW with precision-aware optimizer disabled;
- identical seed, batches, parameter order, dtype policy, checkpoint setting,
  CUDA/NCCL/TE environment, and instrumentation.

Performance and precision are separate jobs. A 15-step performance run cannot
stand in for the required 50-step precision run, even if it records finite
losses. Conversely, the precision run is not used to replace the fixed
warmup/measurement performance protocol.

### Performance and memory: 15 steps

Run exactly 15 optimizer steps, with the first 5 as warmup and the final 10 as
the measurement window. Run four arms from the same code tree in one allocation,
sequentially so every arm uses the same physical node and software environment:

| Arm | Backend | `offload_fraction` | Purpose |
|---|---|---:|---|
| A0 | FSDP2 | 0 | non-offloaded reference |
| B0 | M-FSDP Route A | 0 | zero-overhead and non-offload comparison |
| A1 | FSDP2 `FP32AdamW(cpu_update=True)` | 1 | CPU-update reference |
| B1 | M-FSDP Route A | 1 | bounded-pipeline candidate |

Before the allocation, the zero-GPU configuration-only gate must execute the
complete runtime initialization chain and print the resolved four-arm config.
The 8-GPU job must print an explicit non-skip marker, source commits, container
identity, Python/torch/TE locations, and all eight device names. Slurm job id,
submit/start/first-diagnosis/end timestamps, `nvidia-smi` diagnosis, and final
`sacct` state/exit code are required evidence.

For each performance arm, record:

- optimizer CUDA-event time and host wall time;
- end-to-end step time and tokens/s;
- finite loss and grad norm for all 15 steps as a sanity check, without using
  this short trace to claim precision readiness;
- steady allocated/reserved bytes after steps 2-5;
- full-step allocated/reserved peak for each measured step;
- optimizer-phase allocated/reserved peak for each measured step;
- host RSS, pinned-ring allocated bytes, D2H/H2D bytes, and ring high-water mark;
- count and total bytes of live full-param, full-grad, and transfer leases at
  optimizer entry and return.

The performance and memory gate is conjunctive and uses the corresponding
FSDP2 mode as the reference:

- B0 optimizer time and end-to-end time are strictly lower than A0;
- B1 optimizer time and end-to-end time are strictly lower than A1;
- B0 steady, full-step, and optimizer-phase CUDA peaks are each strictly lower
  than A0;
- B1 steady, full-step, and optimizer-phase CUDA peaks are each strictly lower
  than A1;
- B0 creates zero CPU-offload state, while B1's measured pinned high-water mark
  does not exceed `4*min(N, 2C)` bytes (and therefore never exceeds `8C`) and
  has no live transfer lease on return;

### Precision: independent 50 optimizer-step paired run

Run a second single-node 8xH100 job with the same 8-layer/8-expert, DP8,
4-microbatch, sequence-1024 formal configuration. It executes 50 optimizer
steps, with no substitution by a shorter trace. The three arms are:

| Arm | Backend | `offload_fraction` | Precision question |
|---|---|---:|---|
| PM | M-FSDP Route A | 1 | Candidate path under test |
| PF | FSDP2 `FP32AdamW(cpu_update=True)` | 1 | FSDP2 lifecycle reference |
| PD | dist_opt with native MCore HDO | 1 | Independent optimizer reference |

All three arms start from byte-identical initialization and consume the same
batch and seed at every optimizer step. Record every step's loss and grad norm,
not only an average or final value. Report at least these paired maxima:

```text
max_t abs(loss_PM[t] - loss_PF[t]) / max(abs(loss_PF[t]), epsilon)
max_t abs(loss_PM[t] - loss_PD[t]) / max(abs(loss_PD[t]), epsilon)
```

Both maximum loss relative differences must be at most the existing 0.509%
gate. Report the corresponding maximum grad-norm relative differences without
inventing a new acceptance threshold. All 50 losses and grad norms must be
finite. The real 8-GPU composition must also retain the genuine FP32
accumulation check; the `256 + 1 == 257.0` regression and multi-microbatch BF16
`exact == False` regression remain mandatory prerequisites.

The dist_opt arm is required because FSDP2 and Route A intentionally share the
same update kernel and therefore are not independent numerical references.
All three arms must use a byte-identical MLite tree, benchmark, and model
configuration. dist_opt may additionally load one declared, fixed MCore commit;
that exception must not change the MLite tree or any variable other than the
optimizer backend.

### Checkpoint continuity

A same-configuration 8-GPU save/resume continuation must have the same next-step
optimizer state as its uninterrupted paired arm and remain within the 0.509%
loss gate. This continuity run cannot use a smaller GPU count, model truncation,
or sequence length. It may share the 50-step precision allocation only if its
uninterrupted and resumed paths still consume explicitly identical batches and
seeds and both paths' stepwise evidence is retained.

### Combined readiness decision

Readiness requires every performance, memory, precision, FP32-accumulation, and
checkpoint-continuity condition above. There is no substitute threshold: if any
one comparison fails, Route A is not ready. Report the measured trade-off and
return for a design decision instead of weakening the gate.


## Existing evidence and remaining unknowns

The previous formal 8-GPU workload is directly relevant but predates Route A:

| Backend | Job | Optimizer | End-to-end | CUDA peak |
|---|---:|---:|---:|---:|
| M-FSDP offload=1 | 15149072 | 1032.948 ms | 2.334340 s | 16.838571 GB |
| FSDP2 CPU update | 15158512 | 1470.243 ms | 2.662194 s | 9.327149 GB |

It shows that the old M-FSDP CPU path was faster, but its peak was 80.53% above
FSDP2 and therefore failed the memory requirement. The subsequent M-FSDP memory
work changed gradient-scratch lifetime and pooling, so the old peak cannot be
used to judge the candidate tree. There is also no accepted same-tree A0/B0
formal result. Consequently, neither readiness mode is currently proven.

## Risks and stop conditions

- Sharing the algorithm while retaining two incompatible state formats can
  silently corrupt resume. Stop unless legacy next-step parity is demonstrated.
- Advancing the Adam step once per slice changes bias correction. Stop unless
  the counter is demonstrably per parameter.
- Reading `.grad` instead of M-FSDP `main_grad` loses genuine FP32 accumulation.
  Stop on the low-bit check, even if dtype checks pass.
- Returning before the final H2D event makes export and the next gather race
  stale weights. Stop on any live transfer lease at optimizer return.
- A shared module that imports either backend violates layering; a backend
  wrapper containing another AdamW update loop violates the no-third-optimizer
  decision.
- Faster optimizer timing alone is insufficient. Precision evidence and all
  three memory phases are mandatory.

Subject to those gates, Route A is the smallest reviewable way to eliminate
duplicated AdamW math and unbounded gradient staging while preserving M-FSDP's
distinct sharding lifecycle.

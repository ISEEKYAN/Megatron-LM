# M-FSDP offload Route B: direct MCore HDO reuse research

## Decision status

**GO for direct import; NOT READY to claim the CPU-gradient-residency memory
goal without a separate HDO change.**

The fixed runtime MCore exposes `HybridDeviceOptimizer` (HDO), and a genuine
single-node 8-GPU container probe imports and constructs it successfully.  The
minimal Route-B direction is therefore a thin M-FSDP lifecycle adapter around a
direct HDO import, not a local reimplementation.  This is not a claim that
unmodified HDO already meets M-FSDP's CPU-memory target: its first step allocates
a full CPU gradient staging tensor for every CPU-offloaded parameter, and its
overlap mode creates one CPU optimizer per parameter.

## Evidence source and scope

- Examined MLite source: clean clone `/tmp/mfsdp-task-1.13.24.21-isolated`, HEAD
  `50a644980b871c981e4ac05ee7711e32f6a53266`; all local source/report/probe
  commands explicitly entered this clone.  The shared checkout is not a source
  of fact for this report.
- The clone contains only `experimental/lite/...` as its implementation tree.
  `rg -n -i 'HybridDeviceOptimizer' . -g '*.py' -g '*.md'` produces no matches.
  This establishes only that HDO is not vendored by MLite; it does **not** say
  whether the fixed runtime MCore exposes it.
- Runtime source: fixed MCore
  `/lustre/.../megatron_lite/mcore-pinned-6204b925` at
  `6204b925f3da8b998524c6bb47a9ca779d95ce2e`, observed in the fixed
  `verl.vllm023.sqsh` container.  No shared MLite checkout was used.

## Existing M-FSDP contract to compare with runtime HDO

M-FSDP's current offload implementation is self-contained:

- `experimental/lite/megatron/lite/primitive/optimizers/mfsdp/cpu_offload.py`
  defines `CpuAdamGroup`.  It owns the authoritative pinned CPU FP32 master,
  FP32 `exp_avg`/`exp_avg_sq`, CPU gradient buffers, dedicated D2H/H2D streams,
  event ordering, and checkpoint serialization.
- `.../mfsdp/optimizer.py` selects this path from `offload_fraction`, splits
  parameter groups by element count, rejects a non-Adam optimizer and a custom
  optimizer factory, and makes the CPU group part of the M-FSDP step lifecycle.
- Existing M-FSDP tests assert the exact offload byte ledger and device contract:
  bf16 parameter and FP32 `main_grad` remain on CUDA; master and both Adam
  moments become CPU-resident for full optimizer offload.

An HDO direct-import decision requires the following runtime comparison; the HDO
column is intentionally pending rather than inferred from an absent MLite copy.

| Contract point | In-tree M-FSDP evidence | Runtime-HDO evidence required |
| --- | --- | --- |
| import and construction | `CpuAdamGroup(gpu_param_groups, lr, betas, eps)` | `from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer`; `HDO(params, offload_fraction, cpu_optimizer_cls, gpu_optimizer_cls, param_update_in_fp32, pin_cpu_grads, pin_cpu_params, overlap_cpu_optimizer_d2h_h2d, **kwargs)` |
| parameter/master ownership | per-parameter pinned CPU FP32 leaves | full offload clones CUDA params to pinned CPU (MCore `hybrid_optimizer.py:273-283`); `param_update_in_fp32=True` creates FP32 master copies (`:277-280`) and persists `master_param` in state (`:315-321`) |
| gradients | reads `param.main_grad`, else `param.grad` | HDO reads `getattr(param, "decoupled_grad", param.grad)` (`:91`, `:102`): an adapter can bind M-FSDP FP32 `main_grad` to `decoupled_grad` |
| update/device ordering | D2H and H2D streams plus per-param events | D2H stream copies gradients and records per-CPU-optimizer events (`:162-174`); post-step H2D hooks copy parameters back (`:117-148`) |
| offload selection | `offload_fraction` split by parameter numel | native parameter-numel threshold supports full and partial offload (`:251-292`) |
| optimizer semantics | AdamW, currently rejects non-Adam/custom factory | explicit CPU and GPU optimizer classes; successful probe used `torch.optim.AdamW` and TE `FusedAdam` |
| checkpoint | optimizer plus CPU masters, legacy-load compatibility | HDO pre/post load hooks remap FP32 state and rebuild sub-optimizers (`:383-438`); M-FSDP must integrate its own checkpoint lifecycle |

This establishes only direct-import and construction feasibility.  It does not
establish that the existing HDO data path is a drop-in replacement for M-FSDP's
bucket lifecycle, export/wake lifecycle, CPU-memory contract, numerical
contract, or performance target.

This follows `primitive.optimizer.fsdp`: the optimizer primitive owns
sharding/offload and must preserve the materialized-update invariant; it also
follows `basic.constitution`'s smallest-reviewable-design rule.

## Runtime probe evidence and bounded implementation direction

`validation/probe_mcore_hdo_runtime.sbatch` was synced from this clean clone to
a task-specific remote directory and run in `verl.vllm023.sqsh` on a single
node with 8 GPUs.  The final valid job is **15353917**:

- `sacct`: `COMPLETED`, `ExitCode=0:0`; submitted 2026-08-08T07:08:55-07:00,
  RUNNING 2026-08-08T07:08:57, first diagnosis within the five-minute gate.
- `/usr/bin/python3`, `command -v python3=/usr/bin/python3`, and
  `readlink -f /usr/bin/python3=/usr/bin/python3.12`; TE is
  `/usr/local/lib/python3.12/dist-packages/transformer_engine/__init__.py`.
- MCore commit is `6204b925f3da8b998524c6bb47a9ca779d95ce2e`; HDO imports from
  `.../cpu_offloading/hybrid_optimizer.py`, class lines 14-472.
- Exact class and `__init__` signature:
  `(params, offload_fraction=0.5, cpu_optimizer_cls=None, gpu_optimizer_cls=None,
  param_update_in_fp32: bool=False, pin_cpu_grads: bool=True,
  pin_cpu_params: bool=True, overlap_cpu_optimizer_d2h_h2d: bool=True, **kwargs)`.
- Construction with a CUDA bf16 parameter, `offload_fraction=1.0`,
  `param_update_in_fp32=True`, `torch.optim.AdamW`, and TE `FusedAdam`
  succeeded: `cpu_optimizers=1`, `gpu_optimizer_present=False`, CPU-copy and
  master dtypes are FP32, and the gradient input is the `decoupled_grad`
  fallback surface.

Historical probes are deliberately not mixed into the success claim:

- **15353623**: valid 8-GPU environment identity (`ExitCode=0:0`), but its
  import-path candidate was invalid; it is not Route-B import evidence.
- **15353814**: HDO construction returned successfully, then the probe tried
  the nonexistent plural `gpu_optimizers` field.  Its `1:0` is a
  probe-reporting bug, not an HDO construction failure.

## Route-B lifecycle contract: export, sleep/wake, and ownership

Route B is feasible only as an explicit optimizer-facing adapter.  The probe
does not exercise this contract; the following is the required implementation
and test boundary, not a claim that it already works.

1. **Gradient export.** After M-FSDP has finished reducing and materializing a
   bucket, the adapter must expose that bucket's authoritative FP32
   `param.main_grad` as `param.decoupled_grad` before HDO reads it.  It must
   neither allocate a second GPU gradient nor substitute a stale `param.grad`.
   The export is valid only until HDO has consumed the step; the adapter then
   clears the temporary alias/reference so the next M-FSDP accumulation owns
   the surface again.
2. **Offload/wake ordering.** A CPU-offloaded parameter is not ready for the
   next forward until HDO's post-step H2D copy has completed on its producer
   stream.  The adapter must record the HDO completion event and make the
   M-FSDP all-gather/compute stream wait before it re-materializes that
   parameter.  Conversely, HDO's D2H grad copy must wait for M-FSDP's
   bucket-ready event.  No CPU optimizer may read a tensor while its D2H copy
   is in flight.
3. **State and checkpoint boundary.** The adapter owns only the alias, bucket
   events, and wake barriers.  HDO owns its CPU parameters, FP32 master (when
   enabled), moments, and HDO state-dict hooks; M-FSDP continues to own shard
   metadata and its externally visible checkpoint envelope.  Save/load must
   prove a round trip without duplicate masters or an un-restored wake event.
4. **Failure behaviour.** A missing `main_grad`, an unrecorded bucket-ready
   event, or an unresolved post-step H2D event is an error, not a fallback to
   `param.grad`, a global synchronize, or a silent no-op.  The adapter may not
   export model-specific knowledge into this primitive.

Required pre-production CPU/GPU contract tests are: alias identity and dtype;
one export per optimizer step; no alias surviving a step; D2H-before-CPU-step;
H2D-before-next-use; partial-offload selection; checkpoint round trip; and the
existing FP32-accumulation regression.  These tests are the evidence needed to
upgrade this section from design to implementation feasibility.

### Why unmodified HDO is not ready for the CPU-gradient-residency target

- On first use, `_set_sub_optimizer_grads` allocates
  `cpu_copy_map_grad[param] = torch.empty(param.shape, ...)` for every CPU
  parameter (`hybrid_optimizer.py:98-115`).  This is an approximately **4N**
  FP32 CPU staging allocation.  `pin_cpu_grads=False` merely changes the
  allocation's pin flag; it does not remove the allocation.
- With overlap enabled, `build_cpu_optimizer_list` deliberately instantiates
  one CPU optimizer for each parameter (`:227-249`).

Consequently no configuration flag honestly removes the CPU-resident grad
staging.  If the target is “CPU does not keep full gradients resident,” it is a
separate bounded HDO/upstream change—not a Route-B adapter option.  Do not use
dummy/toy performance to claim otherwise.

### Steady-state byte ledger and bounded-ring decision

Let `N_cpu` be the number of full-offloaded parameter elements and `N_gpu` the
remainder; let `P` be the number of CPU-offloaded parameter tensors.  The
following is the mandatory accounting surface for the implementation review.
It deliberately separates persistent state from transient step peaks.

| Residency / owner | Full-offload steady state | Partial-offload steady state | Contract boundary |
| --- | --- | --- | --- |
| GPU model parameters | M-FSDP sharded bf16 storage plus materialization governed by M-FSDP | same for `N_gpu`, plus normal M-FSDP shards | adapter may not add a permanent full GPU parameter copy |
| GPU gradients | M-FSDP FP32 `main_grad` for the optimizer-step lifetime | proportional to active M-FSDP shards | `decoupled_grad` is an alias, not another allocation |
| GPU optimizer state | zero for fully CPU-offloaded parameters | HDO GPU optimizer state only for `N_gpu` | report `N_gpu` separately; no mixed-arm aggregation |
| CPU parameter/master | HDO CPU parameter copy plus, with `param_update_in_fp32=True`, FP32 master: **8N_cpu bytes** | **8N_cpu bytes** | measured allocator/reserved values must accompany this symbolic ledger |
| CPU Adam moments | FP32 `exp_avg` + `exp_avg_sq`: **8N_cpu bytes** after first step | **8N_cpu bytes** | state is persistent after lazy initialization |
| CPU gradient staging | HDO `cpu_copy_map_grad`: **4N_cpu bytes** after first use | **4N_cpu bytes** | persistent in current HDO, regardless of `pin_cpu_grads` |
| CPU optimizer objects | one per CPU parameter when overlap is enabled: **O(P)** | **O(P_cpu)** | metadata/object overhead must be reported separately, not hidden in tensor bytes |

Thus the current HDO CPU tensor steady state is at least **20N_cpu bytes**
(CPU parameter + FP32 master + two moments + persistent FP32 grad staging),
before allocator padding and Python/optimizer-object overhead.  This is a
symbolic design ledger, not a measured 8-GPU result.  The Route-B adapter may
reuse HDO unchanged only if this 4N persistent grad residency is accepted.

If it is not accepted, the only in-scope design is a separately reviewed HDO
bounded reusable staging ring: `R` pinned FP32 buffers of at most the bounded
bucket capacity `B`, yielding **4RB bytes** rather than 4N bytes, with
`R >= 2` only when copy/update overlap requires two live buffers.  Its required
ownership protocol is `free -> D2H-in-flight -> CPU-update -> H2D-in-flight ->
free`, with an event at every transition; it must back-pressure rather than
reuse an in-flight buffer.  A ring changes HDO internals/upstream behavior and
is explicitly **not** included in a direct-import Route-B implementation.

## Formal GPU validation design (post-approval only)

Both jobs run as **single-node, 8-GPU Slurm jobs** (`torchrun --nproc_per_node
8`), using the same Qwen3.5 model family, `SEQ_LEN=1024`,
`NUM_MICROBATCHES=4`, `TRUNCATE_LAYERS=8`, `KEEP_EXPERTS=8`, and MTP disabled.
They are deliberately not a 2-GPU, 2-layer, or sequence-128 toy run.  Record
the commit, container, Slurm job id, allocation/RUNNING/first-diagnosis times,
`sacct` exit code, output JSON, and W&B URL.  Diagnose a RUNNING job within five
minutes; cancel and retain a stack trace if it hangs.

### A. Performance: separate 15-step job

Use `experimental/lite/examples/bench/scripts/run_qwen35_mfsdp_offload.sh` as
the workload baseline, with `DRY_RUN=0`, `NPROC=8`, `STEPS=15`, and `WARMUP=5`.
Run comparable arms in separate fresh processes with identical checkpoint/model,
topology, synthetic data generation, optimizer hyperparameters, and offload
fraction.  The matrix must include both offload boundaries:

1. M-FSDP `offload_fraction=0` baseline, using its native GPU optimizer path;
2. M-FSDP current `CpuAdamGroup`, full optimizer offload;
3. approved Route-B adapter/HDO, full optimizer offload;
4. FSDP2 `offload_fraction=0` and full-offload reference configurations;
5. dist_opt/HDO `offload_fraction=0` and full-offload reference configurations.

An arm is comparable only if model family, sharding/parallel topology, global
batch, precision, checkpoint, optimizer hyperparameters, and offload fraction
match.  A backend that cannot match the topology remains a separately labelled
non-comparable diagnostic; it cannot be used to rank Route B.

The performance artifact must publish `avg_step_ms`,
`avg_optimizer_step_ms`, `tok_per_s`, `peak_mem_gb`, and per-arm configuration.
There are exactly 10 measured optimizer steps after warmup.  It must also
publish allocator allocated/reserved and the byte ledger above at steady state,
then three distinct peak measurements for each arm: (a) steady state after
warmup, (b) the complete optimizer step from gradients-ready through CPU/GPU
update and H2D completion, and (c) the full training step including
forward/backward, gradient materialization, optimizer step, and wake barrier.
Each peak must name the sampling boundary and include the CUDA peak-reset
boundary; CPU RSS/pinned-memory and GPU allocated/reserved cannot be conflated.
Treat this as performance/memory evidence only (`perf.measure` and
`application.bench`); it cannot establish numerical correctness.

### B. Precision: independent 50-optimizer-step job

Run a distinct 8-GPU Slurm allocation for 50 optimizer steps.  Construct all
three precision arms from byte-identical initial weights and optimizer
hyperparameters; at each step feed the same globally indexed batch and seed to
every arm.  The required precision triplet is current M-FSDP full offload,
proposed Route-B M-FSDP full offload, and the matched full-offload FSDP2 or
dist_opt reference selected before launch.  The `offload_fraction=0` matrix
arms are required baseline controls for the performance/memory design above;
they are not silently substituted for a full-offload numerical reference.  No
timing warmup samples may be repurposed as precision evidence.

The runner must emit, for every step and arm, `loss`, `grad_norm`, global batch
index, seed, and post-update parameter fingerprint.  Compare every Route-B
loss against its matched reference and report the maximum over the 50 steps:

```
max_loss_rel_diff = max_i(abs(loss_route_b[i] - loss_ref[i]) /
                          max(abs(loss_ref[i]), epsilon))
```

Acceptance requires `max_loss_rel_diff <= 0.00509` (0.509%) and an explicit
per-step loss/grad-norm table; failures must retain the first divergent step
and both fingerprints.  Retain the independent FP32 accumulation regression
as a required companion check, rather than weakening it to the relative-loss
gate.  This is the end-to-end precision line required by
`basic.align_e2e_precision`, not an extrapolation from the benchmark.

## Existing validation surfaces to extend after approval

- `experimental/lite/tests/unit/primitive/test_mfsdp.py`: CPU-offload device,
  byte-ledger, numerical, partial-offload, and checkpoint contracts.
- `experimental/lite/tests/smoke/primitive/test_mfsdp_parity_smoke.py`:
  existing 8-GPU offload matrix already records loss and grad norm internally;
  extend it (or a dedicated runner) to emit all 50 per-step observations.
- `experimental/lite/examples/bench/scripts/run_qwen35_mfsdp_offload.sh`:
  already enforces Slurm for real runs and supplies the required 15/5 benchmark
  defaults.

## Research execution ledger and honest status

This research did consume GPU time.  The valid final runtime probe was Slurm
job **15353917**, requested as one node / **8 GPUs** / three-minute walltime;
it was submitted at `2026-08-08T07:08:55-07:00`, RUNNING at
`2026-08-08T07:08:57-07:00`, and `sacct` recorded `COMPLETED ExitCode=0:0`.
Its purpose was runtime import/construction evidence only.  It is not a
performance, memory, training, or 50-step precision run.  Earlier probe jobs
are described above only to preserve their validity boundaries.

No `vk-flow run` (or equivalent task-bound research-run record) was created
for job 15353917.  Therefore the probe's task binding is reconstructed from the
task-specific sbatch filename, remote output directory, recorded task log, and
the checked-in `validation/probe_mcore_hdo_runtime.sbatch`; that is weaker than
a first-class research-run record.  This gap is explicitly recorded rather
than represented as zero GPU cost or as a validated formal experiment.  Any
future performance or precision allocation must be created through the
task-bound run workflow before submission and must retain the job id, `sacct`,
raw log, and output artifacts.

**Current decision:** direct HDO import is feasible at MCore
`6204b925f3da8b998524c6bb47a9ca779d95ce2e`, conditional on implementing and
testing the lifecycle contract above.  It is **not approved to implement in
this task**, and neither the HDO 4N CPU-gradient residency nor the formal
8-GPU performance/memory/precision acceptance design has been validated.  The
stale contrary conclusion that no Slurm job existed has been removed.

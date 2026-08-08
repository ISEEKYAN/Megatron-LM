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

This establishes direct-import feasibility.  It does not establish that the
existing HDO data path is a drop-in replacement for M-FSDP's bucket lifecycle
or CPU-memory contract.

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

The smallest implementation task is: direct-import HDO; add a lifecycle adapter
that binds M-FSDP FP32 `main_grad` to HDO `decoupled_grad`; and explicitly
bridges M-FSDP bucket/event ordering and checkpoint ownership.  It must retain
existing FP32-accumulation regression coverage and add the adapter contract
tests before changing the production path.  The adapter stays optimizer-facing
only—no model knowledge belongs in this primitive.

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
separate bounded HDO/upstream change—e.g. a bounded reusable staging ring—not
a Route-B adapter option.  Do not use dummy/toy performance to claim otherwise.

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
fraction:

1. current in-tree `CpuAdamGroup` M-FSDP full optimizer offload;
2. approved Route-B adapter/HDO M-FSDP full optimizer offload;
3. FSDP2 full optimizer-offload reference, if its configuration matches the
   same model and parallel topology.

The performance artifact must publish `avg_step_ms`,
`avg_optimizer_step_ms`, `tok_per_s`, `peak_mem_gb`, and per-arm configuration.
There are exactly 10 measured optimizer steps after warmup.  Treat this as
performance/memory evidence only (`perf.measure` and `application.bench`);
it cannot establish numerical correctness.

### B. Precision: independent 50-optimizer-step job

Run a distinct 8-GPU Slurm allocation for 50 optimizer steps.  Construct all
three arms from byte-identical initial weights and optimizer hyperparameters;
at each step feed the same globally indexed batch and seed to every arm.  The
three arms are the same current M-FSDP, proposed Route-B M-FSDP, and matched
FSDP2 reference arms above.  No timing warmup samples may be repurposed as
precision evidence.

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

## Research status

No implementation, GPU job, or performance/precision claim is made by this
research task.  The immediate next action is the required runtime import probe,
not a final direct-import ruling.

Execution is currently blocked by this worker environment: on 2026-08-08,
`sbatch validation/probe_mcore_hdo_runtime.sbatch` returned
`sbatch: command not found`, and `/lustre` is not mounted.  Consequently no
Slurm job was created and no container/MCore import result is claimed.  Run the
checked-in probe from the scheduler login environment, then append its job id,
`sacct` result, MCore HEAD, and emitted file/line/signature records here before
making the final decision.

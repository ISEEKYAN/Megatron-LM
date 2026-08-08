# M-FSDP offload Route B: direct MCore HDO reuse research

## Decision status

**PROVISIONAL — no direct-import decision has been made.**

The prior conclusion based solely on an MLite-tree text search is withdrawn.
MCore is a runtime dependency, so its actual pinned source and container import
surface must be probed before Route B can be accepted or rejected.  The required
probe is included in `validation/probe_mcore_hdo_runtime.sbatch`; its raw output
is a prerequisite for any final GO/NO-GO decision.

## Evidence source and scope

- Examined source: clean clone `/tmp/mfsdp-task-1.13.24.21-isolated`, HEAD
  `a9c8d2f4795d8167bebaa2c354c4de27712afc9f` (clean `git status --porcelain`).
- The clone contains only `experimental/lite/...` as its implementation tree.
  `rg -n -i 'HybridDeviceOptimizer' . -g '*.py' -g '*.md'` produces no matches.
  This establishes only that HDO is not vendored by MLite; it does **not** say
  whether the fixed runtime MCore exposes it.
- No result here is derived from the shared checkout.  No upstream fetch is
  performed: the supplied clean-clone-only constraint forbids using its local
  shared-checkout remote as a reference source.

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
| import and construction | `CpuAdamGroup(gpu_param_groups, lr, betas, eps)` | imported module, class, `inspect.signature`, construction dependencies |
| parameter/master ownership | per-parameter pinned CPU FP32 leaves | whether HDO owns/shadows params and master tensors compatibly |
| gradients | reads `param.main_grad`, else `param.grad` | accepted gradient representation and dtype/device requirements |
| update/device ordering | D2H and H2D streams plus per-param events | copy streams, synchronization boundary, and CPU update timing |
| offload selection | `offload_fraction` split by parameter numel | full/partial policy and ordering semantics |
| optimizer semantics | AdamW, currently rejects non-Adam/custom factory | algorithm/group/default handling |
| checkpoint | optimizer plus CPU masters, legacy-load compatibility | state-dict keys, ownership, and restore contract |

If the actual import and signature fit these contracts with a bounded adapter,
Route B remains viable.  A final NO-GO is justified only by recorded import or
construction failure, or by a demonstrated non-adaptable contract conflict.

This follows `primitive.optimizer.fsdp`: the optimizer primitive owns
sharding/offload and must preserve the materialized-update invariant; it also
follows `basic.constitution`'s smallest-reviewable-design rule.

## Required runtime probe before Route B implementation

1. Submit `validation/probe_mcore_hdo_runtime.sbatch` in the established
   `verl.vllm023.sqsh` container.  It records `which python`,
   `transformer_engine.__file__`, `megatron.core.__file__`, MCore git HEAD, HDO
   module/class path, source file and line range, and constructor signature.
2. Record the probe's `sacct` exit status and raw output in this report, then
   fill the runtime-HDO column in the contract table above.
3. Only after a positive probe, specify an adapter contract before code changes:
   parameter-group ordering,
   full and fractional offload selection, FP32 `main_grad` consumption,
   CPU-master/moment ownership, nonblocking copy/event ordering, overflow and
   grad clipping behavior, and state-dict round trip.
4. Keep model knowledge out of the optimizer primitive.  The adapter may only
   consume optimizer-facing tensors and explicit configuration.
5. Preserve the existing FP32 accumulation regression and add contract tests
   before replacing the production path.

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

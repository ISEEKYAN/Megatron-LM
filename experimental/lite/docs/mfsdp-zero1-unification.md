# Unifying MLite optimizer backends around M-FSDP

This document is a source study and migration proposal. It does not implement a
new sharding mode, change a default, or claim new GPU results.

## Decision summary

M-FSDP is a credible common parameter backend, but it is not ready to replace
either `dist_opt` or FSDP2 by changing one default:

- MLite M-FSDP implements only fully sharded `optim_grads_params`. A
  DistOpt-compatible mode still needs persistent replicated compute parameters,
  sharded main parameters and gradients, one gradient reduce-scatter, and one
  post-update parameter all-gather.
- The functional M-FSDP snapshot and the performance snapshot are divergent
  branches. Their checkpoint/export fixes and latest communication changes have
  not yet been qualified as one source tree.
- M-FSDP is wired into Qwen3-MoE and Qwen3.5 only. `dist_opt` and FSDP2 are also
  wired into DeepSeek V4, GLM-5, and Kimi K2.
- VERL checkpointing bypasses the runtime's optimizer-backend adapter. M-FSDP's
  versioned optimizer state format therefore is not used by that production
  path.
- M-FSDP currently excludes frozen parameters from its buckets. A LoRA run can
  be numerically correct, but its frozen base weights remain replicated, so it
  does not have ZeRO-3 memory semantics.
- The M-FSDP optimizer factory is a useful algorithm seam, but it receives flat
  local shards. That is insufficient for Muon without a logical-matrix
  materialization contract and a backend-specific lowering.

The recommendation is to keep all three backend names during migration, add
`mfsdp` stages behind an explicit opt-in, and change the default only after each
stage independently passes its compatibility and performance retirement gates.
Performance is a release condition: M-FSDP stage 1 must be no slower than
`dist_opt`, and M-FSDP stage 3 must be no slower than FSDP2 under matched
optimizer, precision, model, topology, and workload contracts.

## Sources, snapshots, and terminology

The checked-out MLite baseline is `69ea18d07`. It contains `dist_opt` and FSDP2
but no M-FSDP package. This study also reads the following fixed snapshots:

| Snapshot | Purpose | Qualification boundary |
| --- | --- | --- |
| MLite `2a6356824` | M-FSDP checkpoint, export, resume, and offload behavior | Functional branch; not an ancestor of the performance branch |
| MLite `3997c3f15` | Same-optimizer three-arm benchmark and communication feature study | Performance branch; not an ancestor of the functional branch |
| MLite `62404d4ab` | Compact Muon + MCore DistOpt lowering | DistOpt-specific; not an M-FSDP Muon implementation |
| NVIDIA Megatron-LM `fd1121b8` | M-FSDP sharding strategies and buffer ownership reference | Upstream `dev` source snapshot |
| NVIDIA PR snapshot `c7d1aff65` | Prototype `FSDPTensorParallelMuon` | Prototype, not present in the surveyed `dev` snapshot |

The branch divergence matters. Results from `2a6356824` and `3997c3f15` are
useful evidence, but a merged implementation must rerun both suites.

### The “ZeRO-1” naming caveat

Canonical ZeRO terminology calls optimizer-state sharding stage 1 and adds
gradient sharding at stage 2. NVIDIA M-FSDP exposes the more precise strategies
`no_shard`, `optim`, `optim_grads`, and `optim_grads_params`
([`distributed_data_parallel_config.py:96-108`][mcore-ddp-strategy]). Its buffer
mapping is explicit: `optim` shards only main weights, `optim_grads` also shards
gradients, and `optim_grads_params` additionally shards model weights
([`param_and_grad_buffer.py:2005-2030`][mcore-buffer-mapping]).

MLite's existing `dist_opt` behavior uses sharded main parameters and gradients,
a gradient reduce-scatter, and a parameter all-gather. This document therefore
uses **M-FSDP stage 1** to mean the requested **DistOpt-compatible profile**, which
maps internally to NVIDIA's `optim_grads`, not canonical ZeRO-1 `optim`. The
public API must document this choice rather than silently conflating the two.

## Current ownership contracts

### `dist_opt`

`dist_opt` builds MCore DDP and `DistributedOptimizer` together. It enables
`use_distributed_optimizer`, constructs an MCore `TransformerConfig`, annotates
TP/expert metadata, and passes explicit process groups where possible
([`megatron_wrap.py:63-195`][mlite-distopt-build]). Its model parameters stay
replicated for forward/backward while the main parameter, main gradient, and
optimizer update state are sharded.

Its checkpoint is a special MCore distributed-checkpoint path. It declares a
fully reshardable optimizer format, attaches model sharded-state methods, and
calls the optimizer's sharded-state API
([`distckpt.py:25-92`][mlite-distopt-checkpoint]). This is materially different
from saving a per-rank Torch optimizer state dictionary.

### FSDP2

FSDP2 calls PyTorch `fully_shard` on selected units and the root before building
the optimizer. It uses separate dense DP×CP and expert-DP meshes and chooses a
divisible shard dimension when possible
([`wrap.py:117-194`][mlite-fsdp2-wrap],
[`optimizer.py:343-474`][mlite-fsdp2-build]). Parameters are DTensors between
computations and are materialized by FSDP2 around module execution.

Its HF export explicitly recognizes DTensor and calls `full_tensor()` before
the existing TP/EP gather and weight mapping
([`hf_weights.py:32-46`][mlite-fsdp2-export]). Its model checkpoint uses generic
DTensor+DCP placement records; its optimizer state is still a rank-local
sidecar, so cross-topology *optimizer* resume is not implied by model DCP
resharding ([`dcp.py:34-95`][mlite-generic-dcp]).

### Current MLite M-FSDP stage 3

The in-flight M-FSDP config accepts only `optim_grads_params` and rejects every
other strategy before allocation
([`config.py:26-46`][mfsdp-config], [`config.py:169-195`][mfsdp-config-lowering],
[`fully_shard.py:17-35`][mfsdp-fully-shard]).

For each bucket it:

1. flattens parameters and assigns arbitrary byte ranges to ranks;
2. stores persistent FP32 main-parameter and main-gradient shards;
3. replaces module parameters with one-dimensional local shard views;
4. all-gathers compute parameters before use; and
5. reduce-scatters full gradients back to local main-gradient shards.

The ownership and shard calculations are visible in
[`buffer.py:247-380`][mfsdp-bucket]. Forward hooks acquire buckets, saved-tensor
hooks re-acquire them for backward, and full parameters are released after use
([`wrapper.py:42-105`][mfsdp-wrapper]). The actual communication pipelines launch
parameter all-gather and gradient reduce-scatter on a side stream
([`buffer.py:759-872`][mfsdp-pipelines]).

This is a real stage-3 backend. It is not a stage-1 backend with a different
flag.

## Proposed stage-1 design

### Public API

Keep algorithm selection and parameter-backend selection separate:

```python
MegatronLiteConfig(
    optimizer=OptimizerConfig(algorithm="adam", backend="mfsdp"),
    parameter_backend=MFSdpConfig(zero_stage=1),
)
```

The temporary model-local `impl_cfg.optimizer` strings can accept aliases during
migration, but the construction compiler should normalize them to one immutable
record before model allocation:

```python
@dataclass(frozen=True)
class MFSdpConfig:
    zero_stage: Literal[1, 3] = 3

    @property
    def sharding_strategy(self) -> Literal["optim_grads", "optim_grads_params"]:
        # stage 1 is the DistOpt-compatible `optim_grads` profile
        return {1: "optim_grads", 3: "optim_grads_params"}[self.zero_stage]
```

Validation rules:

- `zero_stage=1` derives `optim_grads`; `zero_stage=3` derives
  `optim_grads_params`.
- A user cannot set both fields independently.
- Values 0 and 2 fail loudly until there is a separate requirement and
  validation plan.
- Stage-specific knobs are rejected when they do not execute in production.
  In particular, stage-3 forward prefetch knobs must not appear active in stage
  1, and unsupported multi-instance/offload modes must not be accepted and
  ignored.
- `dist_opt` and `fsdp2` remain explicit fallback backend names until retirement.

This follows the capability-compiler proposal: optimizer algorithm owns update
math; parameter backend owns materialization, gradient reduction, state
sharding, and offload; the runtime owns lifecycle ordering. No primitive checks
a model-name allowlist. The broader source-based composition design describes
this split as optimizer algorithm, parameter backend, precision, sequence
layout, and execution-capture ownership domains
([`feature-composability.md` at `ed2121463`][feature-composability]).

### Stage-1 data flow

```text
build
  replicated compute parameters remain installed in modules
  FP32 main parameters + optimizer state are partitioned by DP/DP×CP

forward/backward
  no parameter all-gather and no module parameter swapping
  full compute gradients accumulate across microbatches

last backward
  bucketed reduce-scatter(full compute grads -> local main-grad shards)
  apply the configured averaging rule exactly once

optimizer step
  update local FP32 main-parameter shards and local optimizer state

post-step parameter sync
  cast local updated main shards to compute dtype
  bucketed all-gather into persistent replicated compute-parameter buffers
  next forward reads those replicas directly
```

The global update must equal an unsharded optimizer given identical accumulated
gradients, and the partition must be deterministic. Those are the governing
`primitive.optimizer.distopt` invariants.

The current optimizer lifecycle steps local shards and then discards materialized
views ([`optimizer.py:235-358`][mfsdp-optimizer]); its norm implementation reduces
dense, TP-replicated, expert, and PP contributions separately
([`optimizer.py:101-154`][mfsdp-grad-norm]), using backend-safe local accumulation
and scalar reduction helpers ([`grad_norm.py:12-83`][mfsdp-grad-norm-helpers]).
Stage 1 changes the post-step materialization lifecycle, not those ownership
domains.

### Required changes by M-FSDP component

| Component | Current stage-3 behavior | Stage-1 change | Estimate |
| --- | --- | --- | --- |
| `config.py` / `fully_shard.py` | Hard-rejects everything except `optim_grads_params` | Add typed stage mapping, stage-specific validation, and truthful feature reporting | 1–2 engineer-days |
| `buffer.py` layout | One distributed model/main/grad layout; module bindings point to local 1-D shard params | Keep compute params replicated and persistently bound; shard only main params/gradients/state; retain a local compute shard solely as the post-step AG input | 4–6 days |
| `buffer.py` communication | AG before forward/backward, RS after backward | Disable execution AG hooks; retain last-microbatch RS; add exactly one post-step AG with overlap hooks that do not expose stale replicas | 3–5 days |
| `wrapper.py` | Installs forward and saved-tensor materialization hooks and discards full views | Stage 1 bypasses parameter swapping and saved-tensor hooks; `full_parameter_context` still gathers authoritative FP32 masters for exact checkpoint/export | 2–3 days |
| `optimizer.py` | Steps local shards, then releases all full views | Step local shards, launch/wait replica refresh, preserve `grad_sync_enabled` microbatch semantics, and fail before the next forward if refresh is incomplete | 2–4 days |
| `grad_norm.py` / optimizer adapter | Sums local shard norms across dense DP×CP, TP, expert, and PP groups | Reuse shard ownership, but prove no duplicated DP contribution and match DistOpt's TP-replicated/expert reductions and overflow result | 2–3 days |
| checkpoint/offload backend | Versioned local M-FSDP sidecar, same-layout contract | Record stage and bucket layout; define stage-1 replica/master restoration and make incompatible stage/topology loads fail loudly | 2–4 days, excluding format migration |

Core stage-1 construction is approximately **14–22 engineer-days** before
cross-model, application, and GPU qualification.

### Communication and performance hypothesis

Let `D` be the data-parallel group size, `P` the compute-parameter bytes, and
`G` the gradient communication bytes. Ignoring protocol constants, the per-rank
collective payload is:

| Backend | Forward/backward parameter AG | Gradient sync | Post-step parameter AG |
| --- | --- | --- | --- |
| `dist_opt` | 0 | `G × (D-1)/D` RS | `P × (D-1)/D` AG |
| proposed M-FSDP stage 1 | 0 | `G × (D-1)/D` RS | `P × (D-1)/D` AG |
| current M-FSDP stage 3 | at least `2P × (D-1)/D` across forward and backward | `G × (D-1)/D` RS | no persistent replica refresh |

Activation recomputation can add stage-3 materializations. Stage 1 therefore
has the same first-order communication volume as DistOpt, not an intrinsic
bandwidth advantage. It can win only through better bucketing, overlap, fused
copy/cast, lower framework overhead, or better buffer registration. It can lose
through wrapper hooks that should have been bypassed, extra FP32↔BF16 copies,
allocator overhead, small buckets, or weaker overlap than mature MCore DDP.

The existing feature study is a warning against assuming those optimizations
work. On its tiny workload, bucket splitting, AG overlap, RS overlap, and double
buffering measured `0.9075×`, `0.9933×`, `0.9922×`, and `0.9994×`; prefetch had
no production call, and NCCL user-buffer/registered-buffer requests fell back
([performance snapshot `3997c3f15`][mfsdp-perf-commit], Slurm 13696986). Stage 1
must earn its performance claim in a new matched benchmark.

## Compatibility with `dist_opt`

### Consumption inventory

| Surface | Existing `dist_opt` contract | M-FSDP stage-1 status | Gap and estimate |
| --- | --- | --- | --- |
| Five model protocols | DeepSeek V4, GLM-5, Kimi K2, Qwen3.5, and Qwen3-MoE all default to and construct `dist_opt` ([DeepSeek:407-454][deepseek-optimizer], [GLM:252-298][glm-optimizer], [Kimi:212-258][kimi-optimizer], [Qwen3.5:226-272][qwen35-optimizer], [Qwen3-MoE:223-292][qwen3moe-optimizer]) | M-FSDP branches wire only Qwen3.5 and Qwen3-MoE | Move backend construction into a shared assembler, pass generic unit types/expert classifier, and qualify all five; 3–5 days plus GPU matrix |
| Training lifecycle | Runtime toggles delayed grad sync on the last microbatch and finalizes gradients; names still say `dist_opt` even though the branch uses them for FSDP2/M-FSDP ([`train_step.py:15-74`][mlite-train-loop]) | Mechanism works in the M-FSDP validation branch, including PP, but the interface leaks one backend name | Replace `dist_opt` boolean with a backend capability/hook and make PP/non-PP use one lifecycle; 1–2 days |
| Gradient norm/overflow | MCore optimizer owns TP/EP/PP-aware norm, clipping, loss scaling, overflow, and chained-optimizer semantics | M-FSDP matches the basic norm groups but returns `num_zeros=0` and has no equivalent mixed-precision/loss-scaler surface | Add independent DistOpt parity for norm, clipping, NaN/Inf skip, zero count if required, chained algorithms, and precision-aware mode; 3–6 days |
| Distributed checkpoint | Fully reshardable MCore model+optimizer state ([`distckpt.py:25-116`][mlite-distopt-checkpoint]) | Versioned M-FSDP sidecar saves local main params and optimizer state and promises the same bucket topology only ([`backend.py:11-92`][mfsdp-backend]) | Define a backend-neutral sharded optimizer schema or a one-time converter; cross-backend and topology migration 5–8 days |
| Resume/export | DistOpt model params are already replicated; HF export and rollout sync consume plain tensors | M-FSDP functionally proved save/resume and exact FP32 export through a full-main-param lease | Merge divergent source first, then repeat every model/PP/EP mapping; 2–4 days |
| Offload | MCore supports fractional optimizer offload and overlap through canonical optimizer fields | M-FSDP proved whole model/grad/state CPU round-trip but does not implement fractional update-state offload or the same overlap policy | Decide supported closed profiles; full-only parity 2–3 days, fractional overlap 4–7 days |
| Muon | MCore DistOpt uses the complete Muon metadata/DDP/LayerWise transaction; direct partial construction is rejected ([Muon lowering `megatron_wrap.py:107-242`][muon-distopt-config]) | Factory seam exists, but production M-FSDP Muon wiring, state resume, and performance remain unimplemented | See the dedicated Muon section; 7–12 days |
| VERL | Runtime handles zero-grad, step, and export, but VERL directly calls checkpoint primitives ([`mlite_engine.py:316-386`][verl-runtime-calls], [`mlite_engine.py:421-509`][verl-checkpoint]) | Launchers accept only `dist_opt`/`fsdp2`; direct checkpoint misses M-FSDP backend state handling | Route checkpoint through runtime and add `mfsdp`+stage config, resume, offload, rollout sync; 2–4 days |
| miles | Backend choice maps only `dist_opt`/`fsdp2`; training/save/load/export otherwise use runtime ([`arguments.py:6-48`][miles-backend], [`backend_patch.py:339-469`][miles-runtime]) | Runtime shape is favorable, but CLI/default/config and PP-local export need qualification | Add staged config and full DAPO/SFT save/load/update path; 1–3 days |
| MCore dependency removal | `dist_opt` is an allowed but substantial MCore import exception | M-FSDP is self-contained except optional fused packages | Architectural benefit only after feature and performance parity; not a reason to waive gates |

### DistOpt retirement gate

Do not remove or stop testing `dist_opt` until all of the following are true:

1. M-FSDP stage 1 passes Adam/AdamW update and grad-norm parity on dense and MoE
   models with TP×EP×ETP×PP×CP, microbatch accumulation, and deterministic input.
2. All five model protocols and both application connectors build stage 1
   through the production assembler.
3. Checkpoints from the supported `dist_opt` release can resume through either
   direct load or a committed conversion tool; scheduler/RNG/optimizer state and
   the next update match.
4. Export/update-weights, full and supported fractional offload, NaN/Inf skip,
   and any advertised precision-aware profile pass.
5. Muon either passes its own stage-1 contract or fails loudly before allocation;
   it must not silently fall back to Adam.
6. The matched performance gate below passes. A regression keeps `dist_opt` as
   the default regardless of functional parity.
7. At least one release keeps `optimizer_backend="dist_opt"` as an explicit
   rollback alias with deprecation telemetry and no behavior change.

## Compatibility with FSDP2

### Consumption inventory

| Surface | Existing FSDP2 contract | M-FSDP stage-3 evidence | Remaining gap and estimate |
| --- | --- | --- | --- |
| Model coverage | All five model protocols select unit types, expert leaf handling, and meshes | Only Qwen3.5/Qwen3-MoE are wired | Shared assembler plus DeepSeek/GLM/Kimi qualification; 3–5 days |
| Parameter coverage | `fully_shard` wraps unit/root parameters regardless of whether the optimizer updates them | M-FSDP bucket construction skips `requires_grad=False` parameters ([`buffer.py:675-700`][mfsdp-buffer-build]) | Separate “shard for compute” from “owns optimizer state”; required for LoRA ZeRO-3 memory semantics; 3–5 days |
| Model checkpoint/resume | Model tensors use DCP placement metadata; optimizer uses a rank sidecar | M-FSDP save→new process→resume matched uninterrupted training on the tested topology | Make stage and layout metadata explicit; qualify topology changes and fail loudly where unsupported; 3–6 days |
| HF/rollout export | DTensor is materialized before model-specific TP/EP mapping | M-FSDP vs FSDP2 export matched 45 Qwen3-MoE tensors by key, shape, dtype, and value; max abs was zero | Merge the fix and repeat all models, PP-local export, MTP, and rollout transports; 2–4 days |
| LoRA import/export | Native Qwen3-MoE adapter code gathers TP/EP LoRA tensors directly and currently supports PP=1, ETP=1 ([`lora_adapter.py:321-420`][mlite-lora-export]) | No M-FSDP LoRA test; direct adapter APIs do not enter the runtime full-parameter lease | Add a generic full-parameter consumer context, sharded frozen-base coverage, train/save/load/export parity; 3–5 days |
| Offload | FSDP2 moves model/grad storage and DTensor optimizer state; update-state policies are separately tested | M-FSDP whole-state round-trip passed for the tested topology | Qualify frozen params, partial state/offload fractions, aliases, and stage-3 forward re-materialization; 3–5 days |
| Optimizer algorithms | Production builder accepts Adam/AdamW | M-FSDP accepts Adam/SGD or a factory, but only Adam has production evidence | Adam parity is enough for FSDP2 retirement; Muon is a separate capability and cannot be inferred from the factory |
| Performance | Mature PyTorch FSDP2 reference | M-FSDP has two different comparisons described below | Repeat on integrated source and production-sized dense/MoE/LoRA workloads; 4–7 days |

The functional snapshot `2a6356824` has strong but bounded evidence:

- Slurm 13697656, 8×H100, TP2×EP2×ETP1×PP2×CP2: FSDP2 and
  M-FSDP exported 45 tensors with identical keys, shapes, dtypes, and values;
  50-step maximum loss/grad-norm relative differences were `3.2691e-3` and
  `1.6981e-3`.
- Slurm 13697781, 8×H100: an independent save/resume process matched the
  uninterrupted next step; optimizer-state offload round-tripped 36 tensors;
  three exports had stable allocated bytes.

The performance evidence must be interpreted in order:

- An earlier backend-native run measured `470715.91 / 196970.06 = 2.3898×`
  M-FSDP/FSDP2 on 8×H100, but M-FSDP used Apex FusedAdam while FSDP2 used its
  FP32 AdamW adapter. It proves both production paths execute; it is not a fair
  backend-only speed ratio.
- The later matched run used Torch AdamW `foreach=False` and the same BF16
  master/shard contract. On tiny Qwen3-MoE TP2×EP2×ETP1×PP2×CP2 it measured
  `23562.21 / 19230.49 = 1.225×` M-FSDP/FSDP2, with maximum loss relative
  difference `0.001328` ([performance snapshot `3997c3f15`][mfsdp-perf-commit],
  Slurm 13697447). This is the stronger backend comparison, but it is still one
  tiny workload and cannot qualify the default globally.

### FSDP2 retirement gate

FSDP2 remains available until integrated M-FSDP stage 3 passes:

1. all five production model protocols, including expert mesh and MTP cases;
2. trainable and frozen parameter ownership, especially LoRA;
3. checkpoint/resume/export/offload and supported topology changes;
4. matched AdamW precision against both FSDP2 and NVIDIA M-FSDP;
5. matched performance with no required workload below the non-inferiority
   threshold; and
6. one release of explicit `fsdp2` rollback after the default changes.

## Muon seam

Muon cannot consume arbitrary flat byte shards as if they were matrices. The
current M-FSDP buffer creates one-dimensional shard parameters and records the
original rank only as an attribute
([`buffer.py:357-380`][mfsdp-bucket]). The factory then receives these already
sharded parameter groups ([`fused_ops.py:21-34`][mfsdp-factory]). That is an API
seam, not a valid Muon lowering.

The surveyed DistOpt branch delegates construction to MCore, where Muon is a
`TensorParallelMuon` wrapped by `LayerWiseDistributedOptimizer`; complete logical
matrices are the ownership unit while Adam fallback parameters retain ordinary
byte-sharded DistOpt semantics ([`emerging_optimizers.py:402-432`][mcore-tp-muon],
[`optimizer/__init__.py:861-966`][mcore-layerwise-muon]). M-FSDP cannot preserve
that behavior by passing its flat local shard parameters to the same constructor.

The backend must provide a typed record for every optimizer parameter:

```python
LogicalParameterShard(
    global_shape,
    global_offset,
    local_shape,
    tp_placement,
    dp_placement,
    expert,
    shared,
    qkv_layout,
    main_dtype,
)
```

Then an algorithm requirement selects a lowering:

- AdamW: elementwise local shard update.
- Muon stage 1: preserve whole-matrix ownership as MCore LayerWise does, or
  gather the pre-Newton–Schulz momentum, run matrix math, and reshard the update.
- Muon stage 3: bounded logical-matrix gather/NS/reshard or a proven distributed
  NS implementation.

The NVIDIA prototype `FSDPTensorParallelMuon` separates local momentum, all
boundary gathers, and local NS/update into three phases to avoid collective
order deadlocks ([`emerging_optimizers.py` in `c7d1aff65`][mcore-fsdp-muon]). It
depends on DTensor global shape/placement metadata. MLite M-FSDP uses plain
one-dimensional shard parameters, so it cannot import that class unchanged.

Muon support is complete only when production configuration selects the real
Muon algorithm, matrix and Adam-fallback groups match an independent reference,
checkpoint/offload restore its momentum state, and matched performance is
reported. A Torch SGD stand-in or a factory-call counter is not evidence.

## Performance acceptance protocol

This protocol follows `perf.measure`: a stable workload, warmup and repeated
measurements, complete throughput/time/memory metrics, and a precision result
from the same configuration.

### Stage-1 mandatory comparison

Run `dist_opt` and M-FSDP stage 1 in one Slurm allocation with:

- one integrated source commit and one frozen container;
- identical model weights, batches, seed, token count, sequence length,
  microbatching, recompute, TP/EP/ETP/PP/CP topology, optimizer algorithm,
  hyperparameters, compute/master/grad dtypes, clipping, and offload policy;
- a common optimizer implementation for the backend-only comparison: first
  Torch AdamW `foreach=False`, then Apex FusedAdam only if both paths use the
  same class and options;
- at least one dense model and one MoE model; the MoE primary topology should be
  TP2×EP2×ETP1×PP2×CP2 on 8×H100, followed by a larger workload where framework
  overhead does not dominate; and
- an A/B/B/A or randomized paired order, at least 5 warmup steps and 50 measured
  steps per arm, repeated at least three times.

Report:

- tokens/s and step-time mean, p50, p95, min, and max;
- peak allocated and reserved memory;
- forward, backward schedule, optimizer, D2H/H2D, and idle/CPU time;
- NCCL collective type, count, bytes, group, and overlap for AG/RS;
- production hit counters for bucket, overlap, prefetch, double buffer, and
  registered/user-buffer paths; fallback is reported as unimplemented;
- loss, grad norm, update success/overflow, and parameter delta against the
  independent reference from the same steps.

The replacement gate is intentionally strict:

- paired median tokens/s ratio `mfsdp_stage1 / dist_opt >= 1.00` on every
  mandatory workload;
- bootstrap 95% lower confidence bound at least `0.98` to bound measurement
  noise;
- no p95 step-time regression greater than 2%; and
- all correctness thresholds pass.

If the median is below 1.00, if the confidence interval is inconclusive, or if
an optimization silently falls back, M-FSDP stage 1 does not replace `dist_opt`.

### Stage-3 mandatory comparison

Repeat the same matched protocol for M-FSDP stage 3 vs FSDP2, adding:

- full fine-tuning and LoRA workloads;
- recompute on/off to expose backward parameter re-gathers;
- model/HF export and offload transitions outside timed steps; and
- NVIDIA M-FSDP as an independent correctness/performance reference where the
  same optimizer contract is possible.

The same non-inferiority threshold applies. Existing `1.225×` matched evidence
is a useful prior, not a waiver.

## Migration plan

### Phase 0 — integrate evidence branches

Rebase the functional and performance changes onto one source commit, resolve
their M-FSDP buffer/runtime differences, and rerun the existing CPU, 8-GPU
functional, precision, and three-arm suites. No default changes.

### Phase 1 — make backend construction generic

Introduce typed optimizer-algorithm and parameter-backend records, one shared
assembler, and generic lifecycle hooks. Remove the `dist_opt`-named boolean from
the train loop and the model-name argument from the optimizer primitive. Keep
each model responsible only for semantic unit types, expert classification, and
weight mapping.

### Phase 2 — add M-FSDP stage 1 as experimental

Implement the replicated-compute/sharded-main-and-grad design. Wire all five
models and both application connectors. Default remains `dist_opt`; users opt in
with `backend="mfsdp", zero_stage=1`.

### Phase 3 — close functional and performance gates

Finish checkpoint conversion, export, LoRA, offload, overflow, Muon, and the
matched benchmark matrix. A missing capability either fails before allocation
or keeps the old backend as default.

### Phase 4 — staged defaults with rollback

Change defaults separately:

1. stage-3-qualified configurations: `fsdp2` → `mfsdp, zero_stage=3`;
2. stage-1-qualified configurations: `dist_opt` → `mfsdp, zero_stage=1`.

Retain explicit `dist_opt` and `fsdp2` rollback values for at least one release.
Emit a clear resolved backend/stage/algorithm record in logs and checkpoints.
Do not rewrite an explicit user selection.

### Phase 5 — retirement

Remove an old backend only after its retirement gate, checkpoint migration
window, connector coverage, and rollback period complete. Remove dead branches,
exports, tests, and docs in the same change; tests alone do not keep production
code alive.

## Risk register

| Risk | Consequence | Mitigation/gate |
| --- | --- | --- |
| “ZeRO-1” means two different layouts | Incorrect communication and memory claims | Publicly document stage-1→`optim_grads`; record the resolved strategy |
| Functional/performance branches diverge | Old evidence does not describe merged code | Integrate first and rerun all evidence |
| Stale replicated stage-1 parameters | Next forward trains the wrong weights | Post-step replica-sync completion is a runtime precondition |
| Gradient double reduction | Wrong loss trajectory and grad norm | Independent unsharded/DistOpt update parity with TP/EP/PP/CP |
| Frozen parameters omitted in stage 3 | LoRA base model remains replicated | Separate compute sharding from optimizer ownership |
| Backend-specific checkpoint formats | Existing jobs cannot resume after default switch | Versioned schema, converter, and fail-loud compatibility matrix |
| Muon receives flat shards | Newton–Schulz computes the wrong update | Typed logical-shard record and backend lowering |
| Precision/offload fields are accepted but unused | Silent memory or numerical mismatch | Closed supported profiles and construction-time validation |
| Tiny benchmark overstates framework effects | False SOTA decision | Include production-sized dense and MoE workloads and paired repeats |
| Overlap/user-buffer fallback is counted as active | False performance attribution | Production hit counters; fallback is unimplemented |
| Primitive learns model names | Growing cross-product and layering debt | Model supplies semantic metadata; primitive remains model-agnostic |

## Final recommendation

Approve M-FSDP as the **target unifying parameter backend**, not yet as the
default optimizer backend. The immediate implementation order should be:

1. integrate the divergent stage-3 functional/performance branches;
2. introduce the generic capability/assembler boundary;
3. implement the DistOpt-compatible stage-1 buffer and lifecycle;
4. close checkpoint, five-model, connector, LoRA/frozen-param, offload, overflow,
   and Muon gaps; and
5. run the matched non-inferiority benchmarks.

Only measured SOTA or non-inferiority authorizes replacement. Functional parity
without the performance gate keeps `dist_opt`/FSDP2 as the defaults.

[mcore-ddp-strategy]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/distributed/distributed_data_parallel_config.py#L96-L108
[mcore-buffer-mapping]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py#L2005-L2030
[mlite-distopt-build]: ../megatron/lite/primitive/optimizers/megatron_wrap.py#L63-L195
[mlite-distopt-checkpoint]: ../megatron/lite/primitive/ckpt/distckpt.py#L25-L116
[mlite-fsdp2-wrap]: ../megatron/lite/primitive/optimizers/fsdp2/wrap.py#L117-L194
[mlite-fsdp2-build]: ../megatron/lite/primitive/optimizers/fsdp2/optimizer.py#L343-L474
[mlite-fsdp2-export]: ../megatron/lite/primitive/ckpt/hf_weights.py#L32-L46
[mlite-generic-dcp]: ../megatron/lite/primitive/ckpt/dcp.py#L34-L95
[mfsdp-config]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/config.py#L26-L46
[mfsdp-config-lowering]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/config.py#L169-L195
[mfsdp-fully-shard]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/fully_shard.py#L17-L35
[mfsdp-bucket]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/buffer.py#L247-L380
[mfsdp-wrapper]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/wrapper.py#L42-L105
[mfsdp-optimizer]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/optimizer.py#L235-L358
[mfsdp-grad-norm]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/optimizer.py#L101-L154
[mfsdp-grad-norm-helpers]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/grad_norm.py#L12-L83
[mfsdp-pipelines]: https://github.com/ISEEKYAN/Megatron-LM/blob/3997c3f15e32181c2acd11ea557d4c1cf5952ee6/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/buffer.py#L759-L872
[mfsdp-buffer-build]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/buffer.py#L675-L700
[mfsdp-backend]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/backend.py#L11-L92
[mfsdp-factory]: https://github.com/ISEEKYAN/Megatron-LM/blob/2a6356824aa36612db5764aa512883a6cd5c236f/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/fused_ops.py#L21-L34
[mfsdp-perf-commit]: https://github.com/ISEEKYAN/Megatron-LM/commit/3997c3f15e32181c2acd11ea557d4c1cf5952ee6
[deepseek-optimizer]: ../megatron/lite/model/deepseek_v4/lite/protocol.py#L407-L454
[glm-optimizer]: ../megatron/lite/model/glm5/lite/protocol.py#L252-L298
[kimi-optimizer]: ../megatron/lite/model/kimi_k2/lite/protocol.py#L212-L258
[qwen35-optimizer]: ../megatron/lite/model/qwen3_5/lite/protocol.py#L226-L272
[qwen3moe-optimizer]: ../megatron/lite/model/qwen3_moe/lite/protocol.py#L223-L292
[mlite-train-loop]: ../megatron/lite/primitive/train_step.py#L15-L74
[muon-distopt-config]: https://github.com/ISEEKYAN/Megatron-LM/blob/62404d4ab9802d2bd53f20f0ceeff1508bc6f72d/experimental/lite/megatron/lite/primitive/optimizers/megatron_wrap.py#L107-L242
[verl-runtime-calls]: ../examples/verl/verl_mlite/engine/mlite_engine.py#L316-L386
[verl-checkpoint]: ../examples/verl/verl_mlite/engine/mlite_engine.py#L421-L509
[miles-backend]: ../examples/miles/miles_mlite/arguments.py#L6-L48
[miles-runtime]: ../examples/miles/miles_mlite/backend_patch.py#L339-L469
[mlite-lora-export]: ../megatron/lite/model/qwen3_moe/lite/lora_adapter.py#L321-L420
[mcore-fsdp-muon]: https://github.com/NVIDIA/Megatron-LM/blob/c7d1aff65090fc04a50db67ddb83a51ea9615606/megatron/core/optimizer/emerging_optimizers.py#L306-L500
[mcore-tp-muon]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/emerging_optimizers.py#L402-L432
[mcore-layerwise-muon]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/__init__.py#L861-L966
[feature-composability]: https://github.com/ISEEKYAN/Megatron-LM/blob/ed2121463a410fa59aaa6f7c6cfd52f15bca916e/experimental/lite/docs/feature-composability.md

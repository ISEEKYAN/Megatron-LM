# M-FSDP `optim_grads` lowering: minimum implementation scope

## Decision summary

This is a source-level estimate for an *experimental*, additive M-FSDP
stage-1 profile.  It does not change a default, retire `dist_opt`, or claim
GPU validation.

The requested profile is MLite's DistOpt-compatible meaning of “ZeRO-1”:
M-FSDP's `optim_grads` strategy.  Compute parameters remain replicated in
modules; FP32 main parameters, optimizer state, and reduced gradients are
partitioned over the DP group.  At the end of an optimizer step, the updated
compute-precision parameter shards are all-gathered into those persistent
replicas.  It is not canonical ZeRO-1 (`optim`), and it is not the existing
M-FSDP `optim_grads_params` implementation.

The smallest credible deliverable is therefore **not** a config alias.  Its
critical path is a second buffer lifecycle: the existing stage-3 path replaces
module parameters with local shards and gathers before module execution,
whereas the stage-1 path must never expose those local shards to a module.

Recommendation: proceed only as a separately reviewable experimental backend
after integrating one qualified M-FSDP source line.  Budget **22–34
engineer-days** for that first deliverable, before GPU queue time.  Do not
authorize a `dist_opt` default change from this work.  A release candidate that
can replace `dist_opt` across its present support surface is a subsequent
**12–20 engineer-day** integration/compatibility increment plus the mandatory
GPU matrix and non-inferiority benchmark.

## Evidence boundary

The current baseline (`69ea18d07`) has `dist_opt` and FSDP2 but no M-FSDP
package.  The detailed source below is therefore inspected at functional
snapshot `2a6356824`; its performance changes are on a divergent snapshot
`3997c3f15`.  Source integration and rerunning the existing M-FSDP suites are
non-optional prerequisites, rather than a hidden assumption of the estimate.

The relevant independent behavior reference is MLite's current DistOpt path:
`primitive/optimizers/megatron_wrap.py` and
`primitive/ckpt/distckpt.py`.  Its governing invariant is: given the same
accumulated gradients, the partitioned update equals an unsharded update and
the partition is deterministic.  This follows
`primitive.optimizer.distopt`; M-FSDP adds the separate requirement that every
next forward observes the post-step replica refresh.

## Required design boundary

Keep optimizer algorithm separate from parameter-backend lowering.  The
resolved M-FSDP record owns the stage-to-strategy mapping:

```python
zero_stage=1  -> sharding_strategy="optim_grads"
zero_stage=3  -> sharding_strategy="optim_grads_params"
```

Users must not set both values independently.  Values 0 and 2, stage-3
prefetch/materialization knobs on stage 1, unsupported offload modes, and an
unsupported algorithm must fail before allocation.  The optimizer primitive
must receive semantic metadata (expert classification, groups, unit types),
not model names.  This preserves the `primitive.contract` ownership boundary
and avoids a model-by-backend cross-product.

## Non-optional prerequisite

| Change | Why it is needed | Estimate | Primary risk |
| --- | --- | ---: | --- |
| Integrate functional `2a6356824` and performance `3997c3f15` M-FSDP lines onto one source revision; rerun their CPU and existing M-FSDP validation suites | The baseline has no implementation to lower.  The two source lines have divergent checkpoint/export and communication behavior. | 4–7 days | Treating results from one line as evidence for the other; resolving conflicts without requalification. |

This prerequisite brings the existing M-FSDP package, its two currently wired
model protocols (Qwen3.5 and Qwen3-MoE), runtime hooks, and existing tests into
the target branch.  It is not a license to copy the source unchanged: the
stage-1 work below remains required.

## Minimum file and function change list

The paths and symbols in this table name the functional snapshot.  “Add” means
a new stage-1-specific method is preferable to a conditional that makes the
stage-3 lifecycle ambiguous.  This table is the implementation minimum for an
opt-in Adam/AdamW stage-1 profile, not the later default-replacement scope.

| File | Symbols to change | Minimum change | Risk that must be contained | Estimate |
| --- | --- | --- | --- | ---: |
| `primitive/optimizers/mfsdp/config.py` | `MFSDPConfig`; `build_mfsdp_config`; `validate_mfsdp_config`; `validate_optimization_knobs` | Add typed `zero_stage`, derive exactly `optim_grads` or `optim_grads_params`, and reject unavailable stage-specific knobs before buffer allocation.  Keep the existing dtype/process-group lowering. | A user selects a contradictory strategy, or accepts a stage-3 knob which has no production effect in stage 1. | 1–2 d |
| `primitive/optimizers/mfsdp/fully_shard.py` | `fully_shard_model` | Replace the hard rejection of every strategy except `optim_grads_params` with the two supported profile dispatch.  `optim_grads` must construct the same wrapper API but select the replica-preserving lifecycle. | Accidentally calling `install_sharded_parameters` in stage 1 makes module math run on one-dimensional local shards. | 0.5–1 d |
| `primitive/optimizers/mfsdp/buffer.py` | `ParamSpec`; `ParamBucket.__init__`; `_initialize_parameters`; `install_sharded_parameters`; `install_full_parameters` | Split “authoritative local main shard” from “replicated module compute parameter.”  For stage 1, retain the original full `nn.Parameter` binding permanently, create only main-shard optimizer parameters, and continue to register the full compute gradient hook. | Aliasing the optimizer parameter to the compute replica, or losing tied/duplicate parameter bindings. | 3–4 d |
| `primitive/optimizers/mfsdp/buffer.py` | `prepare_param_gather`; `wait_param_gather`; `release_full_parameters`; `discard_full_parameter_views`; `materialize_main_parameters`; `copy_full_parameters_to_shards` | Keep the existing temporary materialization methods exclusive to stage 3.  Add a persistent-replica refresh path: cast each local updated main shard, all-gather it, copy the result into the already-bound compute parameters, and expose completion as a precondition of the next forward.  Retain a full-FP32 lease only for checkpoint/export consumers. | A stale replica reaches the next forward; a compute-dtype cast or padding offset is wrong; a checkpoint/external consumer sees a temporary buffer instead of authoritative main state. | 3–5 d |
| `primitive/optimizers/mfsdp/buffer.py` | `prepare_grad_reduce`; `wait_grad_reduce`; `reset_grad_state`; `_make_grad_ready_hook`; `ParamAndGradBuffer._build` | Reuse reduce-scatter over full accumulated compute gradients, write its local result into the main-gradient shard, average exactly once, then clear the replica's `.grad`.  Preserve last-microbatch enablement.  Bucket all `requires_grad` parameters only; ensure frozen parameters stay valid replicated compute parameters outside optimizer ownership. | Double DP reduction/averaging, missed gradient accumulation, TP/EP ownership error, or a frozen parameter disappearing from forward. | 2–3 d |
| `primitive/optimizers/mfsdp/buffer.py` | `AllGatherPipeline`; `CommunicationPipelines.begin_forward`, `end_forward`, `pack_saved_tensor`, `unpack_saved_tensor` | Make forward/backward gather, forward-owner hooks, saved-tensor rematerialization, and full-view discarding stage-3-only.  Add a post-step refresh pipeline with an explicit wait before forward. | Retaining a non-executing “prefetch” branch, or ordering a refresh collective ahead of unfinished reduce-scatter. | 2–3 d |
| `primitive/optimizers/mfsdp/wrapper.py` | `MegatronFSDP.__init__`; `forward`; `start_param_sync`; `full_parameter_context`; `state_dict`; `load_state_dict` | In stage 1 do not register owner pre-hooks, saved-tensor hooks, or `_BeginBackward` materialization work; forward invokes the module with its persistent replicas.  Keep a full-main-parameter context for exact checkpoint/export. | Stage-3 hooks remain reachable but inert, creating incorrect or unmeasured collectives; loading restores masters but not replicas. | 2–3 d |
| `primitive/optimizers/mfsdp/optimizer.py` | `MFSdpOptimizer.zero_grad`, `finish_grad_sync`, `step`; `build_mfsdp_stack`; `build_mfsdp_training_optimizer` | After local optimizer step, invoke and wait for stage-1 replica refresh before permitting the next forward.  Keep `grad_sync_enabled` false until the final microbatch and reset it only after a completed update/refresh transaction. | Optimizer reports success before replicas are current; overflow/failed step still refreshes stale or uninitialized shards. | 2–3 d |
| `primitive/optimizers/mfsdp/optimizer.py` | `_StandaloneOptimizer.step`, `clip_grad_norm`, `_sync_tp_replicated_grads_once`, `_scale_expert_grads_once` | Establish independent parity with DistOpt for TP-replicated and expert gradients, norm/clipping, NaN/Inf skip, and the optimizer return tuple.  Do not infer correctness from reusing the stage-3 implementation. | Duplicate or missing TP/EP/PP contribution causes a plausible but wrong loss curve. | 2–3 d |
| `primitive/optimizers/mfsdp/backend.py` | `_STATE_FORMAT`; `state_dict`; `load_state_dict`; `sync_model_weights_to_main_weights` | Version the state with resolved stage and layout.  On stage-1 restore, copy masters then refresh compute replicas; reject incompatible stage/layout rather than guessing. | A local sidecar silently resumes under a different DP layout or restores only one of master/replica state. | 1–2 d |
| `primitive/train_step.py` | `run_microbatch_loop`; `compute_and_clip_grad_norm` | Replace the `dist_opt`-named boolean with a backend capability/hook (“delay gradient synchronization” and “finish shard gradient sync”).  All call sites must pass the resolved backend behavior. | A model-specific boolean leaks into a primitive, or PP/non-PP paths choose different gradient-finalization rules. | 1–2 d |
| `model/qwen3_5/lite/protocol.py` | optimizer selection and `_post_model_load_hook` | Thread an explicit M-FSDP stage record to the existing post-load construction and log the resolved backend/stage.  No model name is passed to the primitive. | The hook builds before weights are loaded, or a stage is accepted but not recorded in checkpoint/log context. | 0.5–1 d |
| `model/qwen3_moe/lite/protocol.py` | optimizer selection and `_post_model_load_hook` | Same as Qwen3.5 while preserving its expert classifier and LoRA/frozen-parameter behavior. | Dense and expert DP groups are confused; frozen LoRA base weights are incorrectly made optimizer shards. | 0.5–1 d |
| `runtime/backends/mlite/runtime.py`; `runtime/megatron_utils.py` | backend resolution, training-hook registration, state/offload lifecycle callers | Route generic backend hooks, checkpoint synchronization, and failure propagation through the resolved capability rather than a `dist_opt` spelling. | A connector bypasses the adapter and persists an incomplete M-FSDP state. | 1–2 d |

The stage-1 mechanism totals **17–29 engineer-days** in the table because
several rows are partially parallel once the lifecycle contract is frozen.  A
single reviewable implementation sequence, including the integration
prerequisite and tests below, is more realistically **22–34 engineer-days**.

## Tests and verification files

| File | Minimum additions | Purpose |
| --- | --- | --- |
| `tests/unit/primitive/test_mfsdp.py` | Config mapping/rejections; stage-1 parameters remain full-shaped and permanently bound; deterministic uneven shard offsets; stage-1 refresh changes the next forward replica; 1- and 2-rank CPU/Gloo update parity where supported. | Catch the lifecycle distinction without GPU. |
| `tests/smoke/primitive/test_mfsdp_parity_smoke.py` | Small CUDA parity against unsharded and DistOpt for Adam/AdamW: accumulated microbatches, dense + MoE/expert groups, grad norm, clipping, NaN/Inf skip, checkpoint/resume, and the post-step refresh assertion. | Verify the distributed invariant before application wiring. |
| `tests/run_mfsdp_hopper_validation.sh` | An explicit stage-1 invocation and results collection; do not reuse a stage-3 recipe under a new label. | Preserve a reproducible GPU validation surface. |
| `tests/unit/runtime/test_runtime_backend_unit.py`; `tests/unit/primitive/test_checkpoint_runtime.py` | Resolved capability dispatch and stage/layout checkpoint rejection/restoration tests. | Ensure runtime/restore cannot silently bypass the lowering. |
| New config-only test at the production entry point | Exercise full connector initialization, environment inheritance, configuration parsing, backend resolution, and the post-load hook without allocating a GPU. | Required zero-GPU gate; an isolated config parser is insufficient. |

No GPU command is run for this estimate.  The implementation's GPU gate must
use Slurm: first a small same-recipe proxy with scaled TP/EP/CP where possible,
then the matched `dist_opt` comparison.  A non-scaling communication group must
be recorded as a deliberate exception, not silently skipped.

## Scope deliberately excluded from the first experimental profile

| Excluded capability | Why it is not a free follow-on | Later estimate |
| --- | --- | ---: |
| Default replacement for `dist_opt` | Requires every existing model/application surface, migration/rollback, correctness, and the performance gate. | See next section |
| DeepSeek V4, GLM-5, and Kimi K2 M-FSDP wiring | Their protocols presently construct only `dist_opt`/FSDP2.  Duplicating model-specific optimizer branches would violate primitive layering; use a shared assembler/capability hand-off first. | 3–5 d |
| VERL and miles configuration/checkpoint support | VERL currently calls checkpoint primitives directly; a runtime adapter must become the single state owner. | 3–7 d |
| Cross-topology or cross-backend optimizer resume | The current M-FSDP state is a versioned local-sidecar format, unlike DistOpt's reshardeable optimizer format. | 5–8 d |
| Fractional update-state offload and precision-aware optimizer | Existing M-FSDP supports only bounded profiles; accepting a field without implementing it is a silent-contract bug. | 4–7 d |
| Muon | Its matrix operation cannot consume arbitrary flat main-parameter shards.  It requires logical-shard metadata and a separate lowering. | 7–12 d |
| Removing `dist_opt` or dead compatibility branches | Removal is permitted only after the retirement gate and rollback window.  Tests do not make a production branch live. | Separate cleanup change |

## Default-replacement gate and decision impact

The first experimental implementation is sufficient to decide whether the
lowering can be made correct.  It is insufficient to decide whether it should
replace DistOpt.  Keep `dist_opt` as the default until all of the following are
true:

1. Adam/AdamW update, gradient norm, clipping, overflow, accumulation, and
   deterministic shard ownership match independent unsharded/DistOpt references.
2. All five model protocols and both application connectors use the same
   model-agnostic assembler/capability boundary.
3. Checkpoint/resume, export/update-weights, supported offload, and an explicit
   rollback value pass; unsupported combinations fail before allocation.
4. Muon either passes its own lowering contract or fails loudly before any
   allocation—never falls back to Adam.
5. A matched Slurm benchmark uses the same source, container, model, seed,
   topology, batch/sequence/microbatching, optimizer implementation, precision,
   clipping, and offload policy.  Its mandatory workloads meet the already
   proposed non-inferiority criteria: median throughput ratio at least 1.00,
   bootstrap 95% lower bound at least 0.98, and no more than 2% p95 step-time
   regression.

The source study shows equal first-order collective volume for DistOpt and this
stage-1 design: one gradient reduce-scatter and one post-step parameter
all-gather.  M-FSDP can therefore win only through a measured implementation
detail (bucketing, overlap, cast/copy, or framework overhead), not by the
strategy name.  A correct implementation that misses the performance gate is
an opt-in backend, not a replacement.

## MLite skill compliance checklist

This estimate applies the following implementation contracts:

- `basic.constitution`: prefer the smallest reviewable design, use the MCore
  DistOpt/unsharded update as the checkable reference, and require an end-to-end
  path before delivery.
- `primitive.contract` and `primitive.optimizer.distopt`: define the exact
  state/config/process-group/update ownership before code; retain deterministic
  sharding and unsharded-update equivalence.
- `primitive.optimizer.fsdp`: keep model modules responsible only for exposing
  parameters; let the parameter backend own sharding, materialization, state,
  and offload.
- `model_compose.config_mapping`: normalize one resolved stage/backend record
  before allocation; do not accept aliases with divergent meaning.

The corresponding implementation review must reject dead stage-3-only
branches reachable from stage 1 and any optimizer primitive that imports a
model-name allowlist.

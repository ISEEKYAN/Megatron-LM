# Hopper Blockwise FP8 Contract and Parity Protocol

This document is the normative contract for the first Megatron Lite FP8
training implementation. It freezes behavior and validation before production
code is added. It does not claim that the profiles described here are already
implemented.

The first release is deliberately closed. It has two named Hopper blockwise
profiles, one fixed compute coverage set, and no user-composable recipe policy.
Any extension requires a separate contract and evidence.

## Scope

The only new public configuration values are:

- `hopper_blockwise_bf16_weight`
- `hopper_blockwise_fp8_weight`

Implementation status: `hopper_blockwise_bf16_weight` is implemented. The
`hopper_blockwise_fp8_weight` profile (FP8 parameter storage) is the closed
design's declared-but-pending second profile: resolving it fails loud with a
`NotImplementedError` rather than being advertised as usable, and FP8-weight
initialization is tracked as a separate follow-up. The rest of this document
describes both profiles as the frozen design contract.

`bf16` remains the default and preserves the current behavior. No public
`recipe`, `format`, `target`, `weight_dtype`, rule list, ordered matcher, or
model-specific FP8 configuration is part of this release.

Both Hopper profiles apply blockwise FP8 compute to every covered instance of:

- attention projection GEMMs, including QKV and output projections;
- dense MLP GEMMs; and
- routed and shared MoE expert GEMMs.

The attention core, router, normalization, embedding, vocabulary/LM head,
residual paths, losses, and public module boundaries stay BF16. In particular,
attention projections and the attention core are separate semantic sites.
Transformer Engine does not support FP8 DPA or MHA with
`Float8BlockScaling`.

MXFP8, delayed/tensorwise scaling, FP8 attention core, FP8 parameter
all-gather, lower-precision optimizer state, FP8 communication, arbitrary
target combinations, model allowlists, and model-local FP8 implementations are
out of scope.

## Independent Reference

Parity uses direct upstream source, not Megatron Lite through a second runtime
path:

- Megatron-Core commit
  [`cf2f07d7b1315c96c05554c670c43207c6783e5e`][mcore-commit];
- the canonical training image's released Transformer Engine `2.15.0`, whose
  blockwise kernels (Linear, LayerNormLinear, GroupedLinear) run forward and
  backward under `Float8BlockScaling` -- verified by a read-only capability
  inventory on that image (SM90, CUDA 13.2, cuBLAS 13.4 present, block scaling
  supported, all mandatory ops ran); and
- an NVIDIA GPU on which TE's blockwise-FP8 capability probe passes.

Blockwise FP8 runs on the same image that runs BF16 -- there is no FP8-only
build overlay. The runtime gate is deliberately *not* a hard pin on an exact
device class, TE version, or CUDA/cuBLAS threshold: the single authoritative
gate is TE's own `check_fp8_block_scaling_support()` capability probe. Pinning
exact versions would recreate a special-environment requirement (whatever runs
BF16 must run FP8) and could reject a newer or different accelerator where BF16
is fine. The released TE `2.15.0` is the version the parity evidence was
recorded against; it is kept as provenance, and a probed toolchain that differs
from it emits a fail-loud provenance warning but is not blocked. The `[te-*]`
source citations point at readable upstream source (pinned to the `v2.15` tag)
for the blockwise recipe, quantization, and grouped-GEMM APIs.

The reference driver must import Megatron-Core from the frozen checkout and
Transformer Engine from the canonical image. It records the Megatron-Core
revision and the TE version for provenance and asserts TE's capability probe
before allocating model state.

### Environment seal

Before any target result is produced, the reference owner must write and seal
an environment manifest containing:

- immutable container path and content digest;
- GPU name, UUIDs, compute capability, and driver version;
- CUDA build and runtime versions, cuBLAS build and runtime versions, cuDNN,
  NCCL, PyTorch, and Python versions;
- the full Megatron-Core and Transformer Engine commit IDs;
- installed TE distribution versions and the TE shared-object hashes;
- all `NVTE_*`, determinism, and allocator environment variables; and
- the Slurm cluster, partition, node list, and job ID used for qualification.

Reference calibration and Megatron Lite comparison jobs must use the same
sealed container digest and hardware class. A mutable tag, an unrecorded
overlay, a different TE wheel, or a different CUDA/cuBLAS build invalidates the
comparison.

The image is qualified only if all of the following preflight checks pass:

1. TE's `check_fp8_block_scaling_support()` succeeds. This capability probe is
   the single authoritative gate -- the same one the runtime uses -- and it
   determines device/toolchain support (SM, CUDA, cuBLAS, block scaling)
   directly rather than through hard-coded version thresholds.
2. The device class, CUDA, and TE versions are recorded in the environment
   manifest for provenance and diagnosis, not enforced as independent hard
   gates: whatever image runs BF16 must run FP8, so an exact SM class, TE
   version, or general CUDA/cuBLAS threshold is deliberately not pinned. The
   canonical qualification image reported SM90, CUDA 13.2, cuBLAS 13.4, and TE
   2.15.0. The one exception is scoped and TE-owned: a profile selecting MoE
   experts additionally gates the grouped-GEMM cuBLAS requirement, because TE
   2.15's blockwise `GroupedLinear` path requires cuBLAS 13.3+
   (`CUBLAS_GROUPED_GEMM_VERSION`) and `check_fp8_block_scaling_support()` does
   not cover it. The runtime fails loud on an under-versioned cuBLAS
   (`CUBLAS_GROUPED_GEMM_MIN_VERSION`, encoded like `get_cublasLt_version()` as
   `130300`) rather than crashing inside the grouped GEMM. This is not a
   reintroduced device/CUDA pin: it is the grouped path's own hard requirement,
   scoped to MoE profiles, and the canonical image (cuBLAS 130401) satisfies it.
3. `NVTE_FP8_BLOCK_SCALING_FP32_SCALES` is absent or `0`,
   `NVTE_BACKWARD_OVERRIDE` is absent, and the constructed recipe equals the
   frozen recipe below.
4. The reference linear and grouped-linear probes execute, rather than skip or
   fall back to a non-blockwise kernel.

There is no pre-qualified image assumed by this document. The first image that
passes these checks is bound by its digest in the environment manifest before
reference calibration. If no available image passes, implementation parity is
blocked; dropping MoE, changing TE, or accepting a fallback is not a valid
substitute.

## Primitive Principle and Invariants

The primitive changes internal GEMM quantization and, in one profile, compute
weight storage. It must not change model math, topology, tensor shapes, public
dtypes, process-group membership, parameter ownership, optimizer math, or
checkpoint continuity.

The following invariants apply to both profiles:

1. Inputs and outputs at primitive, residual, pipeline, loss, and checkpoint
   API boundaries keep their existing shapes and BF16 dtype.
2. Forward output, input gradient, weight gradient, and the FP32-master
   optimizer update match the frozen reference under the gates below.
3. Every selected semantic site is covered exactly once by a compatible TE
   primitive. Missing, duplicate, or incompatible coverage fails during model
   construction.
4. Every fixed BF16 site remains outside the FP8 context. It is an error to
   make attention core, router, norm, embedding, or LM head FP8 as a side
   effect of a broad context.
5. Coverage and selection use typed semantic declarations. Module-name globs,
   ordered matchers, model names, model allowlists, and class-name heuristics
   are forbidden.
6. The precision primitive owns no model configuration, model registry,
   optimizer state, checkpoint mapping, or process-group construction.
7. FP8 compute precision, compute-parameter storage, parameter communication,
   optimizer master precision, gradients, and optimizer states remain
   separate fields with one explicit owner each.
8. Unsupported hardware, shapes, modules, parallel combinations, or state
   transitions fail loudly before training. Silent BF16 fallback is forbidden.

## Closed Profile Contract

The recipe is constructed with these exact values:

```text
Float8BlockScaling(
  fp8_format=Format.E4M3,
  x_block_scaling_dim=1,
  w_block_scaling_dim=2,
  grad_block_scaling_dim=1,
  fp8_gemm_fprop.use_split_accumulator=True,
  fp8_gemm_dgrad.use_split_accumulator=True,
  fp8_gemm_wgrad.use_split_accumulator=True,
  fp8_dpa=False,
  fp8_mha=False,
  backward_override=None,
)
```

Scales are FP32 containers constrained to powers of two. Activations and
output gradients use 1x128 blocks; weights use 128x128 blocks. Rowwise and
columnwise 1D representations must both be quantized from the high-precision
source. No amax history or amax-reduction process group exists for this
blockwise recipe.

| Contract field | `hopper_blockwise_bf16_weight` | `hopper_blockwise_fp8_weight` |
| --- | --- | --- |
| GEMM compute | frozen blockwise E4M3 recipe | frozen blockwise E4M3 recipe |
| selected compute weights | BF16 parameters, quantized for GEMM | TE blockwise FP8 compute parameters |
| public inputs/outputs | BF16 | BF16 |
| non-GEMM parameters and bias | BF16 | BF16 |
| authoritative initialization/load source | BF16/FP32 source tensor | the same high-precision source, never reconstructed from FP8 |
| optimizer master parameters | FP32 | FP32 |
| accumulated/main gradients | FP32 | FP32 |
| Adam moments and step | FP32 moments, integral step | FP32 moments, integral step |
| parameter all-gather | BF16; no FP8 parameter gather | BF16; no FP8 parameter gather |
| recipe checkpoint state | stateless | stateless |
| parameter checkpoint state | BF16 compute weight | FP8 data and block scales |

For the FP8-weight profile, model construction uses TE
`quantized_model_init` (the frozen revision's `fp8_model_init` is only a
deprecated forwarding alias) with the frozen recipe and
`preserve_high_precision_init_val=True`. The same high-precision tensor must
initialize the TE compute parameter and the single FP32 optimizer master.
After master ownership is established, the temporary high-precision init value
is cleared. An FP32 master reconstructed by dequantizing the FP8 parameter is a
contract violation.

DistOpt and FSDP2 consume the same explicit parameter contract. Neither backend
may infer ownership from tensor class or profile-name string. Exactly one
component owns and updates each FP32 master. TE must not retain a competing
optimizer master when MLite owns it.

## Public and Internal API Boundary

The public runtime field is a string with exactly these three closed values:

```python
MegatronLiteConfig(precision="bf16")
MegatronLiteConfig(precision="hopper_blockwise_bf16_weight")
MegatronLiteConfig(precision="hopper_blockwise_fp8_weight")
```

Internally, the precision package may share only the following narrow typed
data:

- `PrecisionImplementation`: immutable profile identity, the one frozen recipe
  factory, the fixed covered semantic sites, and its `ParameterContract`;
- `ParameterContract`: compute-weight storage, authoritative load source,
  master/gradient/state dtypes, all-gather dtype, and ownership; and
- typed coverage requirements and claims for `ATTENTION_PROJECTION`,
  `DENSE_MLP`, and `MOE_EXPERT`.

These are closed records for the two profiles, not a base class for arbitrary
recipes or a generic policy language. There is no public constructor that can
mix recipe, format, targets, and weight mode.

### Coverage protocol

Coverage is per concrete semantic site, not merely “at least one module of this
kind”. Model composition emits typed site requirements; reusable primitives
emit typed capability claims for the exact sites they own. Runtime binding
matches object identity plus the semantic enum, injects one
`PrecisionImplementation`, and seals the manifest before optimizer creation.

The manifest must prove:

- each present attention projection, dense MLP GEMM, and expert GEMM has one
  compatible claim;
- every requirement is matched exactly once and there are no unconsumed claims;
- attention core, router, norms, embeddings, and LM head are explicit BF16
  exclusions;
- selected modules are TE Linear, LayerNormLinear, or GroupedLinear paths that
  support the frozen recipe; and
- inline `nn.Linear`, torch matmul/SDPA, local attention, LoRA side paths, and
  any other non-TE implementation cannot satisfy a selected requirement.

Paths and module names may be recorded only as diagnostic evidence after typed
binding. They must never decide selection or eligibility. A model becomes
eligible by composing fully covered primitives, not by appearing in a support
list.

## Owned Files and Replacement Boundary

The precision primitive owns only:

```text
experimental/lite/megatron/lite/primitive/precision/
  __init__.py
  contract.py
  coverage.py
  hopper_blockwise.py
```

`contract.py` owns the closed records and semantic enums. `coverage.py` owns
typed requirement/claim matching and diagnostics. `hopper_blockwise.py` owns
the two profile instances, frozen recipe creation, environment validation, and
model-init/forward contexts. `__init__.py` exposes the three-name resolver and
the narrow records.

The following files remain consumers and retain their existing layer
responsibilities:

- `runtime/backends/mlite/config.py`: the public `precision` string;
- `runtime/backends/mlite/runtime.py`: resolve, preflight, construction context,
  typed injection, and coverage sealing;
- `primitive/parallel/linear.py`, `primitive/modules/gqa.py`,
  `primitive/modules/mlp.py`, and `primitive/modules/experts.py`: capability
  declarations and correctly scoped forward contexts;
- `primitive/optimizers/megatron_wrap.py` and `primitive/optimizers/fsdp2/`:
  consume `ParameterContract` and own FP32 master/update state;
- `primitive/ckpt/distckpt.py` and `primitive/ckpt/dcp.py`: save and restore
  model/optimizer state without learning profile policy; and
- model protocol compose sites: emit typed requirements and remove old
  model-threaded FP8 switches only after their primitive coverage is complete.

No `model/*/fp8.py`, FP8 model subclass, recipe-specific model registry entry,
generic `policy.py`/`base.py`, or glob matcher may be added.

Existing whole-model `fp8_autocast` branches and the delayed-scaling
`build_fp8_recipe()` helper are not a second supported path. They are currently
disabled by production protocols. When the first production composition moves
to this contract, those unreachable branches and any exports used only by them
must be removed rather than kept alive by tests. Other model families remain
BF16 and fail typed coverage until separately migrated; they do not receive
copied FP8 branches.

## State, Device, and Process Groups

The recipe object is immutable and reused. `Float8BlockScaling` is stateless at
the frozen TE revision, so it has no delayed-scaling amax history to synchronize
or checkpoint. TE-owned ephemeral rowwise/columnwise quantized activations are
not MLite checkpoint state.

Process-group ownership does not change:

- column/row parallel linears continue to use their existing TP group;
- expert dispatch continues to use EP, expert computation continues to use ETP
  where configured, and expert-data-parallel ownership stays with the optimizer;
- router padding follows the frozen Megatron rule and pads the routing map to
  the blockwise FP8 alignment of 16;
- blockwise scales are local and do not introduce an amax collective; and
- DP/FSDP groups remain optimizer/runtime state, not precision-primitive state.

Ordinary blockwise linear operands must satisfy TE's shape rules: rank at least
two, last dimension divisible by 128, and the product of preceding dimensions
divisible by 128. Grouped expert GEMMs keep their K/N dimensions block-aligned
and follow Megatron's separate routing-map alignment of 16 for the routed M
dimension. The frozen TE GroupedLinear capability check is authoritative for
that path; MLite must not invent a different padding rule.

## Legal Combinations and Failure Modes

The initial validated topology has TP and EP/ETP, with `pp=1` and `cp=1`.
Pipeline and context parallel compositions remain unavailable for these two
profiles until separately validated. BF16 retains its existing topology
support.

The implementation must reject during config parsing, environment preflight,
model construction, or checkpoint load as indicated below.

| Condition | Required failure |
| --- | --- |
| unknown profile or any ad-hoc recipe/format/target value | config error listing the accepted profile names |
| `hopper_blockwise_fp8_weight` (FP8 parameter storage) is requested | construct selected TE GEMMs under `quantized_model_init`; preserve the high-precision source for the single MLite FP32 master |
| TE's `check_fp8_block_scaling_support()` reports no blockwise FP8 support | environment error before model allocation, quoting TE's reason and the recorded toolchain (device, TE, CUDA, cuBLAS) |
| MoE-expert profile on cuBLAS below the grouped-GEMM requirement (`< CUBLAS_GROUPED_GEMM_MIN_VERSION`) | environment error before model allocation; the grouped path is gated instead of crashing inside `GroupedLinear` |
| recipe differs, FP8 DPA/MHA is enabled, or scale/backward env overrides are set | environment/recipe mismatch error |
| selected shape violates blockwise rules | construction error with semantic site and shape |
| missing, duplicate, or incompatible typed coverage | construction error before optimizer creation |
| selected site is `nn.Linear`, torch matmul/SDPA, or an unsupported TE path | construction error; never BF16 fallback |
| attention core, router, norm, embed, or LM head is selected | construction error identifying the fixed BF16 boundary |
| LoRA or another side path changes a selected GEMM without reviewed coverage | construction error |
| `pp>1`, `cp>1`, CUDA graph, or another unvalidated composition is requested | unsupported-combination error |
| `fp8_param_gather`, FP8 communication, MXFP8, or lower-precision optimizer state is requested | unsupported-combination error |
| FP8-weight profile cannot establish one FP32 master from the high-precision source | construction error; no dequantize-and-continue |
| checkpoint lacks FP8 data/scales, FP32 master, moments, step, RNG, or contract identity | checkpoint compatibility error |
| checkpoint profile/contract differs from the requested profile | checkpoint compatibility error |

If the frozen TE/optimizer revisions cannot legally implement a mandatory
DistOpt or FSDP2 cell, that cell is blocked with source and runtime evidence.
The implementation must not relax ownership, change precision, or drop a
backend to make the matrix green.

## Controlled Parity Variables

Reference and target consume the same serialized input and initialization
artifacts. The artifact hashes, not matching generator code, define equality.
Reference and target drivers must not share MLite implementation helpers.

Unless a cell below explicitly changes one variable, freeze:

- seed `1234`, three exact repeats, dropout `0`, deterministic algorithms on,
  and `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- BF16 public activations, E4M3 blockwise recipe above, no TF32, no autocast
  outside the selected contexts, no activation recompute, no CUDA graph, and no
  offload;
- hidden size `1024`, sequence length `256`, micro-batch size `2`, attention
  heads `16`, KV heads `8`, head dimension `64`, and dense/intermediate size
  `4096`;
- MoE experts `4`, top-k `2`, no expert bias, routing-map padding to `16`, and
  expert intermediate size `4096`;
- AdamW with learning rate `1e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight
  decay `0.1`, constant learning rate, no warmup, FP32 master/main gradients and
  FP32 moments; and
- identical loss reduction, gradient accumulation count, parameter order,
  process-group rank order, initialization/load tensors, labels, and optimizer
  step number.

The BF16 baseline must align before either FP8 profile is evaluated. Each FP8
profile is compared to its matching direct Megatron reference; the two profiles
are not used as references for each other.

## Precision Gate Seal

Thresholds are derived only from Megatron reference repeats. Target output must
not exist when the threshold artifact is created.

The formula, repeat count, multiplier, exact-comparison cases, and review rule
in this section are the pre-implementation threshold freeze. Numeric values are
filled only by the later reference-only calibration; they are not guessed in
advance and cannot depend on a target run.

For every tensor or scalar metric `x`, run the identical reference cell three
times and compute over all reference pairs:

```text
noise_abs(x) = max(max(abs(reference_i(x) - reference_j(x))))
noise_l2(x)  = max(
  norm(reference_i(x) - reference_j(x), 2)
  / max(norm(reference_i(x), 2), norm(reference_j(x), 2), tiny_fp32)
)
```

The sealed comparison gate is:

```text
atol(x) = 4 * noise_abs(x)
rtol_l2(x) = 4 * noise_l2(x)
```

There is no target-derived floor and no default “1% FP8 tolerance”. If both
reference noise terms are zero, comparison is bitwise. Otherwise the generated
non-bitwise gate requires review, as required by `basic.review_threshold`, and
the approved `thresholds.json` content hash must be recorded before target
runs. Excessively noisy or non-finite reference results block the cell; they do
not authorize a wider gate.

A target tensor passes only when shape and dtype are exact, all values are
finite, max-absolute error is at most `atol`, relative L2 error is at most
`rtol_l2`, and the target's three-repeat noise is no greater than the reference
noise gate. Loss and grad norm use the same rule. Coverage, profile identity,
parameter dtype/ownership, checkpoint keys, and failure behavior are exact
comparisons rather than tolerant metrics.

Save/load continuity is stricter: within one implementation, uninterrupted and
save-load-resume executions from the same state must be bitwise equal for the
next-step public outputs, restored FP32 optimizer state, and updated
high-precision weights. A source-backed exception must be reviewed before the
run, never after seeing the result.

Once sealed, a gate can only be replaced by rerunning reference-only
calibration in a newly sealed environment manifest. The old and new artifacts
must both remain in evidence. A failing target run cannot trigger recalibration.

## Validation Matrix

All GPU cells run through Slurm. A result is valid only with a real job ID,
`sacct` state `COMPLETED`, exit code `0:0`, and explicit proof that no test or
kernel path skipped.

### Phase 0: CPU and static contract

- Parse and round-trip all three public names, including the FP8-weight
  profile's closed parameter contract.
- Assert the exact recipe mapping and the `hopper_blockwise_bf16_weight`
  `ParameterContract` record with mocked TE capability results.
- Reject every unsupported combination in the failure table.
- Exercise missing, duplicate, incompatible, and unconsumed typed coverage.
- Assert the precision package imports no model package and contains no model
  names, glob matcher, ordered rule, or generic recipe policy.
- Prove production reachability for every new function/export; test-only callers
  do not count.

### Phase 1: direct primitive reference

1. A single TE linear on one H100, then TP=2, with input shape
   `[256, 2, 1024]` and output size `4096`.
2. A one-layer attention split with TP=2. Compare QKV and output projection
   sites under the profile while proving DotProductAttention and Q/K norm sites
   remain BF16.
3. A reduced MoE with four experts, top-k two, TP=2 and EP=2. Prove routing-map
   padding, GroupedLinear blockwise kernel use, BF16 router/dispatch, and expert
   dX/dW.

For both profiles, collect public forward output, dX, every selected dW, every
fixed-BF16 boundary, and one optimizer update from identical FP32 masters.

### Phase 2: optimizer and checkpoint ownership

Run the same one-step case with DP/FSDP size two for:

- Megatron-Core DistributedOptimizer versus MLite DistOpt; and
- the direct Megatron math/update reference versus MLite FSDP2.

Materialize global values before comparison. Record compute-parameter dtype and
storage class, FP32 master owner and shard, main-gradient dtype, both Adam
moments, optimizer step, and updated high-precision weight.

For each backend and profile, compare a two-step uninterrupted run against
step-one save/load plus step two. The FP8-weight checkpoint must contain TE FP8
data and block scales as well as the FP32 master and optimizer state. The
blockwise recipe itself has no amax history to save.

### Phase 3: Qwen3 MoE composition

Use a two-layer reduced Qwen3 MoE composition with the frozen dimensions and
TP=2, EP=2, DP=2 on one eight-H100 node. Run:

1. BF16 baseline;
2. `hopper_blockwise_bf16_weight`; and
3. `hopper_blockwise_fp8_weight`.

For DistOpt and FSDP2, compare loss, logits, grad norm, selected parameter
gradients, fixed-BF16 boundaries, updated high-precision weights, and a
checkpoint-resumed next step. The production runtime and model protocol must be
used; isolated constructors and otherwise unreachable FP8 branches do not
count.

### Phase 4: performance after correctness

Performance is measured only after every required correctness cell passes.
Within the same allocation, use ten warmup steps and thirty measured steps per
profile, repeat the paired order at least three times, and report step time,
tokens per second, and peak allocated/reserved memory together with loss.
Report both Hopper profiles relative to BF16 and the matching Megatron
reference. There is no performance pass threshold in this contract, and no
precision or coverage rule may be changed to improve a number.

## Evidence and Delivery Gate

Each cell leaves:

- sealed environment and case manifests with content hashes;
- the three raw reference repeats and reference-only threshold artifact;
- the three raw target repeats;
- tensor-level comparison reports for forward, dX, dW, update, and resume;
- typed coverage and parameter-ownership manifests;
- checkpoint inventory and uninterrupted-versus-resume report;
- Slurm job script, stdout/stderr, job ID, and `sacct` result; and
- proof of actual blockwise Linear/GroupedLinear execution and zero skips.

Delivery also requires the relevant MLite skill invariants: the smallest
replaceable primitive (`basic.constitution`), explicit math/dtype/update gates
(`primitive.principle`), owned API/state/groups/failures
(`primitive.contract`), checkable reference and replacement boundary
(`primitive.design`), and static/proxy/composition/runtime validation
(`primitive.validate`). The final review must separately inspect production
dead code and primitive-to-model layering; a green test suite alone is not
sufficient.

## Frozen Source Anchors

- Megatron maps `blockwise` to TE `Float8BlockScaling`, restricts formats to
  E4M3/HYBRID, and constructs init/forward contexts in
  [`fp8_utils.py`][mcore-fp8].
- Megatron keeps attention DPA/MHA as separate controls in
  [`transformer_config.py`][mcore-config] and applies non-delayed FP8 contexts
  per layer in [`transformer_block.py`][mcore-block].
- Megatron's MoE path validates TE/grouped-GEMM support and quantized routing
  padding in [`transformer_config.py`][mcore-moe-config] and derives the blockwise
  padding alignment in [`moe_utils.py`][mcore-moe].
- Megatron's optimizer contract defaults main parameters, main gradients, and
  Adam states to FP32 in [`optimizer_config.py`][mcore-optimizer].
- TE freezes block dimensions, format, split accumulation, scale behavior, and
  the FP8-attention rejection in [`recipe/__init__.py`][te-recipe].
- TE owns the blockwise FP8 capability check (device, CUDA, cuBLAS, and block
  scaling support) in [`check_fp8_block_scaling_support`][te-quantization] and
  declares blockwise recipe state stateless in
  [`Float8BlockScalingRecipeState`][te-state].
- TE requires cuBLAS 13.3+ for blockwise GroupedLinear (the
  `CUBLAS_GROUPED_GEMM_VERSION` guard in [`cublaslt_grouped_gemm.cu`][te-grouped])
  and enforces it at runtime in
  [`check_grouped_gemm_requirements`][te-grouped-runtime].
- TE defines high-precision-source preservation for FP8 compute weights in
  [`quantization.py`][te-model-init].

[mcore-commit]: https://github.com/NVIDIA/Megatron-LM/commit/cf2f07d7b1315c96c05554c670c43207c6783e5e
[mcore-fp8]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/fp8_utils.py#L554-L672
[mcore-config]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/transformer_config.py#L556-L613
[mcore-moe-config]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/transformer_config.py#L2258-L2289
[mcore-block]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/transformer_block.py#L600-L664
[mcore-moe]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/moe/moe_utils.py#L1340-L1374
[mcore-optimizer]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/optimizer/optimizer_config.py#L184-L207
[te-commit]: https://github.com/NVIDIA/TransformerEngine/tree/v2.15
[te-recipe]: https://github.com/NVIDIA/TransformerEngine/blob/v2.15/transformer_engine/common/recipe/__init__.py#L346-L436
[te-quantization]: https://github.com/NVIDIA/TransformerEngine/blob/v2.15/transformer_engine/pytorch/quantization.py#L120-L127
[te-grouped]: https://github.com/NVIDIA/TransformerEngine/blob/v2.15/transformer_engine/common/gemm/cublaslt_grouped_gemm.cu#L35-L48
[te-grouped-runtime]: https://github.com/NVIDIA/TransformerEngine/blob/v2.15/transformer_engine/common/gemm/cublaslt_grouped_gemm.cu#L302-L309
[te-model-init]: https://github.com/NVIDIA/TransformerEngine/blob/v2.15/transformer_engine/pytorch/quantization.py#L745-L808
[te-state]: https://github.com/NVIDIA/TransformerEngine/blob/v2.15/transformer_engine/pytorch/quantization.py#L1218-L1322

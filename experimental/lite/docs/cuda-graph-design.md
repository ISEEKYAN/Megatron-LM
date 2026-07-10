# CUDA Graph Architecture Study

This document is a source-based design proposal, not an implementation or a
performance claim. The research was completed without running GPU code. The
upstream snapshot is NVIDIA Megatron-LM `fd1121b8` (public `dev` HEAD observed
on 2026-07-10). Open pull-request status is also as observed on 2026-07-10.

## Decision Summary

CUDA Graphs should be an **additive runtime capability of existing MLite
primitives**, not a new family of graphed model implementations. In particular:

- Do not add `QwenCudaGraphModel`, `CudaGraphAttention`, or parallel model trees.
- Do not treat layer-wise, chunk-wise, full-iteration, and partial capture as
  four interchangeable enum values. Layer/chunk/iteration describe capture
  granularity; partial describes coverage within a granularity.
- Keep the public policy model-neutral. A primitive may declare semantic
  graph boundaries such as attention or dense MLP, but it must not recognize
  Qwen, Kimi, GLM, or DeepSeek names.
- Compile the policy against the constructed production model and fail before
  training if any requested target is uncovered or graph-unsafe. Silent eager
  fallback would make performance and correctness claims unauditable.
- Keep optimizer graphing independent from forward/backward graphing. Megatron
  itself captures the optimizer with a separate wrapper and graph.

The recommended first MLite implementation is one closed profile:
**TE-backed partial capture at reusable layer sub-boundaries, initially fixed
shape BSHD attention only**. Dense MLP is the next target after the same
capability is proven on a production composition. BF16/FP8 compute remains controlled
by the precision policy; the CUDA Graph policy only changes launch/replay.
MoE router/preprocess, THD variable length, dynamic CP, chunk-wise capture,
full-iteration capture, and optimizer capture should remain explicit later
gates.

This is intentionally narrower than current Megatron `dev`. MLite has its own
model classes, microbatch loop, PP schedule, THD packing, and optimizer
wrappers. Re-exporting Megatron's flags without reproducing their schedule,
FP8-state, hook, and static-buffer contracts would be incorrect.

## Terminology: Two Axes, Not Four Competing Modes

| Axis | Values | Meaning |
| --- | --- | --- |
| Capture granularity | `layer`, `chunk`, `iteration` | The largest callable represented by one forward/backward graph pair, or by one whole-iteration graph. |
| Capture coverage | `full`, `partial(targets)` | Whether all operations inside that granularity are captured or only named static regions. |
| Backend | MCore-local, Transformer Engine, later custom | Who owns capture, replay, FP8 state, hooks, slots, and memory pools. |
| Shape policy | fixed, bounded/padded, graph-bank, eager fallback | How dynamic inputs are made replay-safe. This is a correctness contract, not a tuning knob. |

The four names requested in this study map as follows:

| Common name | Precise meaning |
| --- | --- |
| Layer-wise | `granularity=layer, coverage=full`; one graph pair per layer and live microbatch slot. |
| Chunk-wise | `granularity=chunk, coverage=full`; one graph pair per local PP/VPP model chunk (`TransformerBlock`) and live slot. |
| Full-iteration | `granularity=iteration, coverage=full`; one graph for all forward/backward microbatches, excluding optimizer. |
| Partial | Usually `granularity=layer, coverage=partial(targets)`; selected attention/MLP/router regions replay while dynamic work stays eager. |

This distinction matters for API design. Otherwise an API eventually grows
invalid combinations such as `variant="partial"` plus a second, hidden answer
to “partial at what granularity?”.

## Megatron `dev` Survey

### Public configuration and dispatch

Megatron exposes three orthogonal fields:

- `cuda_graph_impl`: `none`, `local`, `transformer_engine`, or
  `full_iteration`;
- `cuda_graph_modules`: training coverage such as `attn`, `mlp`, `moe`,
  `moe_router`, `moe_preprocess`, and `mamba`;
- `inference_cuda_graph_scope`: `none`, `layer`, or `block`.

These are defined and documented in
[`transformer_config.py:1020-1105`][mcore-config]. Validation normalizes legacy
flags, rejects per-module coverage for full-iteration graphs, validates MoE
shape constraints, and constrains recompute/offload combinations in
[`transformer_config.py:2708-2953`][mcore-validation]. The user guide summarizes
the same matrix in [`cuda_graph.md:20-47`][mcore-guide-overview].

Graphable modules route every call through a single dispatch point. A local
manager receives the call, while the TE path selects capture or replay and a
microbatch-indexed graph in [`module.py:307-389`][mcore-module-dispatch]. This
single production dispatch point is important: a test-only wrapper around a
module is not evidence that the training path is graphed.

### MCore-local per-layer mechanism

`CudaGraphManager` is a JIT mechanism:

1. During warmup, each module/microbatch runner records forward and backward
   execution order.
2. At the end of the schedule, `create_cudagraphs()` captures graph pairs in
   that exact order so graphs can safely share a pool
   ([`cuda_graphs.py:346-514`][mcore-local-record],
   [`schedules.py:780-804`][mcore-local-schedule]).
3. Replay copies new tensor values into persistent input surfaces, replays the
   forward graph, and uses an autograd node to replay the matching backward
   graph ([`cuda_graphs.py:586-693`][mcore-local-replay]).
4. Pipeline parallelism creates multiple runners when a module can have
   multiple outstanding microbatches; a runner owns one forward/backward graph
   pair ([`cuda_graphs.py:696-760`][mcore-runner]).

For full layer coverage, each `TransformerLayer` owns a manager. For partial
coverage, the layer exposes graphable sub-callables such as attention or MLP;
the manager/replay machinery remains the same. This reuses one schedule-aware
mechanism rather than creating a new model implementation per scope.

### Transformer Engine per-layer mechanism

`TECudaGraphHelper` collects graphable callables from all model chunks, derives
the PP/VPP execution order and the required microbatch slots, builds sample
inputs, and calls TE `make_graphed_callables()`. The captured graphs are then
installed back on each layer by microbatch index
([`cuda_graphs.py:2370-2593`][mcore-te-input],
[`cuda_graphs.py:2665-2710`][mcore-te-capture]). The stock training loop creates
the helper after model/optimizer construction and captures after configured
warmup steps ([`training.py:3968-3977`][mcore-te-helper],
[`training.py:4045-4056`][mcore-te-trigger]).

TE is especially relevant to MLite because `make_graphed_callables()` works
with arbitrary PyTorch callables, whereas MCore's automatic manager assumes
MCore graphable layer classes. The NVIDIA CUDA Graph guide explicitly describes
TE as the custom-model/manual option and MCore local as Megatron-layer-specific
([`TE and Megatron CUDA Graph Support`][nvidia-cg-guide]).

### Full-iteration and optimizer mechanisms

`FullCudaGraphWrapper` first copies every microbatch into persistent CUDA
buffers. At the capture iteration it wraps the complete
`forward_backward_func`; subsequent calls only refresh static inputs and replay
the graph ([`full_cuda_graph.py:104-142`][mcore-static-loader],
[`full_cuda_graph.py:145-241`][mcore-full-wrapper]). The training loop installs
this wrapper around the selected PP schedule, not around a model class
([`training.py:3868-3878`][mcore-full-install]). The graph excludes optimizer,
gradient clipping, and LR scheduling.

Optimizer graphing is a separate `OptimizerCudaGraphWrapper`. It marks Adam as
capturable, captures `optimizer.step()` after warmup, and may share the same
pool/stream as full-iteration capture
([`optimizer_cuda_graph.py:14-52`][mcore-optimizer-wrapper],
[`optimizer/__init__.py:545-587`][mcore-optimizer-build]). This separation should
be preserved in MLite.

### FP8 interaction

Partial graphing cannot merely record TE kernels. FP8 carries persistent amax,
scale, weight-cache, and transpose state across layers and microbatches.
Megatron's local replay:

- restores the current FP8 group and recipe;
- updates cached FP8 weights only on the first microbatch;
- performs delayed-scaling reduction/update after backward replay;
- saves and restores FP8 tensors around capture.

The replay behavior is visible in
[`cuda_graphs.py:620-680`][mcore-local-fp8] and capture-time state preservation
in [`cuda_graphs.py:818-856`][mcore-local-capture-state]. The TE helper passes
per-layer enablement, recipe, weight caching, and the TP/CP amax reduction group
to `make_graphed_callables()` in
[`cuda_graphs.py:2542-2580`][mcore-te-fp8]. The NVIDIA guide explains why
partial graphs must defer global amax reduction and preserve stable addresses
([`TE and Megatron CUDA Graph Support`][nvidia-cg-guide]).

Therefore CUDA Graph and FP8 are orthogonal user policies but coordinated
runtime state. MLite must not create a second FP8 recipe or infer precision from
the graph profile.

### Distributed optimizer and FSDP interaction

Normal MCore distributed optimizer work remains outside per-layer graph
boundaries. Its pre-forward parameter all-gather hooks must still execute at
the right microbatch, which is why the TE helper installs manual hooks after
capture ([`cuda_graphs.py:2712-2720`][mcore-manual-hooks]). Full-iteration
capture excludes the optimizer step; optional optimizer graphing is independent.

Megatron-FSDP has extra address and hook constraints. Current `dev` enables a
graph-safe FSDP mode for any CUDA Graph and disables all-gather from
`start_param_sync()` for full-iteration capture
([`training.py:2346-2362`][mcore-fsdp-setup]). Partial TE+Megatron-FSDP remains
active work: open PRs #4231/#4232 extract and reroute FSDP hooks so TE does not
capture them in the wrong phase, and require persistent double buffers.

MLite's FSDP2 and dist-opt implementations have different hooks and ownership
from MCore. “Megatron supports it” is not sufficient evidence for either MLite
optimizer backend.

### Current parallel and dynamic-shape envelope

The table below distinguishes merged `dev` behavior from open-PR proposals.

| Combination | Current `dev` reading | Main constraint |
| --- | --- | --- |
| TP | Supported by per-layer and full-iteration paths. | Static tensor surfaces and graph-safe collectives are still required. |
| PP/VPP | Per-layer local and TE paths are schedule-aware; full-iteration wraps the PP schedule. | Replay order must match capture order when pools are shared; one layer may need multiple live slots. |
| DP + distributed optimizer | Per-layer forward/backward graphs coexist with eager dist-opt; optimizer graph is separate. | Param-gather/pre-forward hooks must remain reachable and ordered. |
| Megatron-FSDP | Full-iteration has explicit graph-safe setup; partial TE compatibility is still in #4231/#4232 and #5505. | Persistent addresses, correct hook phase, double buffering, and A2A-overlap ordering. |
| Static CP | Supported by graphable attention paths; TE amax reduction can include CP. | CP group and shapes are static graph metadata. |
| Dynamic CP | Not a settled `dev` capability; #5618 proposes one graph bank per supported CP size. | Group identity, THD metadata, RoPE lifetime, and slot capacity change with CP size. |
| Dense model | Layer-wise, partial, and full-iteration are documented. | Recompute/dropout/RNG and `.item()` operations must satisfy graph rules. |
| MoE with padded capacity | Whole MoE can be graphable when dispatch/expert shapes are fixed. | `moe` requires capacity factor plus pad-to-capacity; router/preprocess rules are validated explicitly. |
| Dropless/dynamic MoE | Full layer is not generally capturable. Partial attention/router regions can still be useful. | Routing changes token counts and introduces D2H synchronization/allocation; paged stash/ECHO are separate attempts to make shapes static or device-driven. |
| Fixed/bounded THD | The merged [#4359][pr-4359] foundation exists, but the surface is still evolving. | `cu_seqlens`, maximum sequence metadata, RoPE, padding masks, and microbatch count all need a fixed signature. |
| Truly variable THD | Multiple open PRs (#5672, #5046, #3869, #5258) propose flattening metadata, padding to bounds, graph banks, or fallback. | A changed tensor field, static metadata value, or bound overflow requires rejection, a different graph, or explicit eager fallback. |

The MoE validation is concrete: full MoE capture requires drop-and-pad shapes,
`moe_preprocess` requires `moe_router`, and some all-to-all preprocessing has
D2H synchronization and is rejected
([`transformer_config.py:2813-2868`][mcore-moe-validation]). The MoE guide
recommends partial attention capture when expert shapes are dynamic
([`moe/README.md:501-514`][mcore-moe-guide]). THD plus sequence packing or
dynamic CP requires max alignment in
[`transformer_config.py:3173-3185`][mcore-thd-validation].

## Four Variants

### Layer-wise

**Mechanism.** Capture a forward and backward graph for every graphable layer
and live microbatch slot. Local MCore records the real schedule JIT; TE captures
ahead of replay from an explicit order and sample inputs. The module dispatch
selects the graph for the current microbatch.

**Performance.** It removes Python/kernel-launch overhead inside each layer but
leaves embedding, loss, inter-layer work, PP orchestration, and dynamic
dispatcher work eager. Many small graph replays limit the maximum gain.

**Memory.** Each graph needs persistent IO and saved tensors. Shared pools and
buffer reuse reduce fragmentation, but PP/VPP can multiply graphs by outstanding
microbatch slots. Pool sharing also makes replay order a correctness invariant.

**Best fit.** Dense or shape-stable layers, PP/VPP schedules, FP8-aware TE
modules, and incremental adoption where debugging/fallback boundaries matter.

**Pitfalls.** Full MoE layers fail on dynamic routing shapes; dynamic CP needs a
graph bank; variable THD needs bounded metadata; recompute and FP8 state must be
owned outside the individual graph at the correct scope.

### Chunk-wise

**Mechanism.** Capture one forward/backward graph pair for a whole local model
chunk (`TransformerBlock`). Open PR #5258 proposes a Megatron-owned local mode
that derives live slots from the 1F1B/VPP schedule and replays a single capture
for runtime microbatch counts up to the captured bound.

**Performance.** It can include inter-layer launch gaps, MoE dispatch/combine,
and device-side token-count work that layer-wise graphs leave eager. Fewer,
larger replays can recover more CPU-bound time and reduce PP bubbles.

**Memory.** The captured region retains an entire chunk's activations and static
surfaces. Dynamic-slot replay may need isolated pools because a runtime schedule
can replay only a subset in a different order. It can reduce graph count while
increasing each graph's retained live set.

**Best fit.** CPU-bound large chunks with stable/bounded THD, PP/VPP, and a
graph-safe MoE dispatcher. #5258 reports a Moonlight + HybridEP + paged-stash
validation, but it is still an open PR rather than settled `dev` behavior.

**Pitfalls.** One graph-unsafe operation invalidates the entire chunk. It has a
larger debugging blast radius, more complicated PP slot planning, harder FSDP
hook ownership, and unresolved FP8/FP4 dynamic-slot details. It should not be
independently reimplemented in MLite before the upstream contract settles.

### Full-iteration

**Mechanism.** Copy all microbatches into persistent device buffers, then
capture the complete forward/backward schedule as one graph. The optimizer,
gradient clipping, logging, checkpointing, and scheduler remain eager. Training
and validation own separate graphs.

**Performance.** This maximizes launch-latency reduction and includes PP
communication/schedule work that smaller graphs leave outside.

**Memory.** It pins the largest live set and all static microbatch buffers for
the iteration. A shared pool can reduce reserved-memory duplication with
optimizer/eval graphs, but the graph has the broadest retention lifetime.

**Best fit.** A stable production loop whose complete FWD/BWD path is already
sync-free and shape-static. Dense models are the simplest case; MoE needs
capacity/paged-stash/ECHO-style static or device-driven dispatch.

**Pitfalls.** Any `.item()`, data-dependent Python branch, allocation, host
logging, changing microbatch count, or unsupported collective breaks capture.
NaN checks must be disabled. MLite currently performs `.item()` in PP tensor
sizing and loss handling and creates per-microbatch tensors, so it is not ready
to wrap its loop wholesale.

### Partial

**Mechanism.** Capture only named static regions inside a layer, such as
attention or dense MLP; run dynamic MoE dispatch, loss, hooks, and unsupported
ops eagerly. MCore uses the same local/TE machinery and changes only which
sub-callables are graphable.

**Performance.** It gives up some launch coverage but targets the high-frequency
static work. For dynamic MoE and evolving THD, this may be the only safe way to
obtain useful coverage without padding the entire model to worst case.

**Memory.** Smaller regions retain less state per graph, though more boundaries
mean more graph objects, copies, and replay launches.

**Best fit.** MLite's first delivery: model-neutral attention primitives with
fixed input signatures, plus PP=1 initially. Dense MLP can use the same
capability after attention proves the production path. Partial capture also provides
the foundation for later PP slot planning without committing to chunk/full-loop
capture.

**Pitfalls.** Coverage can silently become empty after a model refactor unless
construction emits and validates a manifest. FP8 amax/weight-cache state and
optimizer hooks cross graph boundaries. Capturing only a test wrapper does not
prove that production dispatch replays the graph.

### Compatibility matrix by variant

“Supported” below refers to the surveyed Megatron mechanism. “Proposed” means
the evidence is in an open PR, not the `dev` snapshot. MLite support remains
none until a later implementation produces its own evidence.

| Variant | PP/VPP | EP and dynamic MoE routing | CP | THD / variable length | FP8 and distributed optimizer |
| --- | --- | --- | --- | --- | --- |
| Layer-wise | Supported; schedule order and multiple live microbatch slots are first-class. | Whole-layer capture needs fixed/drop-and-pad expert shapes; otherwise use partial coverage. | Static CP is supported; dynamic CP graph banks are proposed in #5618. | Fixed/bounded THD foundation is merged, while TE object signatures and truly variable paths remain active work. | Local and TE paths coordinate FP8 state; dist-opt stays eager and requires correctly ordered parameter hooks. |
| Chunk-wise | Proposed #5258 derives 1F1B/VPP slots and bounded dynamic microbatch replay. | #5258 reports HybridEP + paged-stash, but dynamic routing must be made device-driven/static inside the whole chunk. | No independent dynamic-CP qualification was found; it inherits the full chunk's static group/signature requirement. | THD is its primary proposed use case; a bound overflow or metadata change cannot reuse the same graph. | Larger capture crosses more FP8/hook boundaries; #5258's dynamic-slot path still has FP8/FP4 caveats. |
| Full-iteration | Supported by wrapping the complete Megatron FWD/BWD PP schedule. | Only graph-safe fixed/device-driven routing qualifies; ordinary dropless dynamic MoE does not. | Only a fixed CP topology/signature for the whole iteration; no per-step CP change. | Inputs and microbatch count must fit persistent static buffers; host-dependent variation invalidates capture. | FP8 RNG/state must be graph-safe; dist-opt is outside the graph, with optional Adam optimizer graph as a separate capability. |
| Partial | Supported on schedule-aware layer callables. | Best current fit: capture attention/router/preprocess where valid and leave dynamic dispatch/experts eager. | Static CP follows the selected callable; dynamic CP still needs #5618-style graph selection. | Can isolate a fixed attention signature while packing stays eager, but changing `PackedSeqParams` fields still needs a bound/adapter. | Uses TE/local FP8 coordination for selected callables; eager optimizer hooks must remain production-reachable. |

## Open NVIDIA/Megatron-LM Pull Requests

The search included open PRs whose title or body directly concerns CUDA Graphs.
“Ready” means GitHub reports a non-draft open PR; it does not mean merged,
approved, or validated by this study.

### Training architecture and compatibility

| PR | Status | What it changes |
| --- | --- | --- |
| [#5258][pr-5258] Add chunk-wise (whole-block) CUDA graph support for THD training | Open, ready | Adds the chunk granularity described above, stacked on layer-wise THD support; includes PP/VPP slot planning and bounded dynamic microbatches. |
| [#5618][pr-5618] Partial CUDA graph support for dynamic CP | Open, ready | Builds/selects layer-wise TE graph banks per CP size and preserves CP groups, THD metadata, and RoPE lifetime. |
| [#5672][pr-5672] Support `PackedSeqParams` in TE CUDA graphs | Open, ready | Flattens dynamic THD tensor fields into graph inputs, keeps non-tensor metadata static, rebuilds the object inside capture, and validates the signature on replay. |
| [#5046][pr-5046] MCore local training with variable-sized sequences | Open, draft | Pads `cu_seqlens` to a configured bound and proposes an explicit per-step eager bypass when a batch exceeds it. |
| [#3869][pr-3869] Packed-sequence variable-length training | Open, draft | Earlier Mamba/Transformer proposal using padded `cu_seqlens`, bounded fallback, precomputed `seq_idx`, and a TE CP patch. |
| [#4231][pr-4231] / [#4232][pr-4232] TE partial CUDA Graph + Megatron-FSDP | Open, draft (dev/main pair) | Reroutes FSDP forward-wrapped backward hooks to the correct TE phase and requires persistent double buffers to keep addresses stable. |
| [#5505][pr-5505] Megatron-FSDP A2A overlap with partial CUDA Graph | Open, draft | Extends the FSDP/TE hook and buffer contract to the fine-grained 1F1B EP A2A-overlap schedule; explicitly restricts unsupported captured MoE/MLP regions. |
| [#4490][pr-4490] / [#4491][pr-4491] Require CUDA Graph warmup | Open, draft (dev/main pair) | Requires at least one warmup step; the PR reports incorrect 1F1B partial capture and retained-memory changes when capture occurs at step zero. |
| [#2931][pr-2931] Enable optimizer CUDA graph for Adam | Open, draft | Tracks standalone Adam optimizer-step capture. The surveyed `dev` snapshot already contains `OptimizerCudaGraphWrapper`; the still-open PR should not be treated as a merged-status signal. |
| [#2368][pr-2368] MoE ECHO | Open, draft | Research prototype for sync-free dropless MoE using device-driven shapes, preallocated buffers, hot-expert cloning, and full-iteration capture. |
| [#4519][pr-4519] CUDA Graph for VLM `language_model` | Open, ready | Fixes TE helper traversal so a decoder nested under a VLM wrapper is discovered. It illustrates why MLite coverage should be capability-based rather than model-name-based. |

### Correctness and inference-adjacent work

| PR | Status | What it changes |
| --- | --- | --- |
| [#5697][pr-5697] Register private sampling RNG generators | Open, draft | Prevents a private FlashInfer generator from replaying the same Philox offset and producing identical samples. |
| [#1507][pr-1507] Correct MoE load-balancing-loss logging | Open, draft | Avoids caching a process group during capture when the loss must be reduced later. |
| [#5173][pr-5173] Avoid TE CUDA Graph dummy attention masks | Open, ready | Main-branch MoE correctness fix around dummy masks. |
| [#4192][pr-4192] Inference CG + EP behavior | Open, draft | Inference-specific expert-parallel CUDA Graph fixes. |
| [#4285][pr-4285] CUDA graph fix | Open, ready | Underspecified open fix; the title/body do not provide a stable architecture contract, so it is listed but not used to justify the design. |

This volume of in-flight work is itself evidence against claiming a broad
“CUDA Graph supported” switch in MLite. THD, dynamic CP, FSDP hooks, RNG,
logging, and MoE routing each require an explicit contract.

## Current MLite Inventory

There is no CUDA Graph configuration, controller, capture call, or replay path
under `experimental/lite` today.

| Surface | Current behavior | Consequence for CUDA Graphs |
| --- | --- | --- |
| Common runtime config | `MegatronLiteConfig` owns parallel/optimizer/common fields and delegates model-specific fields to `impl_cfg` ([`config.py:24-53`][mlite-config]). | A cross-model graph policy belongs here, not in every model `ImplConfig`. |
| Build lifecycle | Runtime builds the protocol/model/optimizer, loads HF weights, applies post-load updates, and returns a handle ([`runtime.py:170-259`][mlite-build]). | Policy compilation should occur after the final modules, weights, optimizer, and hooks exist, but before training warmup. |
| PP=1 training | `run_microbatch_loop` is a Python loop with per-microbatch scale tensors, model calls, and backward ([`train_step.py:15-74`][mlite-loop]). | It can host partial/layer replay, but is not full-iteration graph-safe as written. |
| PP/VPP training | MLite owns a custom 1F1B/interleaved schedule ([`pipeline.py:20-112`][mlite-pipeline]). | MCore/TE schedule ordering cannot be copied blindly; MLite needs its own slot/order adapter before PP graph support. |
| THD/CP | `PackedSeqParams` contains both tensors and Python/static metadata ([`packed_seq.py:13-58`][mlite-packed]). THD split/reconstruct uses `.item()` and Python loops ([`thd.py:71-129`][mlite-thd-sync]). | Variable THD/CP is not a first-profile candidate. Static metadata and tensor fields need an explicit signature and persistent buffers. |
| Model structure | Models compose ordinary `nn.Module` layers from reusable attention, MLP, router, expert, and parallel primitives. | MCore-local `CudaGraphManager` cannot be attached as-is; TE callable graphing is the smaller reference-backed first backend. |
| FP8 | Some models already open TE FP8 contexts, but the production precision design is separately evolving. | Graph capture must consume the compiled precision policy and TE state; it must not invent an FP8 boolean or recipe. |
| Optimizers | MLite owns dist-opt and FSDP2 paths plus explicit grad-finalization. | Each backend needs independent graph-hook/address evidence; forward/backward graph support must not imply optimizer capture. |

## Design Invariants from the MLite Skills

This proposal follows the repository function-model skills:

- `basic.constitution`: choose the smallest correct design, keep primitives
  replaceable, and use Megatron as the first validation reference.
- `basic.find_reference`: freeze shapes, dtype, seed, process groups, schedule,
  precision, and optimizer while changing only graph replay.
- `primitive.principle`: define forward/backward/update equivalence and a small
  proxy before implementation choices.
- `primitive.contract`: declare owned modules, public API, state, placement,
  valid/invalid combinations, failure modes, and composition validation.
- `primitive.design`: make selection and replaceability explicit; do not hide a
  model-specific dependency inside the primitive.
- `perf.measure`: performance evidence is invalid without the corresponding
  precision/correctness evidence and a stable workload.

The resulting non-negotiable invariants are:

1. Enabling a graph policy must not change model topology, tensor shape/dtype,
   RNG semantics, process groups, loss scaling, gradient reduction, parameter
   ownership, or optimizer update.
2. Graph replay must be bitwise-equal to eager execution when the same kernels
   and reduction order are used. Any non-bitwise threshold requires explicit
   review and an independent reason.
3. Every requested target must appear in a production-reachable coverage
   manifest. Missing, duplicate, or graph-unsafe targets fail during setup.
4. Static input addresses persist for the graph lifetime; shape, stride, dtype,
   device, tensor-field presence, and non-tensor metadata form a replay
   signature.
5. PP/VPP graph slot assignment is derived from MLite's actual schedule. Shared
   pool replay order is checked, not assumed.
6. FP8 recipe/amax/cache state remains owned by the precision capability; graph
   capture coordinates with it through an explicit interface.
7. Optimizer hooks and graphing are independent capabilities. An FWD/BWD graph
   policy cannot silently capture or bypass optimizer behavior.
8. The primitive layer contains no model imports, model-name predicates, or
   model-specific allowlists. Eligibility comes from composed capabilities.
9. Unsupported dynamic THD/CP/MoE inputs fail loudly in the first profile; no
   per-step silent eager fallback is allowed in benchmark evidence.

## API Alternatives

### A. One Boolean Threaded Through Every Model

```python
cfg = MegatronLiteConfig(..., impl_cfg={"cuda_graph": True})
```

This is rejected. It does not say what is captured, which backend owns state,
how dynamic shapes behave, or whether the optimizer is included. Threading it
through each model recreates the model × graph-variant matrix and makes
production coverage impossible to audit.

### B. Fully General Cartesian Policy

```python
CudaGraphPolicy(
    backend="te",
    granularity="layer",
    coverage={"attention", "moe_router"},
    shape_policy="bounded_fallback",
    pool="shared",
    dynamic_cp_sizes={1, 2, 4, 8},
    optimizer=True,
)
```

This is expressive but rejected for the first implementation. It exposes a
large product of combinations before MLite has evidence for them. It also
makes pool selection, graph banks, optimizer capture, and fallback public
semantics that are better kept behind a compiled capability plan.

### C. Closed Profiles Compiled to Semantic Targets (Recommended)

```python
from megatron.lite.primitive.cuda_graph import (
    CudaGraphConfig,
    CudaGraphProfile,
    CudaGraphTarget,
)
from megatron.lite.runtime import MegatronLiteConfig

graph = CudaGraphConfig(
    profile=CudaGraphProfile.PARTIAL_LAYER,
    targets=frozenset({CudaGraphTarget.ATTENTION}),
    warmup_steps=3,
)

cfg = MegatronLiteConfig(model_name="auto", cuda_graph=graph)
```

The first release accepts only:

- `OFF`; and
- `PARTIAL_LAYER` with the fixed-shape `ATTENTION` target.

Later profiles are added only with their own implementation and evidence;
config parsing must reject, not reserve-and-ignore, `LAYER`, `CHUNK`, or
`FULL_ITERATION` until supported. Internal backend and pool selection remain an
implementation plan, not user axes.

Suggested internal flow:

```python
# Runtime, after final model/optimizer/hooks and weight load:
plan = compile_cuda_graph_policy(
    cfg.cuda_graph,
    capabilities=collect_cuda_graph_capabilities(bundle.chunks),
    precision=handle.precision_plan,
    parallel_state=bundle.parallel_state,
)
coverage = bind_cuda_graph_plan(bundle.chunks, plan)
plan.validate_coverage(coverage)

# Production forward/backward path:
controller.warmup_or_replay(
    schedule=mlite_schedule,
    model_call=production_forward,
    batch_signature=batch_signature,
)
```

Reusable primitives expose a small protocol rather than importing the runtime:

```python
class CudaGraphCapability(Protocol):
    target: CudaGraphTarget

    def graph_callable(self) -> Callable: ...
    def graph_signature(self, sample) -> GraphSignature: ...
```

The controller owns graphs, persistent buffers, capture order, RNG
registration, and TE coordination. A model only becomes eligible by composing
capable primitives.

## Proposed Ownership and Failure Contract

### Minimal owned surface

- `megatron/lite/primitive/cuda_graph.py`: immutable public config/enums,
  capability protocol, compiled plan, signatures, controller, coverage
  manifest, and TE adapter;
- `runtime/backends/mlite/config.py`: one common `cuda_graph` field;
- `runtime/backends/mlite/runtime.py`: compile/bind after model construction and
  weight/optimizer finalization; invoke warmup/capture/replay from the real
  `forward_backward` path;
- `primitive/train_step.py`: PP=1 microbatch slot identity and capture boundary;
- later `primitive/parallel/pipeline.py`: MLite schedule order/slot adapter;
- existing attention/dense primitives: semantic capability declarations only.

No model registry entry, model-name allowlist, FP8 recipe builder, or duplicate
model implementation is needed.

### State and placement

- Graph objects, static IO buffers, TE callable wrappers, and registered RNG
  state live in one controller per `ModelHandle`.
- Graphs are device- and rank-local. Process groups are inputs to compilation,
  not globally rediscovered during replay.
- The replay signature contains shape, stride, dtype, device, requires-grad,
  optional tensor presence, and reviewed static metadata.
- BF16/FP8 public outputs preserve the eager dtype contract. Raw internal FP8
  buffers never cross a residual or pipeline boundary merely because graphing
  is enabled.
- Capture starts only after warmup, weight load, optimizer master-weight reload,
  and final hook installation.

### Fail loudly

Setup rejects:

- an unknown profile or target;
- a target with zero coverage, or overlapping/duplicate ownership of one
  primitive boundary;
- PP>1 in the first profile;
- `use_thd=True`, dynamic CP, or a changing microbatch count in the first
  profile;
- MoE router/expert targets before a fixed-shape dispatcher contract exists;
- an FP8 plan that cannot provide TE graph state coordination;
- FSDP2/dist-opt hooks that the adapter cannot keep outside capture at the
  required phase;
- warmup less than one;
- a runtime signature different from the captured signature.

There should be no `try graph; except: eager` path in the first release. An
explicit future bounded-fallback profile may be reasonable for production THD,
but its hit/fallback counts must be observable and excluded from pure graph
benchmark claims.

## What MLite Should and Should Not Build

| Variant | Recommendation | Reason |
| --- | --- | --- |
| Partial layer | **Build first, narrowly.** PP=1, fixed-shape BSHD attention, TE backend; add dense MLP only after the path is proven. | Smallest reusable capability; works with MLite's custom modules; leaves dynamic model/runtime work eager; creates coverage/signature infrastructure needed by every later mode. |
| Whole layer | **Consider second.** Dense/static layers only, then PP after schedule-slot evidence. | More coverage with the same controller, but every operation in the layer must satisfy the signature and RNG contract. |
| Chunk-wise | **Do not implement now.** Track upstream #5258 and reuse its settled slot/memory contract later. | High PP/VPP and dynamic-microbatch complexity; a separate MLite implementation would duplicate active upstream research. |
| Full-iteration | **Do not implement in MLite now.** Use the Bridge/MCore backend when this feature is required. | MLite's runtime contains host syncs and dynamic Python control; full-loop capture disables checks and has the largest correctness/memory blast radius. |
| Optimizer graph | **Separate future task, not part of FWD/BWD CG.** | Backend-specific capturable Adam, master weights, clipping, overflow, and FSDP/dist-opt behavior need their own contract. |

The proposed boundary is deliberately conservative. If partial layer capture
does not show a meaningful measured gain after correctness alignment, MLite
should stop rather than escalate automatically to chunk/full-iteration
complexity.

## Validation Contract for a Later Implementation

This design-only study ran no GPU validation. A later implementation must
leave the following evidence.

### CPU and static proxy

- Parse and round-trip every supported profile/target.
- Reject unknown profile, overlapping/missing coverage, PP/THD/dynamic CP/MoE
  combinations outside the first contract, and warmup zero.
- Build a production-reachable coverage manifest from real model composition;
  test-only wrapper reachability does not count.
- Verify the primitive imports no model package and contains no model-name
  predicates.
- Mock TE availability/version, precision coordination, signature creation,
  and capture-plan selection without importing CUDA-only state.
- Unit-test MLite microbatch slot/order derivation independently of GPU capture.

### Primitive GPU reference

Use the same callable eagerly as the independent reference. Freeze weights,
inputs, seed, RNG offsets, shapes/strides, process groups, precision policy,
microbatch count, and optimizer state. Compare:

- output shape, dtype, and bitwise values;
- input and parameter gradients;
- RNG advancement across multiple replays;
- one optimizer update from the same main weights;
- FP8 amax/scale and weight-cache update timing when FP8 is enabled;
- eager execution for unselected targets and fail-loud behavior for uncovered
  selected targets.

The first useful case is one attention primitive in BF16, followed by one dense
MLP and then one reviewed FP8 recipe. If bitwise comparison is impossible, the
threshold and reason require explicit review; a default tolerance is not
automatic acceptance.

### Composition and end to end

- Start with a real PP=1 production model path whose selected primitives are
  present; do not hard-code its model name in the capability.
- Compare graph OFF vs `PARTIAL_LAYER` from identical checkpoints and input
  streams for loss, gradients, updated weights, logits, and RNG state.
- Add PP only after the same checks pass for the MLite schedule/slot adapter;
  then add static CP and padded MoE as separate cells.
- Add THD only with a fixed signature/bound contract and explicit overflow
  behavior. Dynamic CP needs one graph-bank test per CP size.
- Test dist-opt and FSDP2 independently. Passing one backend does not qualify
  the other.
- Run all GPU work through the repository's Slurm environment and report real,
  non-skipped job results.

### Performance and memory

Following `perf.measure`, freeze workload, precision, parallel layout, warmup,
repeat count, and correctness evidence. Report at least:

- tokens/s and median/p95 step time after capture;
- capture time separately from steady-state replay;
- `memory_allocated` and `memory_reserved` before capture, after capture, and at
  steady state;
- graph count, persistent-buffer bytes, and pool strategy;
- coverage manifest and percentage of step GPU time inside captured regions;
- any eager-fallback count (which must be zero for the first profile).

No throughput number is acceptable without the matching OFF baseline and
correctness/update comparison.

## Staged Delivery

1. **Policy, signatures, coverage, and fail-loud CPU tests.** No CUDA behavior
   changes by default.
2. **PP=1 BF16 partial attention capture.** TE backend, fixed-shape BSHD, one
   production model composition, bitwise eager parity.
3. **Dense MLP, then FP8 coordination.** Reuse the same capability boundary;
   then reuse the precision plan and prove amax/cache/update
   timing against eager execution.
4. **MLite PP schedule adapter.** Derive capture/replay order and live slots from
   `primitive.parallel.pipeline`; validate PP and VPP separately.
5. **Static CP and padded MoE targets.** Add only after their process-group and
   shape contracts pass composition tests.
6. **Bounded THD.** Flatten tensor fields, freeze static metadata, define bound
   overflow behavior, and measure padding/fallback tradeoffs.
7. **Re-evaluate whole-layer capture.** Proceed only if partial capture leaves a
   measured launch-bound gap.
8. **Do not automatically proceed to chunk/full-iteration/optimizer graphs.**
   Each requires a new design and evidence gate.

## Decisions Requested

The following architecture decisions do not require GPU data:

1. Approve capture granularity and coverage as separate concepts.
2. Approve a model-neutral capability/coverage contract and reject CG model
   variants or model allowlists.
3. Approve the closed `OFF` / `PARTIAL_LAYER` first API, with PP=1 fixed BSHD
   attention scope only; dense MLP is the next evidence-gated target.
4. Approve TE callable graphing as the first internal backend while keeping the
   backend out of the public first API.
5. Defer chunk-wise, full-iteration, dynamic THD/CP/MoE, FSDP2/dist-opt graph
   hooks, and optimizer graphing to separate measured gates.

## Source Links

[mcore-config]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_config.py#L1020-L1105
[mcore-validation]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_config.py#L2708-L2953
[mcore-guide-overview]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/docs/user-guide/features/cuda_graph.md#L20-L47
[mcore-module-dispatch]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/module.py#L307-L389
[mcore-local-record]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L346-L514
[mcore-local-schedule]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/pipeline_parallel/schedules.py#L780-L804
[mcore-local-replay]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L586-L693
[mcore-runner]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L696-L760
[mcore-te-input]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L2370-L2593
[mcore-te-capture]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L2665-L2710
[mcore-te-helper]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L3968-L3977
[mcore-te-trigger]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L4045-L4056
[mcore-static-loader]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/full_cuda_graph.py#L104-L142
[mcore-full-wrapper]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/full_cuda_graph.py#L145-L241
[mcore-full-install]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L3868-L3878
[mcore-optimizer-wrapper]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/optimizer_cuda_graph.py#L14-L52
[mcore-optimizer-build]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/__init__.py#L545-L587
[mcore-local-fp8]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L620-L680
[mcore-local-capture-state]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L818-L856
[mcore-te-fp8]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L2542-L2580
[mcore-manual-hooks]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L2712-L2720
[mcore-fsdp-setup]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L2346-L2362
[mcore-moe-validation]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_config.py#L2813-L2868
[mcore-moe-guide]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/moe/README.md#L501-L514
[mcore-thd-validation]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_config.py#L3173-L3185
[nvidia-cg-guide]: https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html

[mlite-config]: ../megatron/lite/runtime/backends/mlite/config.py#L24-L53
[mlite-build]: ../megatron/lite/runtime/backends/mlite/runtime.py#L170-L259
[mlite-loop]: ../megatron/lite/primitive/train_step.py#L15-L74
[mlite-pipeline]: ../megatron/lite/primitive/parallel/pipeline.py#L20-L112
[mlite-packed]: ../megatron/lite/primitive/utils/packed_seq.py#L13-L58
[mlite-thd-sync]: ../megatron/lite/primitive/parallel/thd.py#L71-L129

[pr-5258]: https://github.com/NVIDIA/Megatron-LM/pull/5258
[pr-4359]: https://github.com/NVIDIA/Megatron-LM/pull/4359
[pr-5618]: https://github.com/NVIDIA/Megatron-LM/pull/5618
[pr-5672]: https://github.com/NVIDIA/Megatron-LM/pull/5672
[pr-5046]: https://github.com/NVIDIA/Megatron-LM/pull/5046
[pr-3869]: https://github.com/NVIDIA/Megatron-LM/pull/3869
[pr-4231]: https://github.com/NVIDIA/Megatron-LM/pull/4231
[pr-4232]: https://github.com/NVIDIA/Megatron-LM/pull/4232
[pr-5505]: https://github.com/NVIDIA/Megatron-LM/pull/5505
[pr-4490]: https://github.com/NVIDIA/Megatron-LM/pull/4490
[pr-4491]: https://github.com/NVIDIA/Megatron-LM/pull/4491
[pr-2931]: https://github.com/NVIDIA/Megatron-LM/pull/2931
[pr-2368]: https://github.com/NVIDIA/Megatron-LM/pull/2368
[pr-4519]: https://github.com/NVIDIA/Megatron-LM/pull/4519
[pr-5697]: https://github.com/NVIDIA/Megatron-LM/pull/5697
[pr-1507]: https://github.com/NVIDIA/Megatron-LM/pull/1507
[pr-5173]: https://github.com/NVIDIA/Megatron-LM/pull/5173
[pr-4192]: https://github.com/NVIDIA/Megatron-LM/pull/4192
[pr-4285]: https://github.com/NVIDIA/Megatron-LM/pull/4285

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
- Qualify the constructed production model before capture, automatically bind
  the strongest validated coverage, and expose `enabled`, `partial`, or
  `not-applicable` with structured reasons. A statically ineligible region may
  remain eager; an unexpected failure after a region was declared applicable
  is fatal. Silent eager fallback would make performance and correctness claims
  unauditable.
- Keep optimizer graphing independent from forward/backward graphing. Megatron
  itself captures the optimizer with a separate wrapper and graph. Once an
  optimizer/backend combination is qualified, MLite should select that graph
  automatically rather than expose another user toggle.

MLite production training is THD-only, so a BSHD bring-up profile would not
qualify a production path. The recommended first implementation is
**TE-backed, chunk-wise capture of the max-aligned THD `TransformerBlock`**,
shipped directly rather than staged behind a separate layer-wise partial
delivery. It is gated first by a real packing-utilization cost model and then
by eager parity and end-to-end tokens/s. Within its qualified envelope the
whole chunk is enabled automatically; a chunk that still holds a graph-unsafe
region (for example dropless dynamic MoE dispatch) reports `partial` for the
stable sub-regions it can still capture, or `not-applicable`, with structured
reasons.
BF16/FP8 compute and optimizer choice remain user policy; CUDA Graph changes
only launch/replay, and MLite owns compatibility across precision, optimizer,
parallel backend, and capture coverage. Chunk-wise is the MLite granularity
ceiling: MLite should not implement full-iteration capture, whose extra
coverage is not worth the optimizer and whole-loop static-state complexity for
this runtime. Dynamic CP, dynamic MoE, and optimizer capture remain separate
implementation qualification gates, not public feature switches.

CUDA Graph is a semantics-preserving, monotonically stronger optimization:
once a broader coverage level is qualified for an implementation, it replaces
the narrower default. Diagnostic disablement remains available only as a
correctness oracle, A/B baseline, and debugging escape hatch; `OFF` and
`PARTIAL_LAYER` are not long-term peer user profiles.

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
| Shape qualification | fixed, bounded/padded, graph-bank, or ineligible | How dynamic inputs are proven replay-safe before capture. This is an implementation correctness contract, not a user tuning knob. |

The four names requested in this study map as follows:

| Common name | Precise meaning |
| --- | --- |
| Layer-wise | `granularity=layer, coverage=full`; one graph pair per layer and live microbatch slot. |
| Chunk-wise | `granularity=chunk, coverage=full`; one graph pair per local PP/VPP model chunk (`TransformerBlock`) and live slot. |
| Full-iteration | `granularity=iteration, coverage=full`; one graph for all forward/backward microbatches, excluding optimizer. |
| Partial | Usually `granularity=layer, coverage=partial(targets)`; selected attention/MLP/router regions replay while dynamic work stays eager. |

This distinction matters for API design. Otherwise an API eventually grows
invalid combinations such as `variant="partial"` plus a second, hidden answer
to “partial at what granularity?”. These axes describe implementation and
observability; they do not imply that users should choose among them.

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
| Truly variable THD | Multiple open PRs (#5672, #5046, #3869, #5258) propose flattening metadata, padding to bounds, graph banks, or fallback. | A changed tensor field, static metadata value, or bound overflow requires rejection, a different graph, or a proposal-specific pre-replay eager bypass; it cannot trigger silent recovery from capture failure. |

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
modules, and incremental adoption where debugging/qualification boundaries
matter.

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
hook ownership, and unresolved FP8/FP4 dynamic-slot details. Because MLite
ships this granularity directly, these become the qualification gates the first
delivery must clear rather than reasons to defer: reuse #5258's slot/memory
contract instead of reinventing it, capture only where the whole chunk is
graph-safe, and report `partial`/`not-applicable` (never a silent eager retry)
when a chunk still holds a dynamic region.

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

**Best fit.** The fallback coverage inside MLite's chunk-wise delivery: when a
`TransformerBlock` still holds a dynamic region (dropless MoE dispatch or an
unfused RoPE path), capture the model-neutral static sub-callables — attention
with a fixed input signature, dense MLP — and leave the dynamic work eager,
reporting `partial` with reasons. It is a degraded coverage outcome of the same
controller, not a separate earlier delivery that MLite builds first.

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
| THD/CP | `PackedSeqParams` contains both tensors and Python/static metadata ([`packed_seq.py:13-58`][mlite-packed]). THD split/reconstruct uses `.item()` and Python loops ([`thd.py:71-129`][mlite-thd-sync]). | THD is the required first production envelope, but packing/CP helpers remain outside capture. The graph boundary needs tensorized metadata, persistent bounded buffers, and fixed CP topology. |
| Model structure | Models compose ordinary `nn.Module` layers from reusable attention, MLP, router, expert, and parallel primitives. | MCore-local `CudaGraphManager` cannot be attached as-is; TE callable graphing is the smaller reference-backed first backend. |
| FP8 | Some models already open TE FP8 contexts, but the production precision design is separately evolving. | Graph capture must consume the compiled precision policy and TE state; it must not invent an FP8 boolean or recipe. |
| Optimizers | MLite owns dist-opt and FSDP2 paths plus explicit grad-finalization. | Each backend needs independent graph-hook/address evidence; forward/backward graph support must not imply optimizer capture. |

## THD-only First-delivery Path

MLite's production RL path is packed THD, not BSHD. A BSHD-only CUDA Graph
prototype could exercise the controller but would not satisfy the production
composition gate in `basic.constitution` or `primitive.module.thd`. The first
qualified envelope must preserve `cu_seqlens` boundaries, padded/unpadded
metadata, CP ownership, and the eager forward/backward/update contract on real
packed input.

### What merged Megatron `dev` actually makes static

Merged [#4359][pr-4359] bridges variable THD to CUDA Graphs by changing the
physical representation, not the mathematical sequence boundaries:

1. `pad_packed_seq_alignment` must be `max` (or equal to the configured
   per-DP/CP-rank maximum) when THD packing or dynamic CP is combined with CUDA
   Graphs ([`transformer_config.py:3173-3185`][mcore-thd-validation]).
2. Token-like tensors are padded to `max_seqlen_per_dp_cp_rank`; all four
   `cu_seqlens` tensors are padded to `thd_max_packed_sequences + 1`; the tail
   can be represented as a dummy sequence, and a padding mask keeps padded
   tokens out of loss/router accounting
   ([`packed_seq_params.py:122-191,331-458`][mcore-thd-padding]).
3. Layer capture allocates fixed hidden-state, padding-mask, and `cu_seqlens`
   surfaces at those maxima
   ([`transformer_layer.py:1195-1227`][mcore-thd-static-inputs]).
4. Because a graph boundary cannot safely depend on an arbitrary Python
   dataclass, MCore decomposes `PackedSeqParams` into four tensor kwargs and
   reconstructs it inside the callable. `max_seqlen_q/kv` and other metadata
   come from static config, not a device-to-host read during capture
   ([`transformer_layer.py:1298-1348`][mcore-thd-decompose]).

This allows the values inside fixed-size `cu_seqlens` buffers to change between
replays while their address, shape, dtype, maximum sequence count, maximum
token capacity, CP topology, and non-tensor metadata stay fixed. It is a
fixed-capacity THD contract, not support for arbitrary shapes.

### TE callable and kernel boundary

TE remains the preferred MLite mechanism because `make_graphed_callables()`
accepts arbitrary PyTorch callables. That does not make every THD callable
graph-safe. The MCore adapter above still has to provide tensor-only dynamic
inputs and static metadata. The open main-branch [#5672][pr-5672] follows the
same decomposition/reconstruction approach and records a concrete caveat:
unfused THD RoPE paths that call `.tolist()`/`.item()` on device metadata are
not graph-safe, while the fused RoPE path is the validated candidate.

For MLite the first chunk-wise boundary is therefore:

- **graphable after qualification:** the TE-backed THD `TransformerBlock` —
  attention, QKV/output projections, layer norms, and dense MLP — captured as
  one forward/backward pair per live PP/VPP slot, with fixed token capacity and
  maximum sequence count, tensorized `cu_seqlens`, static CP group, fixed
  optional-field presence, and fused/vectorized RoPE; every op inside the chunk
  must satisfy the same replay signature or the chunk is not enabled;
- **eager by design:** rollout packing/binning, construction and end-padding of
  `PackedTHDBatch`, loss/unpack, logging, and graph-bank selection;
- **eager until separately qualified:** MLite's current THD/CP helpers that use
  Python loops or `.item()` ([`parallel/thd.py:451-595`][mlite-thd-pack]),
  dynamic CP group selection, custom unfused/MRoPE paths, GDN/DSA kernels, and
  optimizer/FSDP hooks. Dropless dynamic MoE dispatch keeps the whole chunk from
  qualifying; such a chunk drops to `partial` (its static attention/MLP
  sub-regions) or `not-applicable` until a pad-to-capacity or device-driven
  dispatcher is qualified.

MLite already supplies `max_seqlen_q/kv` from the packed protocol, so the graph
path must reject the GQA fallback that derives a Python integer from
`cu_seqlens[-1]` ([`gqa.py:204-213`][mlite-gqa-thd]). If fused RoPE is not
available for a model, the safe first boundary narrows to the TE THD attention
core and reports the RoPE/projection exclusions as `partial`; it does not catch
a failed whole-attention capture and retry eagerly.

### Max-alignment cost model

For one CP-local packed microbatch, let:

- `M` be the fixed captured token capacity;
- `T` be real plus existing per-sequence alignment tokens before max-padding;
- `P = M - T` be the added max-alignment tail;
- `l_j` be the true/padded length of each real packed sequence.

Then utilization is `U = T / M`. Token-linear work such as projections, MLPs,
normalization, and many MoE preprocessing kernels sees an upper-bound overhead
of `M / T - 1`. When the tail is represented as one dummy THD sequence,
attention can add work proportional to `P^2`; the useful packed attention work
is proportional to `sum(l_j^2)`, so its first-order overhead indicator is
`P^2 / sum(l_j^2)`. Every retained activation surface also adds approximately
`P * hidden_size * element_bytes` before recompute/pipeline multipliers.

The retained production evidence is not sufficient for an honest point
estimate. It identifies an 8K-class physical pack, a 96-global-sample ×
8-rollout workload, and a real call with roughly 200 sequences, but does not
retain the per-microbatch `T`/`l_j` histogram in this repository. The following
8,192-token sensitivity table is therefore a bound, not a measured result:

| Pack utilization | Added tail `P` | Extra token-linear work | Dummy-tail `P^2` indicator |
| ---: | ---: | ---: | ---: |
| 95% | 410 | 5.3% | 168,100 |
| 90% | 819 | 11.1% | 670,761 |
| 75% | 2,048 | 33.3% | 4,194,304 |
| 50% | 4,096 | 100.0% | 16,777,216 |

Before implementation qualification, a CPU-only pass over the existing
production rollout cache must record `T`, `M`, sequence count, all `l_j`, and
the already-present alignment padding for every packed microbatch. Report
p50/p95/p99 for `U`, `M/T-1`, and `P^2/sum(l_j^2)`, plus estimated persistent
buffer bytes. The decision is not “padding below an arbitrary percentage.” The net value of
max-aligned THD is defined by a measured criterion, not asserted in advance.
The chunk-wise graph qualifies if and only if, on the retained production
rollout distribution, an identical-useful-token A/B measurement shows captured
step time strictly below the diagnostic-off eager baseline at both p50 and
p95/p99, while loss, gradients, RNG advancement, and one optimizer update stay
at parity. Operationally the gate is `replay_step_time < eager_step_time` at p50
and p95 with every parity check green; anything else fails and MLite leaves that
envelope `not-applicable`. The CPU-only padding ledger above plus this GPU A/B
are that measurement protocol — the net value is measurement-pending until both
are run, and MLite ships no capture it has not passed. It is never assumed
positive or negative in advance.

### Static-subgraph alternatives and corrected recommendation

There is no large sequence-independent projection shortcut: embedding,
attention projections, MLPs, normalization, and the LM head all carry the
physical token dimension, so variable `T` still requires max-alignment or a
qualified graph bank. Capturing only a fixed-shape projection after padding can
reduce launch overhead, but it pays the token-linear padding tax above.

The optimizer step is the main genuinely sequence-independent candidate because
parameter/state shapes are static. It remains a separate implementation
contract: capturable optimizer state, overflow, clipping, master weights,
dist-opt/FSDP hooks, and NCCL order must be qualified for the user's optimizer
and backend. Its likely smaller launch-only gain does not justify making it the
first path or exposing `optimizer_graph=True`.

The corrected first envelope is therefore **BF16, TE-backed, chunk-wise capture
of the max-aligned THD `TransformerBlock`, fixed CP topology, fixed max
token/sequence capacity, fused RoPE, and dynamic packing/MoE outside capture**,
informed by open [#5258][pr-5258] and reusing its slot/memory contract rather
than reinventing it. MLite does not build a separate layer-wise partial
delivery first; layer-wise partial exists only as the degraded `partial`
coverage a chunk falls back to when part of it is not graph-safe. Start with a
PP=1 primitive proxy for the eager-parity oracle, but the delivery gate is the
production PP schedule and the real rollout length distribution. A chunk that
still contains dropless dynamic MoE dispatch qualifies only for its static
attention/MLP sub-regions until a pad-to-capacity or device-driven dispatcher is
separately qualified. MLite stops at chunk-wise and does not build
full-iteration capture.

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
- `primitive.module.thd`: preserve `cu_seqlens` sequence boundaries, distinguish
  physical padding from mathematical tokens, and keep CP zigzag ownership in
  the THD/CP primitives.
- `perf.measure`: performance evidence is invalid without the corresponding
  precision/correctness evidence and a stable workload.
- `perf.optimize`: promote a broader graph plan only after the measured
  candidate preserves precision and beats the current qualified default.

The resulting non-negotiable invariants are:

1. Activating graph replay must not change model topology, tensor shape/dtype,
   RNG semantics, process groups, loss scaling, gradient reduction, parameter
   ownership, or optimizer update.
2. Graph replay must be bitwise-equal to eager execution when the same kernels
   and reduction order are used. Any non-bitwise threshold requires explicit
   review and an independent reason.
3. Every automatically selected target must appear in a production-reachable
   coverage manifest. Missing or duplicate ownership is an implementation
   error; a statically graph-unsafe region is excluded with a reason before
   capture.
4. Static input addresses persist for the graph lifetime; shape, stride, dtype,
   device, tensor-field presence, and non-tensor metadata form a replay
   signature.
5. PP/VPP graph slot assignment is derived from MLite's actual schedule. Shared
   pool replay order is checked, not assumed.
6. FP8 recipe/amax/cache state remains owned by the user-selected precision
   capability; graph capture coordinates with it through an explicit
   interface. There is no CG-specific precision toggle.
7. Optimizer selection remains user policy, while optimizer graphing is an
   implementation capability. An FWD/BWD graph plan cannot silently capture or
   bypass optimizer behavior, and users do not select optimizer graphing
   independently.
8. The primitive layer contains no model imports, model-name predicates, or
   model-specific allowlists. Eligibility is decided structurally — whether the
   concrete composed primitives satisfy the replay signature — not by a
   capability registry or a model lookup.
9. Qualification emits exactly one observable aggregate state: `enabled` when
   the strongest verified coverage applies, `partial` when a stable subset is
   captured, or `not-applicable` when no region qualifies. `partial` and
   `not-applicable` include stable reason codes for excluded regions.
10. Dynamic shape, THD, dynamic CP, MoE, optimizer, FSDP, and NCCL constraints
    may restrict full or optimizer capture. The controller prefers a qualified
    stable subgraph; it never turns a capture/replay exception into an eager
    execution path.

## API Alternatives

### A. One Boolean Threaded Through Every Model

```python
cfg = MegatronLiteConfig(..., impl_cfg={"cuda_graph": True})
```

This is rejected. It does not say what is captured, which backend owns state,
how dynamic shapes behave, or whether the optimizer is included. Threading it
through each model recreates the model × graph-variant matrix and makes
production coverage impossible to audit. A diagnostic disable bit may exist at
the common runtime boundary, but it is not a model implementation option and
does not select coverage.

### B. Fully General Cartesian Policy

```python
CudaGraphPolicy(
    backend="te",
    granularity="layer",
    coverage={"attention", "moe_router"},
    shape_policy="bounded",
    pool="shared",
    dynamic_cp_sizes={1, 2, 4, 8},
    graph_optimizer=True,
)
```

This is expressive but rejected. It exposes a large product of combinations
and transfers MLite's compatibility problem to users. Pool selection, graph
banks, backend choice, optimizer capture, and coverage are resolved by the
runtime's explicit assembly, not exposed as a user-facing product.

### C. Closed User-selectable Profiles (Transitional, Rejected Long Term)

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

This is a useful bring-up scaffold because it can force one narrow experiment,
but it must not become the stable API. It incorrectly makes `OFF` and
`PARTIAL_LAYER` peer feature choices, asks users to select implementation
coverage, and would require profile migration every time MLite qualifies a
stronger default. The `CudaGraphProfile`/`CudaGraphTarget` selection surface is
therefore hard-walled: if retained at all during development it stays an
experimental, runtime-level construct and must never appear in
`MegatronLiteConfig.impl_cfg` or any model `ImplConfig` schema. Model
implementations select no capture profile, target, or granularity.

### D. Default-on Explicit Assembly with Diagnostic Override (Recommended)

Normal construction contains no CUDA Graph coverage, backend, target, FP8, or
optimizer-graph choice:

```python
from megatron.lite.primitive.cuda_graph import CudaGraphDebugMode
from megatron.lite.runtime import MegatronLiteConfig

cfg = MegatronLiteConfig(model_name="auto", ...)

# Diagnostic use only: eager correctness oracle, A/B baseline, or debugging.
cfg = MegatronLiteConfig(
    model_name="auto",
    cuda_graph_debug=CudaGraphDebugMode.OFF,
    ...,
)
```

The absence of the debug override means “apply the strongest implementation
coverage qualified for this exact model/runtime plan.” It does not promise
that every configuration is graphable. The resulting plan and status are
observable:

```python
@dataclass(frozen=True)
class CudaGraphStatus:
    state: Literal["enabled", "partial", "not-applicable"]
    implementation: str | None
    captured: tuple[CoverageEntry, ...]
    excluded: tuple[ExclusionReason, ...]
```

`enabled` means the strongest verified coverage for the declared envelope was
bound. `partial` means at least one stable region was bound while other regions
were excluded before capture. `not-applicable` means qualification found no
safe region, so the step is intentionally eager. Reason codes include dynamic
shape/signature, THD metadata, dynamic CP group, dynamic MoE routing,
unqualified optimizer/FSDP hooks, and unqualified NCCL graph behavior.

Suggested internal flow — explicit assembly, not a capability compiler. The
runtime knows it is building chunk-wise THD capture, so it wires the controller
directly to the concrete objects it already holds. There is no capability
registry, capability collector, or generic policy compiler:

```python
# Runtime, after final model/optimizer/hooks and weight load:
controller = CudaGraphController(
    chunks=bundle.transformer_blocks,   # the concrete graphed callables
    schedule=bundle.pp_schedule,        # supplies live-slot capture/replay order
    precision=handle.precision_plan,
    optimizer=handle.optimizer_plan,
    parallel_state=bundle.parallel_state,
    debug=cfg.cuda_graph_debug,
)

# The controller checks each chunk's replay signature directly and records
# coverage: a fully graph-safe chunk is captured whole (enabled); one that still
# holds a dynamic region captures its static sub-regions (partial) or none
# (not-applicable). Nothing is discovered through an abstract interface.
status = controller.qualify_and_bind()

# Production forward/backward path:
controller.warmup_or_replay(
    schedule=bundle.pp_schedule,
    model_call=production_forward,
    batch_signature=batch_signature,
)
```

There is no `CudaGraphCapability` protocol and no capability collector. The
runtime already holds the constructed `TransformerBlock` chunks, the precision
plan, the optimizer plan, and the parallel state, so it assembles the
controller from those concrete objects explicitly. The controller owns graphs,
persistent buffers, capture order, RNG registration, and TE coordination, and
checks each chunk's replay signature directly. Eligibility is not declared by a
primitive implementing an interface; it is decided by whether the concrete chunk
the runtime already built satisfies the signature. The runtime, not the user,
resolves CG × FP8 × optimizer × parallel-backend compatibility by wiring the
plans it holds.

## Proposed Ownership and Failure Contract

### Minimal owned surface

- `megatron/lite/primitive/cuda_graph.py`: internal semantic targets, the
  chunk-wise controller, replay signatures, coverage/status manifest, reason
  codes, and TE adapter — no capability protocol and no generic policy compiler;
- `runtime/backends/mlite/config.py`: at most one common diagnostic override;
  no per-model profile, target, backend, or optimizer-graph field;
- `runtime/backends/mlite/runtime.py`: explicitly assemble and bind the
  controller from the constructed chunks after model construction and
  weight/optimizer finalization; invoke warmup/capture/replay from the real
  `forward_backward` path;
- `primitive/train_step.py`: PP=1 microbatch slot identity and capture boundary;
- later `primitive/parallel/pipeline.py`: MLite schedule order/slot adapter;
- existing `TransformerBlock` and attention/dense primitives: wired directly
  into the controller by the runtime; they declare no capability interface.

No model registry entry, model-name allowlist, FP8 recipe builder, or duplicate
model implementation is needed.

### State and placement

- Graph objects, static IO buffers, TE callable wrappers, and registered RNG
  state live in one controller per `ModelHandle`.
- Graphs are device- and rank-local. Process groups are inputs to controller
  assembly, not globally rediscovered during replay.
- The replay signature contains shape, stride, dtype, device, requires-grad,
  optional tensor presence, and reviewed static metadata.
- BF16/FP8 public outputs preserve the eager dtype contract. Raw internal FP8
  buffers never cross a residual or pipeline boundary merely because graphing
  is enabled.
- Capture starts only after warmup, weight load, optimizer master-weight reload,
  and final hook installation.
- Status, coverage, exclusions, capture count, replay count, and unexpected
  fallback count are emitted per rank in machine-readable form. The last count
  is required to remain zero.

### Qualification versus failure

Qualification happens before capture. Known configuration properties may
produce `partial` or `not-applicable` and intentionally keep the excluded
region eager:

- dynamic shapes or changing microbatch counts without a qualified graph bank;
- THD tensor/static metadata outside a fixed or bounded signature;
- dynamic CP groups or RoPE lifetime outside a qualified bank;
- dropless/dynamic MoE routing outside a fixed/device-driven dispatcher;
- optimizer, FSDP2/dist-opt hooks, or NCCL collectives whose address/order
  contract has not been qualified for the selected implementation.

These are planned exclusions, not recovery from an exception. Partial capture
is preferred whenever at least one stable production subgraph remains useful.
The coverage manifest must show exactly which calls remain eager and why.

The following are fatal implementation or runtime failures:

- an unknown diagnostic value or an inconsistent chunk/signature declaration;
- missing, overlapping, or duplicate ownership for a region the controller
  selected;
- failure to provide FP8 state coordination for a precision combination that
  the implementation declared qualified;
- failure to preserve optimizer/FSDP/NCCL hook phase or address stability for
  a combination declared qualified;
- warmup less than one;
- capture failure after applicability was declared;
- a runtime signature different from the captured signature, replay failure,
  or observed eager execution of a bound call.

There is no `try graph; except: eager` path. A later bounded THD or dynamic-CP
implementation may select an already-qualified graph bank (or determine
`not-applicable`) before capture/replay, but it must expose selection and reason
counts. It must never catch an unexpected capture/replay error and continue
eagerly.

## What MLite Should and Should Not Build

| Variant | Recommendation | Reason |
| --- | --- | --- |
| Chunk-wise | **Build first and directly.** Max-aligned THD `TransformerBlock`, TE backend, fused RoPE, fixed CP; reuse #5258's slot/memory contract; qualify PP=1 proxy first, production PP as the delivery gate. | It captures inter-layer/dispatcher gaps and the bulk of CPU-bound launch pressure in one graph pair per slot; MLite ships this granularity rather than staging a narrower layer-wise delivery first. |
| Partial (fallback coverage) | **Do not build as a separate delivery.** It is the degraded outcome of the same chunk-wise controller: capture static attention/MLP sub-regions when a chunk still holds a dynamic region. | Keeps packing/dynamic MoE eager and still yields a `partial` result with reasons; it is a coverage state of the chunk-wise path, not an earlier milestone. |
| Full-iteration | **Do not implement in MLite.** Use the Bridge/MCore backend if this feature is required. | The incremental gain beyond chunk-wise does not justify whole-loop static state, disabled checks, and the largest correctness/memory blast radius. |
| Optimizer graph | **Separate future implementation task, not a user CG toggle.** | Backend-specific capturable optimizer, master weights, clipping, overflow, and FSDP/dist-opt behavior need their own contract. Once qualified for the user-selected optimizer/backend, MLite enables it and reports the resulting coverage. |

The proposed boundary is deliberately conservative. Coverage expands only
after correctness and performance qualification. If chunk-wise capture does not
show a meaningful measured gain on the production rollout distribution, MLite
should leave that envelope `not-applicable` rather than make users manage a
non-beneficial feature. Chunk-wise is the granularity ceiling; full-iteration is
out of scope.

## Validation Contract for a Later Implementation

This design-only study ran no GPU validation. A later implementation must
leave the following evidence.

### CPU and static proxy

- Parse and round-trip the diagnostic override; reject unknown values and
  warmup zero.
- Verify deterministic `enabled` / `partial` / `not-applicable` aggregation and
  stable reason codes for PP, THD, dynamic CP, MoE, optimizer, FSDP, and NCCL
  exclusions.
- Reject overlapping/missing ownership for automatically selected coverage.
- Build a production-reachable coverage manifest from real model composition;
  test-only wrapper reachability does not count.
- Verify the primitive imports no model package and contains no model-name
  predicates.
- Mock TE availability/version, precision coordination, signature creation,
  and controller assembly/coverage recording without importing CUDA-only state.
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
- eager execution plus observable reasons for statically excluded targets, and
  fail-loud behavior for any bound target that fails capture or replay.

The first useful case is one max-aligned THD `TransformerBlock` chunk in BF16
with tensorized metadata and fused RoPE at PP=1 (reduced layer/expert count per
`basic.constitution`), followed by the production PP composition. One reviewed
FP8 recipe widens the same path only after the padding ledger stays net
positive. If bitwise comparison is impossible, the threshold and reason require
explicit review; a default tolerance is not automatic acceptance.

### Composition and end to end

- Start with a real PP=1 production model path whose graphed `TransformerBlock`
  chunks are present; do not hard-code its model name in the controller or
  signature.
- Compare normal automatic graphing vs the diagnostic-off oracle from identical
  checkpoints and input streams for loss, gradients, updated weights, logits,
  and RNG state.
- Add PP only after the same checks pass for the MLite schedule/slot adapter;
  then add static CP and padded MoE as separate cells.
- Exercise the first THD envelope with a fixed token/sequence-capacity
  signature, fused RoPE, and explicit bound behavior. Dynamic CP needs one
  graph-bank test per CP size.
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
- aggregate status and exclusion-reason counts;
- any unexpected eager-fallback count, which must always be zero.

No throughput number is acceptable without the matching diagnostic-off
baseline, correctness/update comparison, and exact coverage/status manifest.

## Staged Delivery

1. **Qualification, signatures, coverage/status, and fail-loud CPU tests.** Add
   the diagnostic-off oracle and three-state observability; do not expose
   profiles, targets, backend, or optimizer graphing as user axes.
2. **Production-length padding ledger.** Run the CPU-only pack analysis on the
   retained rollout cache and report utilization, token-linear padding tax,
   dummy-tail attention indicator, and persistent bytes. A synthetic BSHD case
   cannot pass this gate.
3. **PP=1 BF16 chunk-wise THD capture.** Qualify TE whole-`TransformerBlock`
   capture (attention, projections, norms, dense MLP) with fixed token and
   sequence-count capacity, tensorized `cu_seqlens`, fixed CP topology, fused
   RoPE, and bitwise eager parity, reusing #5258's slot/memory contract. A chunk
   still holding a dynamic region reports `partial` (static sub-regions) or
   `not-applicable`; capture/replay failures are fatal.
4. **MLite PP schedule adapter and production composition.** Derive
   capture/replay order and live slots from `primitive.parallel.pipeline`, then
   prove correctness and a net tokens/s gain on the same rollout distribution.
   Validate PP and VPP separately.
5. **Static CP and padded/device-driven MoE coverage.** Add only after their
   process-group and shape contracts pass composition tests, so a MoE
   `TransformerBlock` can qualify for whole-chunk capture; otherwise preserve
   useful stable subgraphs and report `partial` with reasons.
6. **FP8 coordination and THD graph banks.** Automatically widen coverage only if
   the padding ledger stays net positive. Consume the existing precision plan,
   prove amax/cache/update timing, and add bounded graph-bank eligibility without
   a CG-specific FP8 toggle. Bound/signature misses are explicit outcomes;
   capture/replay failures remain fatal. Chunk-wise remains the ceiling; do not
   proceed to full-iteration capture.
7. **Qualify optimizer graphs independently.** Preserve the user's optimizer
   choice, prove backend-specific master-weight/clipping/overflow/hooks, then
   let MLite select optimizer capture automatically for qualified combinations.
   Optimizer work requires its own design and evidence gate even though it is not
   a user toggle; full-iteration is not planned.

## Decisions Requested

The following architecture decisions do not require GPU data:

1. Approve capture granularity and coverage as separate concepts.
2. Approve a model-neutral coverage/status contract, assembled explicitly by the
   runtime, and reject CG model variants, model allowlists, or a generic
   capability protocol/policy compiler.
3. Approve CUDA Graph as a default-on, semantics-preserving progressive
   optimization: no long-term `OFF` / `PARTIAL_LAYER` peer profiles; off is a
   diagnostic oracle only.
4. Approve observable `enabled` / `partial` / `not-applicable` outcomes with
   structured reasons, planned eager execution for statically ineligible
   regions, and fatal unexpected capture/replay failures.
5. Approve TE callable graphing as the first internal backend and fixed-capacity
   max-aligned THD chunk-wise `TransformerBlock` capture as the first qualified
   envelope (shipped directly, without a separate layer-wise partial delivery),
   with a real rollout-distribution padding ledger and A/B step-time criterion
   before GPU implementation evidence; BSHD-only qualification is insufficient.
6. Keep FP8 precision and optimizer selection as user policy, while assigning
   CG × FP8 × optimizer × parallel-backend compatibility and automatic graph
   selection to MLite.
7. Approve chunk-wise as the MLite granularity floor and ceiling — the first and
   only forward/backward granularity MLite builds — and reject full-iteration
   implementation. Defer dynamic THD/CP/MoE, FSDP2/dist-opt/NCCL, and optimizer
   graph qualification to separate measured gates; fall back to stable partial
   sub-region coverage whenever whole-chunk coverage is not yet qualified.

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
[mcore-thd-padding]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/packed_seq_params.py#L122-L458
[mcore-thd-static-inputs]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_layer.py#L1195-L1227
[mcore-thd-decompose]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_layer.py#L1298-L1348
[nvidia-cg-guide]: https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html

[mlite-config]: ../megatron/lite/runtime/backends/mlite/config.py#L24-L53
[mlite-build]: ../megatron/lite/runtime/backends/mlite/runtime.py#L170-L259
[mlite-loop]: ../megatron/lite/primitive/train_step.py#L15-L74
[mlite-pipeline]: ../megatron/lite/primitive/parallel/pipeline.py#L20-L112
[mlite-packed]: ../megatron/lite/primitive/utils/packed_seq.py#L13-L58
[mlite-thd-sync]: ../megatron/lite/primitive/parallel/thd.py#L71-L129
[mlite-thd-pack]: ../megatron/lite/primitive/parallel/thd.py#L451-L595
[mlite-gqa-thd]: ../megatron/lite/primitive/modules/gqa.py#L204-L213

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

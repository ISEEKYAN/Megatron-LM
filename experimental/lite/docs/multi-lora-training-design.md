# Multi-LoRA Training Architecture Study

This document is a source-based design proposal, not an implementation or a
performance claim. The research was completed without running GPU code. Public
references and repository state were observed on 2026-07-10.

Code snapshots used for the source reading are mLoRA `89aa53fb`, PEFT
`79f4c362`, Punica `591b5989`, vLLM `c227aaa3`, and LoRAFusion `c48d7fdb`.

## Decision Summary

Multi-LoRA training should be an additive, model-neutral extension of MLite's
existing LoRA primitives. One frozen base model executes a fused input batch;
each sequence selects exactly one independently trainable adapter. The first
implementation should preserve the following properties:

- Existing single-LoRA configuration and checkpoint behavior remain valid.
- Adapter selection is explicit batch metadata. It is not process-global
  mutable state and is not inferred from model names, dataset names, or parameter
  names.
- The eager implementation is the correctness reference. A future fused kernel
  implements the same primitive contract and retains an unfused fallback.
- Losses are normalized per adapter before being summed. Merely averaging over
  the fused batch would change each independent job's effective gradient scale.
- All adapters in the first closed profile use the same rank, targets, dtype,
  dropout, optimizer configuration, and step cadence. Heterogeneous jobs are a
  later scheduling feature, not an implicit extension of the first API.
- Model code only selects and composes primitives. Adapter storage, routing,
  gradient isolation, and execution backends belong below the model layer.
- MoE dispatch may carry a generic token sidecar, but the dispatcher must not
  know what an adapter is. This keeps the primitive reusable and avoids leaking
  model or training-job knowledge into the dispatcher.
- Unknown adapter IDs, missing routing metadata, unsupported parallel layouts,
  and incompatible adapter configurations fail before forward. Silent routing
  to the base model or to adapter zero is forbidden.

The recommended first production profile is:

```text
one frozen Qwen3-MoE base
+ K statically declared LoRA adapters
+ one adapter per sequence
+ homogeneous rank/targets/dropout/dtype
+ every adapter steps once per fused global step
+ eager segmented execution
+ attention projections first
```

The first distributed expansion should add MoE expert targets through generic
routing sidecars, then TP/SP, PP/VPP, THD/CP, and optimizer backends under
separate validation gates. Dynamic adapter loading, weighted adapter mixtures,
heterogeneous ranks, independent per-adapter clocks, and a fused kernel are not
part of the first delivery.

## Scope and Terminology

"Multi-LoRA training" is overloaded in the ecosystem. This proposal uses it to
mean **concurrent training of independent adapters that share a frozen base
model**. If sequence `s` selects adapter `k(s)`, a linear surface computes:

```text
Y_s = X_s W^T + scale[k(s)] * dropout(X_s) A[k(s)]^T B[k(s)]^T
```

`W` is shared and frozen. Each pair `A[k], B[k]` is independently trainable.
The proposal does not mean:

- activating and summing several adapters for one token;
- learning a router over LoRA experts, as in mixture-of-adapter methods;
- merging adapters into the base weights;
- switching one process-wide active adapter between sequential jobs; or
- serving many already-trained adapters.

Those are distinct semantics and should not be hidden behind the same config
keys. Serving systems are useful kernel and slot-management references, but
they are not training-correctness references because they do not preserve
backward, optimizer, loss-normalization, or distributed gradient contracts.

## Current MLite Baseline

MLite already has a substantial single-LoRA path. Multi-LoRA should extend it,
not create a second adapter stack.

### Primitive and model path

[`primitive/modules/lora.py`][mlite-lora] owns:

- `LoraConfig` and configuration normalization;
- `LinearLoRA` with TP/SP-aware sharding and autograd collectives;
- `GroupedLinearLoRA` and `SharedGroupedLinearLoRA` for expert surfaces; and
- helpers that freeze the base and report trainable parameter counts.

[`primitive/modules/gqa.py`][mlite-gqa] adds one `LinearLoRA` delta to QKV and
output projections. [`primitive/modules/experts.py`][mlite-experts] adds one
shared expert adapter to grouped FC1 and FC2 surfaces. The Qwen protocol
normalizes one config, constructs the model, freezes non-LoRA parameters, and
builds one optimizer in [`model/qwen3_moe/lite/protocol.py`][mlite-protocol].

This is a good primitive base, but its state is singular: every targeted layer
owns one pair of LoRA tensors and every sample uses it.

### Batch, loss, and optimizer path

[`runtime/contracts/data.py`][mlite-data] provides `PackedBatch`, including an
`extras` extension point, but has no typed adapter selection. The runtime passes
one batch to a model protocol and reduces one scalar loss. The non-pipeline
microbatch loop divides that scalar by the number of microbatches; PP uses the
same model-provided forward/loss boundary.

One scalar is sufficient only if the model constructs it correctly. For
independent adapters, the correct fused objective for an active set `K` is:

```text
L = sum(k in K) [sum(i routed to k) token_loss[i] / valid_tokens[k]]
```

Because adapter parameters are disjoint and the base is frozen, this sum gives
each adapter the same gradient as its isolated mean-loss job. A single mean over
all valid tokens instead weights adapters by their token counts and is not an
isolated-job reference. This equivalence additionally requires the frozen-base
forward to be sequence-separable. Batch-level MoE capacity, token dropping, or
auxiliary losses can couple jobs and need their own declared semantics.

MLite currently creates one optimizer for the model bundle. This can support a
homogeneous first profile where every adapter steps together. It does not yet
represent independent learning rates, schedulers, accumulation windows, or
completion times.

### Adapter I/O path

[`model/qwen3_moe/lite/lora_adapter.py`][mlite-lora-io] maps one native adapter
to and from a PEFT-style checkpoint. It correctly owns Qwen-specific name and
shape conversion, but it currently assumes one adapter configuration and has
explicit PP/ETP export limits. A bank must orchestrate multiple calls to this
model-owned mapping; the generic LoRA primitive must not learn Qwen checkpoint
names.

### Existing gaps that Multi-LoRA exposes

The current single-adapter path leaves five design gaps that should be fixed or
made explicit during implementation:

1. `freeze_non_lora_params` classifies parameters by substrings. A bank should
   expose its trainable parameters structurally instead of teaching generic
   optimizer code a larger naming convention.
2. Adapter identity is absent from the typed forward path.
3. The loss path has no per-adapter denominator or metrics.
4. Expert token permutation does not carry arbitrary sidecar metadata.
5. Adapter save/load has no bank manifest, stable slot identity, or optimizer
   state ownership per adapter.

## Reference Survey

No single reference supplies the complete MLite contract. The useful pieces
must be separated by responsibility.

| Reference | What is reusable | What is not evidence for MLite |
| --- | --- | --- |
| Megatron Bridge LoRA | Megatron target naming, module matching, single-adapter training structure, and the rule that PEFT is configured separately from the base architecture. | It exposes one PEFT configuration, not mixed-adapter batch training. |
| Hugging Face PEFT | Adapter-name keyed parameter banks and a clear mixed-batch eager decomposition into per-adapter sub-batches. | Its documentation states that mixed-adapter batches are inference-only. It is not a training reference. |
| mLoRA | A real training reference: one shared base, contiguous per-adapter batch ranges, custom autograd, isolated A/B gradients, and multiple training tasks. | Its Python loop and global FP32 adapter policy do not establish Megatron TP/PP/EP correctness. |
| ASPEN / BatchFusion | Fusing job batches over one frozen base and treating scheduling as separate from adapter math. | Reported performance does not transfer to MLite models, hardware, or distributed layouts. |
| Punica and S-LoRA | Segment/slot metadata, grouped low-rank operations, heterogeneous adapter batching, and avoiding base-weight duplication. | They are serving systems. Punica's SGMV path has no training backward contract. |
| vLLM | Stable adapter IDs, active slots, mapping tensors, capacity limits, and explicit MoE adapter layouts. | Its model manager and kernels target serving, caching, and request scheduling. |
| LoRAFusion | Training-specific graph splitting: retain high-performance GEMMs and fuse memory-bound operations around LoRA forward/backward; adaptive multi-job microbatching is separate from the kernel. | Its speedups require fresh MLite measurement and precision evidence. |
| tLoRA | A useful future model for heterogeneous ranks, nano-batches, shared-super-model planning, and per-job progress constraints. | It is a 2026 research design, not a settled MLite or Megatron interface. |

The strongest directly checkable primitive reference is first principles: run
each adapter's current `LinearLoRA` on its selected rows, scatter-add the
results, and compare forward plus all gradients. mLoRA's
[`LoRAFunction`][mlora-lora-function] independently supports this decomposition:
it uses explicit batch start/end ranges in forward and computes adapter-specific
`grad_a`, `grad_b`, and input gradients in backward.

PEFT's [`_mixed_batch_forward`][peft-mixed-forward] is a useful negative and
eager reference. It groups batch indices by adapter and applies the selected
low-rank branch, but PEFT explicitly documents that this mode is inference-only.
MLite must not cite PEFT mixed batching as proof of training correctness.

Punica expresses the same segmented forward algebra and implements it with
SGMV. vLLM adds a production adapter manager with stable slots and mappings.
Both reinforce the proposed data representation, while LoRAFusion is the more
relevant source for a later training kernel because it covers forward/backward
memory traffic rather than only serving-time matrix-vector operations.

## Primitive Principle and Invariants

The proposed primitive is an **adapter bank plus an explicit selection**. Its
principle is:

> Execute disjoint low-rank residual branches over selected sequence/token
> segments while preserving the existing single-LoRA shape, dtype, parallel,
> forward, backward, and update semantics for every adapter independently.

The following invariants are required before choosing implementation details.

### Shape and routing invariants

- Adapter identity is per sequence in the public contract. Token-level IDs are
  derived from sequence lengths and layout transforms.
- `adapter_ids.shape == [num_sequences]` before packing/parallel transforms.
- Every ID resolves to one statically declared slot, or to an explicitly enabled
  base-only sentinel. Unknown IDs fail loudly.
- All tokens derived from one sequence retain its adapter identity through THD
  packing, CP slicing, SP gather/scatter, PP handoff, and MoE permutation.
- Selection metadata never requires gradients and never contributes to the
  checkpointed trainable state.
- The first profile requires homogeneous rank and targets so adapter tensors can
  share one shape contract. Later heterogeneous ranks use explicit offsets or
  rank buckets, never padding hidden behind configuration normalization.

### Forward and backward invariants

- With `K=1` and all samples routed to the only adapter, output and gradients
  match the current single-LoRA primitive under the same seed and dtype.
- With dropout disabled, segmented eager output and gradients match isolated
  per-adapter execution bitwise where PyTorch operations are deterministic.
- Adapter `k` receives gradients only from samples routed to `k`.
- The input gradient is the sum of the frozen-base path and exactly one selected
  adapter path per sample.
- An adapter with zero local samples has a defined zero gradient when collective
  participation requires it; it must not leave an ambiguous `None` gradient
  that causes rank-dependent optimizer or reduction behavior.
- Per-adapter loss normalization matches isolated training. Gradient clipping
  semantics are declared: the first profile may clip the joint bank globally,
  but must not claim equivalence to independently clipped jobs.
- Replica loss scaling matches the gradient collective. For example, if DDP
  averages gradients across `D` replicas, rank `r` contributes
  `D * local_loss_sum[k] / global_valid_tokens[k]`; the subsequent average then
  yields the isolated global-token mean instead of an extra `1/D` factor.

### State and lifecycle invariants

- Adapter names are user-facing identities; integer slots are execution details.
- Slot assignment is deterministic from a saved manifest and identical on all
  participating ranks.
- All adapters are registered before DDP/FSDP wrapping, optimizer construction,
  CUDA Graph capture, and checkpoint restore in the first profile.
- Base parameters remain frozen and are stored once.
- Optimizer and scheduler state ownership is explicit. A synchronized optimizer
  is one profile; independent optimizer clocks are a different runtime profile.
- Eager execution remains independently callable after a fused backend is added.

## Proposed Public and Internal Interfaces

The user configuration should distinguish one adapter from a bank without
turning every existing model config into an adapter scheduler.

```python
@dataclass(frozen=True)
class LoraAdapterSpec:
    name: str
    rank: int
    alpha: int | None = None
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"
    )
    init_from: str | None = None


@dataclass(frozen=True)
class MultiLoraConfig:
    adapters: tuple[LoraAdapterSpec, ...]
    route_key: str = "lora_adapter_ids"
    allow_base_only: bool = False
    execution: str = "eager"  # later: "fused_required"
```

The exact names are provisional, but the distinctions are not:

- `LoraConfig` remains the single-adapter compatibility surface.
- `MultiLoraConfig` owns a static bank and routing policy.
- The model implementation config accepts one or the other, never a partially
  normalized dictionary that can mean both.
- `execution` selects an implementation of the same primitive. `fused_required`
  fails if unavailable; it does not silently fall back and invalidate a
  performance experiment.

### Batch route

Keep the runtime generic by using `PackedBatch.extras`, its documented extension
point, with a typed value:

```python
@dataclass(frozen=True)
class AdapterBatchRoute:
    adapter_ids: torch.Tensor  # int32 [num_sequences], stable slot IDs
```

The Qwen protocol extracts `batch.extras[config.route_key]`, validates it once,
and passes the typed route explicitly through model/layer calls. A context
variable, module-global "active adapter", or mutable singleton would be unsafe
under interleaved PP/VPP, recompute, concurrent evaluation, and CUDA Graph
replay.

At the primitive boundary, layout-specific code compiles the public route into:

```python
@dataclass(frozen=True)
class AdapterSegments:
    slot_ids: torch.Tensor       # int32 [num_segments]
    token_starts: torch.Tensor   # int32 [num_segments]
    token_ends: torch.Tensor     # int32 [num_segments]
```

The first collator should group whole sequences by adapter before THD packing,
so one adapter occupies contiguous ranges. Reordering sequences is valid;
reordering tokens inside a sequence is not. Any inverse permutation needed for
per-sequence metrics is created at the batch boundary, not in every LoRA layer.

### Adapter bank primitive

The eager design should reuse the current sharded modules:

```python
class LinearLoRABank(nn.Module):
    adapters: nn.ModuleList  # each entry is a current LinearLoRA

    def forward(self, x: Tensor, segments: AdapterSegments) -> Tensor:
        # Validate once, execute selected contiguous slices, scatter into delta.
        ...
```

Equivalent banks are needed for grouped expert surfaces. Model composition
chooses the proper bank for a linear surface; the bank does not inspect a model
family. A later packed tensor representation may replace `ModuleList` inside the
primitive, but public config, routing, checkpoint identity, and eager semantics
stay unchanged.

### Loss and metrics

The protocol or a reusable loss helper should return both the summed training
scalar and per-adapter detached metrics:

```python
loss = sum(per_adapter_loss_sum[k] / per_adapter_valid_tokens[k] for k in active)
metrics = {
    "adapter_loss_sum": ...,
    "adapter_valid_tokens": ...,
}
```

Denominators must be reduced over the same replica domain as the corresponding
adapter gradients, and the local numerator must account for whether that
collective sums or averages gradients. Empty adapters are represented
explicitly. Logging may compute a weighted aggregate, but that reporting choice
must not silently change the backward scalar.

### Optimizer contract

For the first profile, one optimizer owns all adapter parameters and steps all
of them together. Construction receives structurally enumerated trainable
parameters from the banks. The optimizer primitive remains unaware of Qwen or
adapter names.

Independent jobs eventually require a higher-level `AdapterOptimizerGroup`
with separate optimizer/scheduler/accumulation state and explicit step masks.
That cannot be emulated safely by setting absent gradients to `None` inside one
ordinary optimizer: weight decay, momentum clocks, gradient clipping, and
scheduler progress would still differ from isolated jobs.

## End-to-End Data Flow

```text
dataset/collator
  -> group complete sequences by adapter
  -> PackedBatch.extras[route_key] = AdapterBatchRoute
  -> protocol validates IDs and compiles AdapterSegments
  -> model forwards route explicitly
       -> attention LoRA bank uses sequence/token segments
       -> MoE dispatcher permutes a generic token sidecar
       -> expert LoRA bank uses the permuted sidecar
  -> loss reduces numerator/denominator per adapter
  -> backward produces disjoint adapter grads and input grads
  -> distributed finalization materializes/reduces required zero grads
  -> optimizer profile steps the declared adapter set
  -> checkpoint writes model shards + bank manifest + optimizer state
```

No layer performs string lookup, dataset lookup, CPU synchronization, or
adapter loading during forward.

## Change Surface

The implementation should be split into reviewable slices. The following map is
an expected change surface, not permission to modify all files in one patch.

| Layer | Expected files | Responsibility |
| --- | --- | --- |
| Primitive contract | `primitive/modules/lora.py` plus focused helper/module files if needed | Config union, bank, route validation, eager segmented forward/backward, structural trainable-parameter enumeration. |
| Generic MoE routing | `primitive/modules/dispatcher.py` and dispatcher utilities | Optional sidecar permutation/all-to-all/combine with no LoRA-specific branch or names. |
| Attention/expert composition | `primitive/modules/gqa.py`, `primitive/modules/experts.py` | Select single adapter versus bank; pass compiled routing explicitly. |
| Model protocol | `model/qwen3_moe/lite/model.py`, `protocol.py` | Extract typed route, compose banks, normalize per-adapter loss, expose manifest hooks. |
| Adapter I/O | `model/qwen3_moe/lite/lora_adapter.py` | Reuse one-adapter PEFT mapping per named slot; preserve model-specific tensor mapping. |
| Runtime optimizer boundary | optimizer adapters only if required by evidence | Ensure static registration, zero-gradient/collective behavior, and state ownership without model lists. |
| Application adapters | VERL/Miles/bench only in follow-up patches | Populate route metadata and synchronize/export named adapters through stable protocol APIs. |
| Validation | unit primitive/model/runtime tests, then committed distributed smoke scripts | Isolated reference parity, integration, parallel composition, checkpoint round-trip, and later performance. |

Two tempting changes are explicitly rejected:

- Do not add `QwenMultiLoraLinear`, `MultiLoraExperts`, or optimizer branches
  keyed by model names.
- Do not copy serving kernels or vendor a second LoRA implementation into each
  model. A fused backend belongs behind the shared bank primitive.

## Feature Compatibility Matrix

"Compatible" below means the proposed contract has a path to preserve the
feature. It is not a claim that current MLite already supports the combination.
Every row needs its own implementation evidence.

| Feature | Contract and required work | First profile |
| --- | --- | --- |
| Single LoRA | Normalize the legacy config to one bank slot internally or keep the current direct fast path; prove output, gradients, and PEFT I/O unchanged. | Required regression gate. |
| TP + SP | Reuse each current `LinearLoRA` sharding rule. Segment metadata must describe the gathered logical sequence order; it is replicated control data, not independently sharded adapter state. | Follow-up distributed gate. |
| DP | All ranks must register identical banks. Either require the same active slot set per synchronized step or materialize zero grads before collectives; reduce each adapter denominator consistently. | Restricted homogeneous active set. |
| PP/VPP | Pass route metadata to every stage and keep deterministic slot identity across chunks. Interleaved stages forbid global active-adapter state. Bank manifests aggregate stage-local parameters. | PP=1 initially. |
| EP/ETP + MoE | Adapter IDs must follow token duplication, expert permutation, all-to-all/DeepEP dispatch, padding, and local expert regrouping. Add a generic dispatcher sidecar rather than LoRA-aware dispatcher code. | Attention targets only initially. |
| THD sequence packing | Public route stays per sequence; compile token segments from `seq_lens/cu_seqlens` after sequence grouping. Padding and loss masks never create adapter tokens. | Supported by design after CPU route tests. |
| Static CP | Derive each CP rank's local token/segment view with the same zigzag/layout transform as activations and labels. Adapter identity is constant across all shards of one sequence. | CP=1 initially. |
| Dynamic CP | Recompile local segments after the selected CP layout. Changing groups/shapes also interacts with graph banks and must fail outside declared profiles. | Deferred. |
| MTP | Every predicted-token branch inherits its parent sequence adapter. Per-adapter MTP numerators and denominators must follow the same label rolling and loss mask as the main loss. | Deferred validation gate. |
| Activation recompute | The route is an immutable explicit input. Dropout needs deterministic RNG preservation; a fused autograd path must document saved versus recomputed tensors. | Dropout=0 in first parity gate. |
| Offload | Banks are declared before wrapping. Parameter/optimizer offload must preserve slot identity and must not reload adapters inside forward. | No dynamic adapter paging. |
| `dist_opt` | Preserve current TP metadata on every adapter tensor; validate absent-gradient finalization and global clipping. One optimizer is compatible only with synchronized adapter steps. | Follow-up optimizer gate. |
| FSDP2 / future mFSDP | Register banks before wrapping and verify sharded parameters, mixed DTensor/Tensor optimizer groups, checkpoint state, and unused-slot gradients. Dynamic module insertion after wrap is invalid. | Deferred backend gate. |
| Muon | Algorithm choice stays orthogonal to the bank. LoRA matrices are 2-D, but eligibility, state, weight decay, and per-adapter clock semantics require an explicit policy and parity tests. | Not enabled by default. |
| FP8 | Precision policy remains independent. A safe first combination keeps adapter math/weights in BF16 while the base may use FP8; FP8 adapter math needs its own recipe and independent precision evidence. | BF16 adapter branch only. |
| CUDA Graphs | Predeclare capacity, keep routing as fixed-shape device tensors, avoid Python dispatch during replay, and capture only after bank/optimizer construction. Dynamic load or changing segment count needs bounds/graph banks. | Eager only. |
| Checkpoint / PEFT I/O | Save a bank manifest and one standard PEFT adapter directory per name. Do not invent a combined tensor format as the only export. Resume also restores slot mapping and optimizer state. | Required before production. |
| VERL / Miles RL | Carry adapter identity from rollout/sample metadata into `PackedBatch`; keep policy loss normalization per adapter and export/sync named adapter weights through protocol hooks. | Separate application patch. |

### Important interaction details

#### MoE routing is the hardest composition boundary

Attention receives tokens in packed sequence order, but expert FC1/FC2 receives
tokens after top-k duplication, permutation, optional EP all-to-all or DeepEP,
expert grouping, and padding. Reusing pre-dispatch segment offsets at the expert
layer would silently apply the wrong adapter.

The dispatcher should accept an optional generic sidecar tensor whose first
dimension matches tokens, apply the same index maps and collectives as hidden
states, and return the dispatched sidecar. No gradient is needed. This primitive
can later carry other token metadata and does not mention adapters or models.

#### DP sparsity affects collectives and loss semantics

One DP rank may have no samples for adapter `k` while another does. A missing
gradient is not automatically equivalent to a zero gradient in DDP, dist-opt,
FSDP, or optimizer state progression. The simplest first distributed contract
requires the same active adapter set on every replica. A later sparse contract
must explicitly materialize/reduce zeros and test deadlock, weight decay, and
state-step behavior.

#### Batch-level model behavior can couple nominally independent jobs

The low-rank branches are disjoint, but some base-model operations are not
necessarily sequence-separable. MoE capacity limits, token dropping, and
load-balancing auxiliary losses may change when two job batches are fused. The
first Qwen profile should use the existing no-aux, no-drop routing contract and
prove isolated parity. Any model with batch-coupled routing needs an explicit
multi-job objective or must stay out of the independent-job profile.

#### Independent job clocks are not a primitive flag

Different datasets may have different batch sizes, accumulation windows,
learning rates, schedulers, or completion points. A shared base and segmented
kernel do not solve scheduling. LoRAFusion and tLoRA both separate multi-job
batch scheduling from LoRA operator fusion. MLite should do the same: first
prove one synchronized profile, then add a runtime scheduler with explicit
per-job progress and fairness contracts.

## Checkpoint and Manifest Design

A bank checkpoint should contain a small manifest next to ordinary distributed
model/optimizer state:

```json
{
  "format": "mlite_lora_bank_v1",
  "base_model": "<identity or digest>",
  "slots": [
    {"slot": 0, "name": "math", "config": "math/adapter_config.json"},
    {"slot": 1, "name": "code", "config": "code/adapter_config.json"}
  ]
}
```

The manifest is illustrative, not a frozen schema. Required invariants are:

- names and slots are unique;
- base model identity and adapter target/rank metadata are validated;
- rank-local distributed state and user-facing PEFT state are separate formats;
- saving one adapter produces a standard one-adapter PEFT directory;
- loading a bank never relies on directory iteration order;
- strict load reports missing/unexpected tensors per adapter; and
- PP/EP/ETP coverage is measured before removing current export restrictions.

Dynamic hot loading is deferred because adding parameters after optimizer and
distributed wrapping changes ownership. A later serving-style cache can be a
different inference/runtime capability without weakening the training contract.

## Future Fused Kernel

The eager primitive should be deliberately shaped as the reference for fusion.
Fusion is justified only after profiling shows segmented LoRA operations are a
material bottleneck under a representative MLite training composition.

### Kernel boundary

The future operator consumes:

```text
X
stacked or pointer-table A/B weights
segment offsets and slot IDs
per-slot scales
dropout probability plus explicit RNG seed/offset when enabled
parallel-layout metadata already resolved by the caller
```

It returns the LoRA delta in the same layout as the current primitive and has a
training backward for `dX`, `dA[k]`, and `dB[k]`. The base GEMM remains the
existing high-performance linear primitive. LoRAFusion's graph-splitting result
argues for preserving compute-bound GEMMs and fusing memory-bound conversion,
dropout, scale, add, and gradient epilogues instead of replacing the whole
linear stack with one opaque kernel.

For a segment `s` routed to `k`:

```text
H_s  = dropout(X_s) A_k^T
D_s  = scale_k H_s B_k^T
dB_k += scale_k dD_s^T H_s
dA_k += scale_k (dD_s B_k)^T dropout(X_s)
dX_s += dropout_backward(scale_k dD_s B_k A_k)
```

The implementation must define accumulation dtype and reduction order. Adapter
segments may repeat a slot, so `dA/dB` accumulation needs deterministic or
reviewed numerical semantics. Heterogeneous ranks should initially be bucketed
by rank; a pointer/offset ABI can follow only after measurement.

### Staged backend plan

1. **Eager segmented reference.** Loop over contiguous adapter segments using
   existing `LinearLoRA`; support complete autograd and CPU tests.
2. **Grouped-library backend.** Evaluate grouped/batched GEMM or TE primitives
   without changing the public contract. This may be sufficient and is easier
   to validate than custom CUDA.
3. **Training fusion.** Fuse memory-bound LoRA forward/backward operations while
   retaining independently validated GEMMs, following LoRAFusion's direction.
4. **Heterogeneous rank and scheduling.** Add rank buckets/nano-batches only with
   an explicit scheduler and per-job progress evidence.
5. **Graph integration.** Make segment metadata and buffers capture-stable only
   after the eager/fused path passes precision and distributed composition.

Punica SGMV or vLLM kernels may inform segment tables and launch organization,
but directly vendoring them would be wrong: their contract is inference and
often matrix-vector/decode oriented. The training backend must have independent
backward, dropout, optimizer, and distributed validation.

### Fusion admission gate

The `primitive.fuse` and `perf.fusion` contracts imply all of the following
before accepting a fused backend:

- all coupled primitives are named;
- the eager path remains an independent fallback/reference;
- primitive forward/backward/update validation passes first;
- end-to-end model precision passes with controlled variables;
- performance is measured with stable warmup/repeat/workload metadata; and
- tokens/s, step time, peak memory, and loss are all reported.

This zero-GPU study provides no speedup, memory, or kernel-support claim.

## Validation Plan

Validation should follow the strongest available reference in layers.

### CPU and single-process contract tests

1. Compare `K=1` bank execution with current `LinearLoRA` forward, `dX`, `dA`,
   and `dB` at dropout zero.
2. Compare `K>1` eager segmented execution with isolated per-adapter calls and
   scatter-add, including non-contiguous input order at the route compiler.
3. Prove gradient isolation: changing adapter `j`'s samples does not change
   adapter `k`'s gradients for `j != k`.
4. Cover base-only sentinel, unknown ID, missing route, duplicate names, empty
   local adapter, mismatched sequence count, rank mismatch, and target mismatch.
5. Compare per-adapter normalized fused loss/gradients with isolated mean-loss
   jobs using unequal sequence lengths and loss masks.
6. Round-trip every named adapter through its standard PEFT representation and
   verify manifest order does not affect slot restoration.

Bitwise comparison is appropriate for deterministic CPU/dropout-zero paths. A
non-bitwise threshold requires a reviewed numerical justification; a blanket
relative tolerance must not hide routing or reduction mistakes.

### Distributed implementation gates

Each later implementation patch should use committed Slurm scripts and record
non-skipped job evidence. A minimal matrix is:

| Gate | Composition | Required comparison |
| --- | --- | --- |
| TP/SP | TP2 with two adapters and unequal sequence lengths | Isolated single-adapter runs versus fused bank, including grads and one optimizer step. |
| PP/VPP | PP2, then PP2+VPP2 with interleaved microbatches | Same slot on every stage; output/loss/grads and checkpoint resume. |
| EP/MoE | EP2 with both adapters crossing ranks, then DeepEP if available | Sidecar permutation parity, expert A/B grads, no fallback to attention-only. |
| THD/CP | packed THD, then CP2 static | Route follows packed/CP-local tokens; per-adapter denominators and logits/loss. |
| Optimizer | `dist_opt`, FSDP2, then any mFSDP profile | Step-1 parameters, optimizer state, zero-local-sample behavior, save/resume. |
| Feature composition | MTP, recompute, BF16-adapter + FP8-base, CUDA Graph when implemented | Independent baseline first, then feature-on comparison with no silent eager fallback. |

The production path must construct the bank from config, carry routing through
`protocol.forward`, run forward/backward, finalize gradients, call
`optimizer.step`, and save/resume. A unit-only bank or test wrapper is not
delivery evidence.

### Performance protocol for a later kernel task

Freeze model, adapter count, rank distribution, targets, sequence-length
distribution, parallel topology, optimizer, dtype, warmup, repeats, and seeds.
Compare:

1. isolated single-adapter jobs;
2. eager fused-batch bank;
3. grouped-library backend; and
4. fused backend.

Report aggregate and per-adapter tokens/s, step-time distribution, peak memory,
adapter fairness/slowdown, and precision. Establish eager correctness before
interpreting speedups. A serving decode benchmark is not a training benchmark.

## Delivery Slices

To keep primitive boundaries reviewable, implementation should land as small
capabilities rather than a model-by-feature cross product:

1. Config, typed route, eager linear bank, and first-principles CPU tests.
2. Qwen attention composition, per-adapter loss, legacy single-LoRA regression,
   and one real forward/backward/step path.
3. Named adapter I/O, bank manifest, optimizer state, and resume.
4. Generic MoE sidecar plus grouped expert bank and EP validation.
5. TP/SP, PP/VPP, THD/CP, optimizer-backend, MTP/recompute/FP8 composition gates.
6. Profiling and only then a grouped or fused kernel behind the same primitive.

Each slice must remove superseded helpers and exports. Test-only callers do not
make a production-unreachable function live. Primitive code must remain free of
model-family names and application-specific routing policy.

## Explicit Non-Goals for the First Implementation

- Weighted sums or learned mixtures of several adapters per token.
- DoRA, QLoRA, LoRA variants, or arbitrary PEFT injection.
- Heterogeneous ranks/targets/dtypes inside one execution bank.
- Independent optimizer/scheduler/accumulation clocks.
- Online job admission, fairness, paging, eviction, or hot loading.
- Adapter merge/unmerge during training.
- A CUDA/Triton kernel or any performance claim.
- Declaring all feature combinations supported from static tests.

## References

- [Megatron Bridge LoRA API][bridge-lora] and [PEFT training guide][bridge-peft]
- [Hugging Face PEFT mixed-adapter implementation][peft-mixed-forward] and
  [mixed-batch caveats][peft-mixed-caveats]
- [mLoRA repository][mlora-repo], [training operator][mlora-lora-function], and
  [VLDB 2025 paper][mlora-paper]
- [ASPEN / BatchFusion paper][aspen-paper]
- [Punica repository and SGMV overview][punica]
- [S-LoRA MLSys 2024 paper][slora]
- [vLLM LoRA feature guide][vllm-lora] and [adapter manager source][vllm-manager]
- [LoRAFusion paper][lorafusion-paper] and [artifact repository][lorafusion-repo]
- [tLoRA paper][tlora-paper]

[mlite-lora]: ../megatron/lite/primitive/modules/lora.py
[mlite-gqa]: ../megatron/lite/primitive/modules/gqa.py
[mlite-experts]: ../megatron/lite/primitive/modules/experts.py
[mlite-protocol]: ../megatron/lite/model/qwen3_moe/lite/protocol.py
[mlite-data]: ../megatron/lite/runtime/contracts/data.py
[mlite-lora-io]: ../megatron/lite/model/qwen3_moe/lite/lora_adapter.py
[bridge-lora]: https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.peft.lora.html
[bridge-peft]: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html
[peft-mixed-forward]: https://github.com/huggingface/peft/blob/79f4c362248d3b3b4bc2ed24704ed3183528c53f/src/peft/tuners/lora/layer.py
[peft-mixed-caveats]: https://github.com/huggingface/peft/blob/79f4c362248d3b3b4bc2ed24704ed3183528c53f/docs/source/developer_guides/lora.md
[mlora-repo]: https://github.com/TUDB-Labs/mLoRA/tree/89aa53fb9e044bb50bed09ffff1863c150385a89
[mlora-lora-function]: https://github.com/TUDB-Labs/mLoRA/blob/89aa53fb9e044bb50bed09ffff1863c150385a89/mlora/model/modules/lora.py
[mlora-paper]: https://www.vldb.org/pvldb/vol18/p1948-tang.pdf
[aspen-paper]: https://arxiv.org/abs/2312.02515
[punica]: https://github.com/punica-ai/punica/tree/591b59899f0a20760821785d06b331c8a2e5cb86
[slora]: https://proceedings.mlsys.org/paper_files/paper/2024/file/906419cd502575b617cc489a1a696a67-Paper-Conference.pdf
[vllm-lora]: https://docs.vllm.ai/en/stable/features/lora/
[vllm-manager]: https://github.com/vllm-project/vllm/blob/c227aaa3f8edd02dae4583e27246430eebabfb25/vllm/lora/model_manager.py
[lorafusion-paper]: https://arxiv.org/abs/2510.00206
[lorafusion-repo]: https://github.com/CentML/lorafusion/tree/c48d7fdb1b5458e9df00963a63df00436f89a059
[tlora-paper]: https://arxiv.org/abs/2602.07263

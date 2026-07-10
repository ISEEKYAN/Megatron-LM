# Multi-LoRA Training Architecture Study

This document is a source-based design proposal, not an implementation or a
performance claim. The research was completed without running GPU code. Public
references and repository state were observed on 2026-07-10.

Code snapshots used for the source reading are mLoRA `89aa53fb`, PEFT
`79f4c362`, Punica `591b5989`, vLLM `c227aaa3`, SkyRL `ccc181e2`, and
LoRAFusion `c48d7fdb`. The MLite PR references were read on 2026-07-10 and
are open, stacked proposals rather than behavior available in this checkout.

## Decision Summary

"Multi-LoRA training" has two independent axes that must not share one vague
flag: multi-tenant lifecycle management and mixed-adapter batch execution.
MLite should first add a time-sliced tenant store around the existing
single-LoRA training path, plus named-adapter passthrough to a multi-LoRA vLLM
rollout pool. A later adapter-bank primitive may execute a fused training batch
where each sequence selects one adapter. Both profiles preserve the following
properties:

- Existing single-LoRA configuration and checkpoint behavior remain valid.
- Tenant selection is an explicit runtime call in the time-sliced profile and
  explicit batch metadata in the mixed-batch profile. It is never inferred
  from model names, dataset names, or parameter names.
- The eager implementation is the correctness reference. A future fused kernel
  implements the same primitive contract and retains an unfused fallback.
- Mixed-batch losses are normalized per adapter before being summed. Merely
  averaging over the fused batch would change each independent job's effective
  gradient scale. The time-sliced profile reuses the isolated single-adapter
  loss unchanged.
- All adapters in the first closed profile use the same rank, targets, dtype,
  dropout, optimizer algorithm, and parallel layout, but own their optimizer
  and step state. Heterogeneous shapes are a later scheduling feature.
- Model code only selects and composes primitives. Adapter storage, routing,
  gradient isolation, and execution backends belong below the model layer.
- MoE dispatch may carry a generic token sidecar, but the dispatcher must not
  know what an adapter is. This keeps the primitive reusable and avoids leaking
  model or training-job knowledge into the dispatcher.
- Unknown adapter IDs, missing routing metadata, unsupported parallel layouts,
  and incompatible adapter configurations fail before forward. Silent routing
  to the base model or to adapter zero is forbidden.

The recommended first production profile is deliberately the smaller
SkyRL-style lifecycle profile:

```text
one frozen Qwen3-MoE base
+ K registered tenant slots in pinned CPU memory
+ exactly one live GPU adapter during a training call
+ independent gradient, FP32 master, optimizer, and step state per tenant
+ homogeneous rank/targets/dropout/dtype and parallel layout
+ named adapter revisions sent to a multi-LoRA vLLM rollout pool
+ attention projections first
```

This profile gains multi-tenant experiment throughput without claiming fused
training: training remains one adapter at a time. A second profile can add a
static in-GPU adapter bank, per-sequence routing, per-adapter loss normalization,
and eventually a fused kernel. Weighted adapter mixtures and heterogeneous
ranks remain out of scope for both initial profiles.

## Scope and Terminology

"Multi-LoRA training" is overloaded in the ecosystem. This proposal uses it to
mean **concurrent lifecycle management of independently trained adapters that
share a frozen base model**. It distinguishes two execution modes:

- **time-sliced training** swaps a complete tenant state into one stable live
  adapter and invokes the existing single-LoRA path; and
- **mixed-batch training** keeps a bank live and routes each sequence to one
  adapter. If sequence `s` selects adapter `k(s)`, a linear surface computes:

```text
Y_s = X_s W^T + scale[k(s)] * dropout(X_s) A[k(s)]^T B[k(s)]^T
```

`W` is shared and frozen. Each pair `A[k], B[k]` is independently trainable.
Neither mode means:

- activating and summing several adapters for one token;
- learning a router over LoRA experts, as in mixture-of-adapter methods;
- merging adapters into the base weights;
- mutating an untracked process-global active adapter; or
- treating multi-LoRA serving as proof that training backward is correct.

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

MLite currently creates one optimizer for the model bundle. That can support a
homogeneous mixed-batch bank where every adapter steps together. The selected
time-sliced profile instead has to snapshot and restore all optimizer and clock
state; MLite does not yet expose that tenant lifecycle contract.

### Adapter I/O path

[`model/qwen3_moe/lite/lora_adapter.py`][mlite-lora-io] maps one native adapter
to and from a PEFT-style checkpoint. It correctly owns Qwen-specific name and
shape conversion, but it currently assumes one adapter configuration and has
explicit PP/ETP export limits. A tenant store or bank must orchestrate multiple
calls to this model-owned mapping; the generic LoRA primitive must not learn
Qwen checkpoint names.

### Adjacent MLite LoRA proposals

Three open, stacked MLite PRs define constraints that a multi-tenant design
must preserve. They are design inputs, not current-main capabilities:

- [PR #73][mlite-pr-73] adds OLoRA-tail initialization. It derives low-rank
  factors from a minor singular subspace, subtracts the initial LoRA delta from
  the frozen base, and broadcasts factors before sharding so all ranks share
  one residual write-back. The important multi-tenant consequence is that the
  in-memory base may no longer equal the rollout engine's canonical base.
- [PR #75][mlite-pr-75] adds dense merged-weight rollout sync. It materializes
  `base + scale * B @ A`, including DTensor materialization, and automatically
  selects this path when LoRA is active. This fixes single-tenant policy sync,
  but updating a shared vLLM base with one tenant's merged weights would clobber
  every other tenant.
- [PR #78][mlite-pr-78] extends `merge_lora` to generic MoE export with local
  grouped and shared expert deltas, while excluding adapter-only tensors from
  the ordinary HF stream. It establishes the expert mapping required for
  merged fallback, not a named-adapter multi-tenant transport.

Therefore rollout export needs two explicit contracts. `merged_weights` keeps
the #75/#78 single-tenant path. `named_adapter_revision` sends a PEFT-style
delta relative to a declared canonical base and is the only first-profile path
that can coexist in one vLLM multi-LoRA pool. OLoRA-tail cannot silently use the
latter: `W0 - D0 + Dt` is not the ordinary rank-`r` delta `Dt` relative to
`W0`. Until a checked conversion or a shared residual-shifted base deployment
exists, OLoRA-tail plus concurrent named-adapter rollout must fail loudly.

### Existing gaps that Multi-LoRA exposes

The current single-adapter path leaves six design gaps that should be fixed or
made explicit during implementation:

1. `freeze_non_lora_params` classifies parameters by substrings. Runtime stores
   and banks should enumerate state structurally instead of teaching generic
   optimizer code a larger naming convention.
2. Adapter identity is absent from the typed forward path.
3. The loss path has no per-adapter denominator or metrics.
4. Expert token permutation does not carry arbitrary sidecar metadata.
5. Adapter save/load has no bank manifest, stable slot identity, or optimizer
   state ownership per adapter.
6. Rollout export does not distinguish a tenant-safe named adapter revision
   from a merged full-policy update that mutates the shared serving base.

## Reference Survey

No single reference supplies the complete MLite contract. The useful pieces
must be separated by responsibility.

| Reference | What is reusable | What is not evidence for MLite |
| --- | --- | --- |
| Trajectory C-LoRA / SkyRL | Warm multi-tenant service split: vLLM mixes named adapters during inference while training time-slices one live adapter; pinned-CPU `AdapterStore` isolates parameters, gradients, FP32 masters, optimizer moments, and step counters. | It does not fuse multiple adapters in one training forward/backward, and its store is coupled to Megatron DDP/DistributedOptimizer internals. |
| Mind Lab MinT | Treat a versioned adapter revision plus base identity as the durable policy unit; separate catalog, CPU, and GPU working sets; move adapter-only revisions through train, rollout, evaluation, serving, and rollback. | Its scale and throughput results do not establish MLite compatibility, and adapter-only handoff requires an identical canonical base. |
| Megatron Bridge LoRA | Megatron target naming, module matching, single-adapter training structure, and the rule that PEFT is configured separately from the base architecture. | It exposes one PEFT configuration, not mixed-adapter batch training. |
| Hugging Face PEFT | Adapter-name keyed parameter banks and a clear mixed-batch eager decomposition into per-adapter sub-batches. | Its documentation states that mixed-adapter batches are inference-only. It is not a training reference. |
| mLoRA | A real training reference: one shared base, contiguous per-adapter batch ranges, custom autograd, isolated A/B gradients, and multiple training tasks. | Its Python loop and global FP32 adapter policy do not establish Megatron TP/PP/EP correctness. |
| ASPEN / BatchFusion | Fusing job batches over one frozen base and treating scheduling as separate from adapter math. | Reported performance does not transfer to MLite models, hardware, or distributed layouts. |
| Punica and S-LoRA | Two useful layouts: BGMV selects a stacked weight bank per row; SGMV uses contiguous segment offsets plus weight pointers. Both avoid base-weight duplication. | They are serving systems. Punica's SGMV/BGMV paths have no training backward, optimizer, or dropout contract. |
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

### C-LoRA and the fixed SkyRL implementation

Trajectory's C-LoRA report and its linked SkyRL commit `ccc181e2` are the
closest reference for the first lifecycle profile. vLLM keeps all active LoRAs
resident and mixes their decode tokens in one inference batch. Training is
different: exactly one adapter is live, one tenant runs a
`forward_backward`, and the next tenant is swapped in.

The report attributes a 2.81x improvement in final completion time for eight
experiments to the warm, cross-job system, while first-experiment time and
per-step latency become worse. Those source-reported results motivate measuring
throughput and latency separately; they are not performance evidence for MLite.

SkyRL's [`AdapterStore`][skyrl-adapter-store] snapshots the live adapter's DDP
parameter and gradient buffers, FP32 master parameters, per-parameter optimizer
state, and optimizer param-group scalar state into pinned CPU tensors. The last
item matters because a shared Adam `step` would corrupt bias correction across
tenants. A homogeneous signature covers rank, alpha, targets, adapter type, and
TP/PP/EP sizes; DP barriers bracket copies into stable live tensors. Its
[end-to-end tests][skyrl-multi-lora-test] interleave two clients and check
training, optimizer-clock, deletion, weight-sync, and sampling isolation.

MLite should borrow the state inventory, immutable signature, stable live
buffers, barriers, and isolation tests. It should not copy SkyRL's substring
test for parameter ownership or import Megatron DDP internals into a generic
primitive. Each optimizer backend must expose a structural tenant-state
adapter, and an unsupported backend must reject registration.

### Mind Lab candidate resolution: MinT

The relevant Mind Lab source found for this study is the May 2026
[MinT paper][mint-paper]. MinT keeps expensive base deployments resident and
moves versioned LoRA adapter revisions through rollout, update, export,
evaluation, serving, and rollback. Its most useful abstraction for MLite is not
the reported scale; it is the separation of durable policy addressability from
bounded CPU/GPU working sets.

MLite can borrow an identity tuple such as `(base_digest, adapter_name,
revision)` and keep catalog, training residency, and rollout residency separate.
The MinT adapter-only handoff is not universally valid: PR #73's residual base
write-back is a concrete counterexample unless conversion or base identity
accounts for it. This report therefore treats MinT as a control-plane and
revision-format reference, not as precision evidence.

### Punica BGMV versus SGMV

Punica's [`bgmv` interface][punica-ops] stores a homogeneous bank in one tensor
and selects `weights[adapter_id[i], layer]` for each row. Its LoRA helper invokes
the kernel twice with a rank-sized temporary. This layout is attractive for a
static, homogeneous MLite bank and fixed-capacity CUDA Graph buffers.

Punica's SGMV interface instead accepts contiguous segment offsets and a
pointer per segment. It matches MLite's proposed `AdapterSegments` and can
represent separately allocated adapters, but pointer lifetime, repeated-slot
gradient accumulation, and capture stability are harder. Neither Punica path
has a training backward; MLite needs independent `dX`, `dA`, and `dB` evidence
before either name can describe a training backend.

## Candidate B Primitive Principle and Invariants

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
- The first bank profile requires homogeneous rank and targets so adapter
  tensors can share one shape contract. Later heterogeneous ranks use explicit
  offsets or rank buckets, never padding hidden behind configuration
  normalization.

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
  semantics are declared: the first bank profile may clip the joint bank
  globally, but must not claim equivalence to independently clipped jobs.
- Replica loss scaling matches the gradient collective. For example, if DDP
  averages gradients across `D` replicas, rank `r` contributes
  `D * local_loss_sum[k] / global_valid_tokens[k]`; the subsequent average then
  yields the isolated global-token mean instead of an extra `1/D` factor.

### State and lifecycle invariants

- Adapter names are user-facing identities; integer slots are execution details.
- Slot assignment is deterministic from a saved manifest and identical on all
  participating ranks.
- Candidate B registers all bank parameters before DDP/FSDP wrapping, optimizer
  construction, CUDA Graph capture, and checkpoint restore. Candidate A creates
  one live adapter before wrapping and later slots only copy state with the same
  immutable signature; it never inserts new parameters.
- Base parameters remain frozen and are stored once.
- Optimizer and scheduler state ownership is explicit. A synchronized optimizer
  is one profile; independent optimizer clocks are a different runtime profile.
- Eager execution remains independently callable after a fused backend is added.

## Candidate APIs and Decision

The design was evaluated as three API shapes rather than treating the first
plausible interface as settled.

| Candidate | Public shape | Strengths | Costs and decision |
| --- | --- | --- | --- |
| A. Tenant handles over a time-sliced store | `runtime.create_adapter(spec) -> AdapterHandle`; every `forward_backward`, `optim_step`, save, and sample call is made through that handle. | Smallest change; reuses the complete single-LoRA forward; preserves independent clocks; maps directly to SkyRL and named vLLM adapters. | Selected first. Swap cost must be measured, and each optimizer backend needs an explicit state adapter. It is multi-tenant training, not fused training. |
| B. Static adapter bank plus batch route | `MultiLoraConfig` declares all slots and `AdapterBatchRoute` selects one slot per sequence. | Shares the base forward across tenants and supplies the eager reference/ABI for a future grouped or fused kernel. | Second profile. It expands loss, MoE sidecar, distributed unused-gradient, checkpoint, and scheduling contracts at once. |
| C. General pluggable `AdapterProvider` | A provider callback resolves arbitrary adapter state during every layer forward. | Could hide stores, banks, paging, and kernels behind one nominal interface. | Rejected. It puts lifecycle policy on the hot primitive path, obscures static registration and graph capture, and creates an abstraction before two working backends exist. |

### Candidate A: explicit tenant handles

The user-facing runtime should bind every operation to a stable identity rather
than exposing a mutable `set_active_adapter()` call:

```python
@dataclass(frozen=True)
class LoraTenantSpec:
    name: str
    lora: LoraConfig
    optimizer: OptimizerConfig


@dataclass(frozen=True)
class AdapterRevision:
    base_digest: str
    adapter_name: str
    revision: int
    format: Literal["named_adapter_revision", "merged_weights"]


class AdapterHandle:
    def forward_backward(self, batch: PackedBatch) -> ModelOutput: ...
    def optim_step(self) -> OptimizerResult: ...
    def save(self, path: Path) -> AdapterRevision: ...
    def export_for_rollout(self, mode: str) -> AdapterRevision: ...
```

`AdapterHandle` delegates to the runtime, which verifies its tenant ID, swaps
that tenant into stable live tensors if necessary, and then invokes the current
single-LoRA protocol. Forward code never reads a process-global active ID.
Create/delete/swap are runtime lifecycle operations; LoRA math remains a
primitive and model-specific PEFT mapping remains in the model composition
layer.

The internal `AdapterStore` owns a backend-neutral slot manifest and delegates
actual tensor enumeration to a narrow optimizer-backend state codec. The codec
must enumerate adapter parameters structurally and include gradients, FP32
masters, moments, scalar counters, and scheduler/accumulation state. FSDP2,
`dist_opt`, mFSDP, or Muon are accepted only when their codec and isolation test
exist. This prevents a generic store from importing model names or guessing
state from parameter substrings.

### Candidate B: static mixed-batch bank

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

Candidate B is not required to ship Candidate A. It becomes admissible only
after the tenant-store path has an isolated correctness baseline and profiling
shows training-side base sharing is worth the larger composition surface.

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

For Candidate B's first bank profile, one optimizer owns all adapter parameters
and steps all of them together. Construction receives structurally enumerated trainable
parameters from the banks. The optimizer primitive remains unaware of Qwen or
adapter names.

Independent jobs eventually require a higher-level `AdapterOptimizerGroup`
with separate optimizer/scheduler/accumulation state and explicit step masks.
That cannot be emulated safely by setting absent gradients to `None` inside one
ordinary optimizer: weight decay, momentum clocks, gradient clipping, and
scheduler progress would still differ from isolated jobs.

### Rollout control plane

Candidate A should pass adapter revisions through the existing application
boundary rather than merge rollout policy into the LoRA primitive:

```text
AdapterHandle.export_for_rollout("named_adapter_revision")
  -> {base_digest, adapter_name, revision, PEFT tensors}
  -> rollout.load_lora_adapter(adapter_name, revision_path)
  -> rollout.generate(..., model=adapter_name)
```

The serving pool predeclares `max_loras`, `max_cpu_loras`, rank, and base
identity. A monotonic revision prevents stale updates from overwriting a newer
tenant. Reloading tenant A must preserve in-flight or subsequent generation for
tenant B; SkyRL's [serving tests][skyrl-serving-test] provide the relevant
control-plane shape, though MLite still needs its own integration evidence.

`merged_weights` remains an explicit single-tenant fallback implemented by the
#75/#78 export path. It pauses or otherwise synchronizes the full-base update
under the existing rollout contract and cannot share one mutable base with
other named tenants. The API must never auto-switch between full merged sync
and adapter-only sync based only on `rank > 0`, because OLoRA-tail/base identity
and multi-tenant safety require an explicit reviewed choice.

## End-to-End Data Flow

Candidate A's first-profile flow is:

```text
tenant request -> AdapterHandle
  -> runtime verifies signature and revision
  -> AdapterStore snapshots current live state and restores requested slot
  -> existing single-LoRA build/forward/backward/optimizer step
  -> updated tenant state is snapshotted on the next switch or save
  -> named adapter revision is exported relative to the declared rollout base
  -> vLLM loads/reloads that name and generate(model=name) selects it
```

Candidate B's later mixed-batch flow is:

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
| Runtime tenant lifecycle | `runtime/` contracts and the selected backend adapter | Explicit handles, create/delete/swap, stable live buffers, slot manifest, and backend state codec; no model-name knowledge. |
| Primitive contract | `primitive/modules/lora.py` plus focused helper/module files if needed | Config union, bank, route validation, eager segmented forward/backward, structural trainable-parameter enumeration. |
| Generic MoE routing | `primitive/modules/dispatcher.py` and dispatcher utilities | Optional sidecar permutation/all-to-all/combine with no LoRA-specific branch or names. |
| Attention/expert composition | `primitive/modules/gqa.py`, `primitive/modules/experts.py` | Select single adapter versus bank; pass compiled routing explicitly. |
| Model protocol | `model/qwen3_moe/lite/model.py`, `protocol.py` | Extract typed route, compose banks, normalize per-adapter loss, expose manifest hooks. |
| Adapter I/O | `model/qwen3_moe/lite/lora_adapter.py` | Reuse one-adapter PEFT mapping per named slot; preserve model-specific tensor mapping. |
| Runtime optimizer boundary | optimizer adapters only if required by evidence | Ensure static registration, zero-gradient/collective behavior, and state ownership without model lists. |
| Application adapters | VERL/Miles/bench only in follow-up patches | Carry tenant identity; load/reload a named vLLM adapter and generate with `model=name`; retain explicit single-tenant merged-weight sync. |
| Validation | unit primitive/model/runtime tests, then committed distributed smoke scripts | Isolated reference parity, integration, parallel composition, checkpoint round-trip, and later performance. |

Two tempting changes are explicitly rejected:

- Do not add `QwenMultiLoraLinear`, `MultiLoraExperts`, or optimizer branches
  keyed by model names.
- Do not copy serving kernels or vendor a second LoRA implementation into each
  model. A fused backend belongs behind the shared bank primitive.

## Feature Compatibility Matrix

"Compatible" below means the proposed contract has a path to preserve the
feature. It is not a claim that current MLite already supports the combination.
Every row needs its own implementation evidence. "Store" refers to Candidate A
and "bank" to Candidate B; support in one does not imply support in the other.

| Feature | Contract and required work | Initial disposition |
| --- | --- | --- |
| Single LoRA | Store keeps the current direct path as its one live slot. Bank either normalizes legacy config to one slot or keeps the direct fast path. Prove output, gradients, update, and PEFT I/O unchanged. | Required regression gate. |
| TP + SP | Store: all ranks swap the same tenant and the backend codec preserves each LoRA shard plus TP metadata. Bank: reuse current sharding and describe the gathered logical sequence order with replicated route metadata. | Follow-up distributed gate per profile. |
| DP | Store: DP barriers bracket snapshot/restore and state shards restore deterministically. Bank: all ranks register identical slots and either share the active set or materialize zero grads; denominators match gradient reduction. | Restricted homogeneous signature/active set. |
| PP/VPP | Store: a handle triggers the same tenant swap on every stage before a pipeline schedule begins. Bank: pass route metadata to every stage and keep deterministic slot identity across chunks; interleaved stages forbid global active state. | PP=1 initially; store and bank need separate PP2 gates. |
| EP/ETP + MoE | Store: the backend codec includes local expert adapter and optimizer shards while the live model reuses current routing. Bank: IDs follow token duplication, permutation, all-to-all/DeepEP, padding, and regrouping through a generic dispatcher sidecar. | Attention targets only initially. |
| THD sequence packing | Store: swapping occurs outside forward, so the live adapter reuses the current single-LoRA THD path. Bank: public route stays per sequence and compiles token segments from `seq_lens/cu_seqlens`; padding/loss masks never create adapter tokens. | Store only after a single-LoRA THD regression; bank after route tests. |
| Static CP | Store: no new route tensor, but all ranks must swap the same tenant before CP collectives. Bank: derive each CP rank's local token/segment view with the same zigzag/layout transform as activations and labels. | CP=1 initially; CP2 is a separate gate for each profile. |
| Dynamic CP | Recompile local segments after the selected CP layout. Changing groups/shapes also interacts with graph banks and must fail outside declared profiles. | Deferred. |
| MTP | Store reuses the live single adapter for all predicted-token branches. Bank makes every branch inherit its parent sequence adapter, with per-adapter numerators/denominators following main-loss label rolling and masks. | Deferred validation gate. |
| Activation recompute | Store forbids tenant swaps inside a forward/backward schedule. Bank route is immutable explicit input. Both preserve dropout RNG; a fused autograd path documents saved versus recomputed tensors. | Dropout=0 in first parity gate. |
| Offload | Store: pinned-CPU tenant slots are lifecycle state, not forward-time paging, and copies finish before train calls. Bank: all parameters are declared before wrapping. Existing parameter/optimizer offload must preserve slot identity in either profile. | No adapter reload inside forward. |
| `dist_opt` | Store: its codec preserves deterministic state partitions, master ownership, and param sync so a restored tenant update equals an isolated update. Bank: preserve TP metadata, absent-gradient finalization, and global clipping; one optimizer supports only synchronized slot steps. | Follow-up optimizer gate. |
| FSDP2 / future mFSDP | Store: its codec copies real sharded params, gradients, masters, optimizer/scalar state and proves materialized params plus updates match an isolated reference; insertion after wrapping is invalid. Bank: register all slots before wrapping and validate unused-slot gradients and checkpoint state. | Fail registration until the selected backend codec and save/resume isolation test exist. |
| Muon | Algorithm choice stays orthogonal. Store: snapshot all Muon per-matrix state and per-tenant clocks; never classify models or parameters by name. Bank: declare eligibility, state, weight decay, global clipping, and step-mask semantics for every LoRA matrix. | Fail loudly in the first profile; no fallback to Adam or shared clocks. |
| FP8 | Precision policy remains independent. A safe first combination keeps adapter params, gradients, masters, and store copies in BF16/FP32 while the frozen base may use FP8; FP8 adapter math needs its own recipe and independent precision evidence. | BF16 adapter branch only; FP8 base is a later controlled-variable gate. |
| CUDA Graphs | Store: copy tenant state into stable live addresses outside capture and replay only a fixed-shape single-adapter graph. Bank: predeclare capacity, use fixed-shape device route buffers, and avoid Python dispatch. Dynamic allocation/pointer changes invalidate capture. | Eager only; graph use fails until address and replay isolation are tested. |
| Checkpoint / PEFT I/O | Save a tenant manifest and one standard PEFT adapter directory per name. Store resume restores complete tenant state; bank resume also restores deterministic slot mapping. Do not make a combined tensor format the only export. | Required before production. |
| VERL / Miles RL | Store: bind trainer and rollout calls to one tenant handle and sync a named revision through protocol hooks. Bank: carry adapter identity into `PackedBatch` and keep policy loss normalization per adapter. The rollout backend is transport, not the precision reference. | Separate application patch and end-to-end gate. |

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

In Candidate B, one DP rank may have no samples for adapter `k` while another
does. A missing gradient is not automatically equivalent to a zero gradient in DDP, dist-opt,
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
batch scheduling from LoRA operator fusion. MLite should do the same: Candidate
A owns independent clocks at the runtime/store boundary; Candidate B first
proves one synchronized bank before adding step masks or scheduling policy.

## Checkpoint and Manifest Design

A tenant store or bank checkpoint should contain a small manifest next to
ordinary distributed model/optimizer state:

```json
{
  "format": "mlite_lora_tenants_v1",
  "base_model": "<identity or digest>",
  "slots": [
    {"slot": 0, "name": "math", "revision": 7, "config": "math/adapter_config.json"},
    {"slot": 1, "name": "code", "revision": 3, "config": "code/adapter_config.json"}
  ]
}
```

The manifest is illustrative, not a frozen schema. Required invariants are:

- names and slots are unique;
- revisions are monotonic per name and bind to the base digest;
- base model identity and adapter target/rank metadata are validated;
- rank-local distributed state and user-facing PEFT state are separate formats;
- Candidate A restores tenant optimizer/scheduler/accumulation state, not only
  adapter tensors;
- saving one adapter produces a standard one-adapter PEFT directory;
- loading a bank never relies on directory iteration order;
- strict load reports missing/unexpected tensors per adapter; and
- PP/EP/ETP coverage is measured before removing current export restrictions.

Candidate A may create a CPU slot at a safe runtime boundary only by cloning the
existing immutable live signature; it does not add parameters. New shapes or
targets after optimizer/distributed wrapping are forbidden. Candidate B defers
all dynamic loading because adding parameters changes ownership. A later
serving-style cache remains a separate runtime capability.

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

### SGMV/BGMV feasibility and relative effort

Punica's names describe serving-forward layouts, not drop-in training kernels.
Their usefulness and missing work are different:

| Direction | Useful starting point | Missing training contract | Relative scope |
| --- | --- | --- | --- |
| Homogeneous BGMV-style bank | Stacked `[slot, layer, ...]` A/B tensors plus one slot ID per row; static capacity is graph-friendly. | Backward for `dX/dA/dB`, repeated-ID reductions, dropout RNG, TP/SP layout, accumulation dtype, optimizer-visible gradients. | **L**: one primitive/backend series after the eager reference; forward alone is not a deliverable. |
| Segmented SGMV-style bank | Contiguous token ranges plus pointer/offset tables; naturally matches grouped sequences and separately allocated adapters. | Everything above plus pointer lifetime, heterogeneous-rank buckets, segment sorting/inverse maps, capture-stable metadata, and safe accumulation when one slot appears in several segments. | **XL**: defer until homogeneous profiling shows a real need. |
| Distributed feature integration | Either kernel behind the same bank primitive. | TP/SP, PP/VPP, EP/MoE sidecars, THD/CP, FSDP/`dist_opt`, checkpoint, recompute, and end-to-end precision/performance evidence. | **XL**, split by feature gates rather than one kernel PR. |

Here **L** means multiple reviewable implementation/validation slices and
**XL** means a follow-on campaign across primitive, distributed, and
application boundaries. These are scope classes, not calendar or speedup
claims. Candidate A does not require either kernel; fusion starts only after a
Candidate B eager profile is correct and profiling attributes material step
time to segmented LoRA work.

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

### Tenant-store and rollout contract tests

1. Enumerate the selected optimizer backend's complete tenant state and prove a
   snapshot/restore round-trip for parameters, gradients, masters, moments,
   scalar step counters, scheduler state, and accumulation state.
2. Interleave identical tenants `A0, B0, A1, B1`; isolated loss, parameters,
   and optimizer state must match at each corresponding step. Then change only
   B's data and prove A is unchanged.
3. Reject mismatched rank, targets, dtype, dropout, optimizer algorithm, and
   parallel layout before the first swap. Delete one tenant and continue the
   other without rebuilding the base model.
4. Export two named revisions relative to the same base, load both into vLLM,
   alternate `model=A/B`, reload A, and prove B is unchanged. Reject stale
   revision numbers and base-digest mismatches.
5. Exercise `merged_weights` separately and prove it cannot be selected for a
   concurrent named-adapter pool. OLoRA-tail plus adapter-only export must hit
   the declared fail-loud gate until a conversion reference exists.

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
| Store lifecycle | Two tenants on the selected optimizer backend, then TP2/PP2 variants | Interleaved calls versus two isolated single-LoRA jobs, including complete state, save/resume, and deletion. |
| Named rollout | Two tenant revisions on one vLLM base | A/B routing, A reload while B remains unchanged, stale revision and wrong-base rejection; no merged-base mutation. |
| TP/SP | TP2 with two adapters and unequal sequence lengths | Isolated single-adapter runs versus fused bank, including grads and one optimizer step. |
| PP/VPP | PP2, then PP2+VPP2 with interleaved microbatches | Same slot on every stage; output/loss/grads and checkpoint resume. |
| EP/MoE | EP2 with both adapters crossing ranks, then DeepEP if available | Sidecar permutation parity, expert A/B grads, no fallback to attention-only. |
| THD/CP | packed THD, then CP2 static | Route follows packed/CP-local tokens; per-adapter denominators and logits/loss. |
| Optimizer | `dist_opt`, FSDP2, then any mFSDP profile | Step-1 parameters, optimizer state, zero-local-sample behavior, save/resume. |
| Feature composition | MTP, recompute, BF16-adapter + FP8-base, CUDA Graph when implemented | Independent baseline first, then feature-on comparison with no silent eager fallback. |

Candidate A's production path must resolve the handle, complete a coordinated
state swap, run the existing protocol through forward/backward/optimizer step,
and save/resume the correct tenant. Candidate B must construct the bank from
config, carry routing through `protocol.forward`, finalize gradients, step, and
save/resume. A unit-only store/bank or test wrapper is not delivery evidence.

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

1. Explicit tenant handles, a static registry, one optimizer-backend state
   codec, and first-principles snapshot/restore/isolation tests.
2. Named adapter revision I/O, vLLM load/reload/model routing, base-digest and
   stale-revision gates, plus explicit separation from #75/#78 merged sync.
3. Store manifest, per-tenant checkpoint/resume, one real
   forward/backward/optimizer-step path, and legacy single-LoRA regression.
4. Typed route, eager linear bank, per-adapter loss, Qwen attention composition,
   and isolated mixed-batch parity as the second profile.
5. Generic MoE sidecar plus grouped expert bank and EP validation.
6. TP/SP, PP/VPP, THD/CP, optimizer-backend, MTP/recompute/FP8 composition gates.
7. Profiling and only then a grouped or fused kernel behind the same primitive.

Each slice must remove superseded helpers and exports. Test-only callers do not
make a production-unreachable function live. Primitive code must remain free of
model-family names and application-specific routing policy.

## Explicit Non-Goals for the First Implementation

- Weighted sums or learned mixtures of several adapters per token.
- DoRA, QLoRA, LoRA variants, or arbitrary PEFT injection.
- Heterogeneous ranks/targets/dtypes inside one execution bank.
- Online admission during a training call, fairness policy, eviction, or
  arbitrary hot loading after distributed wrapping.
- Adapter merge/unmerge during training.
- A CUDA/Triton kernel or any performance claim.
- Declaring all feature combinations supported from static tests.

## References

- [Trajectory C-LoRA field report][trajectory-c-lora], SkyRL's fixed
  [`AdapterStore` implementation][skyrl-adapter-store],
  [multi-LoRA end-to-end tests][skyrl-multi-lora-test], and
  [serving isolation tests][skyrl-serving-test]
- [Mind Lab MinT paper][mint-paper]
- MLite [PR #73 OLoRA-tail][mlite-pr-73],
  [PR #75 merged rollout sync][mlite-pr-75], and
  [PR #78 MoE merged export][mlite-pr-78]
- [Megatron Bridge LoRA API][bridge-lora] and [PEFT training guide][bridge-peft]
- [Hugging Face PEFT mixed-adapter implementation][peft-mixed-forward] and
  [mixed-batch caveats][peft-mixed-caveats]
- [mLoRA repository][mlora-repo], [training operator][mlora-lora-function], and
  [VLDB 2025 paper][mlora-paper]
- [ASPEN / BatchFusion paper][aspen-paper]
- [Punica repository and SGMV overview][punica] plus the pinned
  [BGMV/SGMV Python contracts][punica-ops]
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
[trajectory-c-lora]: https://trajectory.ai/field-notes/multi-lora-training-for-continual-learning
[skyrl-adapter-store]: https://github.com/NovaSky-AI/SkyRL/blob/ccc181e27c04b9f02fe1f7d30483aad96902d7a5/skyrl/backends/skyrl_train/workers/megatron/adapter_store.py
[skyrl-multi-lora-test]: https://github.com/NovaSky-AI/SkyRL/blob/ccc181e27c04b9f02fe1f7d30483aad96902d7a5/tests/tinker/skyrl_train/test_multi_lora_megatron.py
[skyrl-serving-test]: https://github.com/NovaSky-AI/SkyRL/blob/ccc181e27c04b9f02fe1f7d30483aad96902d7a5/tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_multi_lora_serving.py
[mint-paper]: https://arxiv.org/abs/2605.13779
[mlite-pr-73]: https://github.com/ISEEKYAN/Megatron-LM/pull/73
[mlite-pr-75]: https://github.com/ISEEKYAN/Megatron-LM/pull/75
[mlite-pr-78]: https://github.com/ISEEKYAN/Megatron-LM/pull/78
[bridge-lora]: https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.peft.lora.html
[bridge-peft]: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html
[peft-mixed-forward]: https://github.com/huggingface/peft/blob/79f4c362248d3b3b4bc2ed24704ed3183528c53f/src/peft/tuners/lora/layer.py
[peft-mixed-caveats]: https://github.com/huggingface/peft/blob/79f4c362248d3b3b4bc2ed24704ed3183528c53f/docs/source/developer_guides/lora.md
[mlora-repo]: https://github.com/TUDB-Labs/mLoRA/tree/89aa53fb9e044bb50bed09ffff1863c150385a89
[mlora-lora-function]: https://github.com/TUDB-Labs/mLoRA/blob/89aa53fb9e044bb50bed09ffff1863c150385a89/mlora/model/modules/lora.py
[mlora-paper]: https://www.vldb.org/pvldb/vol18/p1948-tang.pdf
[aspen-paper]: https://arxiv.org/abs/2312.02515
[punica]: https://github.com/punica-ai/punica/tree/591b59899f0a20760821785d06b331c8a2e5cb86
[punica-ops]: https://github.com/punica-ai/punica/blob/591b59899f0a20760821785d06b331c8a2e5cb86/src/punica/ops/__init__.py
[slora]: https://proceedings.mlsys.org/paper_files/paper/2024/file/906419cd502575b617cc489a1a696a67-Paper-Conference.pdf
[vllm-lora]: https://docs.vllm.ai/en/stable/features/lora/
[vllm-manager]: https://github.com/vllm-project/vllm/blob/c227aaa3f8edd02dae4583e27246430eebabfb25/vllm/lora/model_manager.py
[lorafusion-paper]: https://arxiv.org/abs/2510.00206
[lorafusion-repo]: https://github.com/CentML/lorafusion/tree/c48d7fdb1b5458e9df00963a63df00436f89a059
[tlora-paper]: https://arxiv.org/abs/2602.07263

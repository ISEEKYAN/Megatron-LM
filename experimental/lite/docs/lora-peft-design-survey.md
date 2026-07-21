# LoRA / PEFT design survey for Megatron Lite

## Decision

Adopt **native, sharding-aware LoRA as a model primitive**.  It must be
configured by the model, expose only adapter parameters as trainable, and be
consumed unchanged by the selected optimizer backend.  The implementation must
not branch on `dist_opt`, FSDP2, or M-FSDP, and the optimizer must not know
which model modules carry adapters.  This is the smallest design that keeps
adapter semantics orthogonal to both optimizer and parallelism.

The existing Qwen3-MoE Lite path already embodies this decision for its native
linear surfaces.  It is therefore a useful audited baseline, not a request to
replace it with generic PEFT monkey-patching.  This report is research/design
only; it changes no production code.

## Reference freshness and scope

References were inspected on 2026-07-20.  The local reference checkouts could
not run `git fetch` because their `.git/FETCH_HEAD` paths are read-only in this
worktree environment.  To avoid treating a stale local checkout as current, I
made a read-only remote HEAD query instead:

| Source | Locked remote HEAD |
| --- | --- |
| `verl-project/verl` | `a32d63538a765577f73b7f420c28cb0fe89f650e` |
| `inclusionAI/AReaL` | `fff7bdfd40ca57c18518a72f676b5108af92e512` |
| `THUDM/slime` | `ea9819f88caa5e043eb8aea992b0969ffe79aa8e` |
| `NVIDIA-NeMo/RL` | `30b24ce7ab1143f216fdff4bccd65727e54275ca` |
| `huggingface/peft` | `667d0c30d59c4fcda7042831b935d40c55dd76f5` |
| `NVIDIA/Megatron-LM` | `12c05a2d89799da77a8532ee1bc4b3c46e9cf426` |

The audited local MLite checkout is
`bfe73d81673200c142af04dc6c139b5c59000377`; the local verl checkout is
`c9f16d7924d5c5df6ea1d4ecf89af4700910251d`.  Remote commit hashes above,
rather than the un-fetched local branches, are the freshness declaration.

The target is conventional training LoRA/PEFT.  The architecture-specific
`q_lora_rank`/`kv_lora_rank` used by MLA is a different model architecture
concept and is not a substitute for adapter fine-tuning.

## Technical basis

For a frozen base linear weight `W`, LoRA trains `A` and `B` and evaluates

```
y = x W^T + (alpha / r) x A^T B^T .
```

The original LoRA work freezes base weights and learns low-rank updates; the
adapter can be merged into `W` for inference [Hu et al., 2021]
(https://arxiv.org/abs/2106.09685).  QLoRA instead holds the frozen base in
4-bit quantized form and backpropagates into LoRA adapters; NF4, double
quantization, and paged optimizers are part of that method [Dettmers et al.,
2023](https://arxiv.org/abs/2305.14314).  DoRA decomposes a weight into
magnitude and direction, applies the low-rank update to direction, and adds a
trainable magnitude [Liu et al., 2024](https://arxiv.org/abs/2402.09353).

**Recommendation:** ship ordinary bf16 LoRA first.  QLoRA is not just a
different adapter: it requires a quantized-base linear/kernel and a checkpoint
contract.  DoRA adds a per-output magnitude parameter and merge math.  Both
remain clean future adapter implementations behind the same primitive-facing
interface, but neither belongs in the first MLite LoRA acceptance scope.

## What the upstreams do

### verl

The required local source inspection shows two distinct meanings of “LoRA”:

* `models/mcore/patch.py` and `config_converter.py` use `q_lora_rank` and
  `kv_lora_rank` for MLA's compressed Q/KV projections.  The patch explicitly
  restores TP-sharded dimensions with gather/scatter where needed.  This is
  **architecture LoRA**, not a PEFT adapter.
* `model_merger/base_model_merger.py` recognizes PEFT `lora_` state entries,
  derives/reads rank and alpha from `lora_train_meta.json`, writes
  `adapter_model.safetensors` plus `adapter_config.json`, and retains a base
  model export.  It is an export boundary, not an optimizer integration.

Thus verl's useful contract for MLite is PEFT-compatible adapter artifacts plus
topology-aware conversion; it does not justify putting PEFT module injection in
an optimizer.  Its MLA collectives reinforce that adapter layout must match the
base linear's shard semantics.

### Cross-framework comparison

| Framework | LoRA and backend position | Parallel/optimizer relationship | Lesson for MLite |
| --- | --- | --- | --- |
| AReaL | Documents LoRA with both Megatron (ZeRO-1) and PyTorch FSDP2; Megatron LoRA is coupled to its vLLM inference path. | Its support matrix separates backend/parallel capabilities from LoRA. | Keep training adapters model-local; make rollout/export a distinct adapter artifact contract. [AReaL README](https://github.com/inclusionAI/AReaL) |
| slime | The public architecture is Megatron training plus SGLang rollout and parameter synchronization; no documented first-party LoRA surface was found at the locked HEAD. | LoRA cannot be inferred from its training/rollout split. | Do not claim rollout support until a concrete adapter-aware resync/load path exists. [slime README](https://github.com/THUDM/slime) |
| NeMo-RL | Supports DTensor/FSDP2 and Megatron Core; its release notes expose Megatron LoRA through Megatron-Bridge PEFT, including target, rank/dim, alpha, dropout, initialization, MoE-sharing, and dtype knobs. | PEFT is configuration/model transformation; its backends own distributed state. | Megatron-native LoRA is the nearest structural reference. [NeMo-RL](https://github.com/NVIDIA-NeMo/RL), [Bridge LoRA API](https://docs.nvidia.com/nemo/megatron-bridge/nightly/apidocs/bridge/bridge.peft.lora.html) |
| Hugging Face PEFT | Generic target matching/injection, checkpoint schema, LoRA/QLoRA/DoRA variants. | Its generic wrapping is valuable at the I/O boundary, but does not preserve a fused Megatron linear's TP/EP layout by itself. | Consume/produce the PEFT artifact schema; do not use generic module traversal as MLite's compute implementation. [PEFT LoRA reference](https://huggingface.co/docs/peft/main/en/package_reference/lora) |

Megatron Bridge validates the chosen direction: it exposes `linear_qkv`,
`linear_proj`, `linear_fc1`, and `linear_fc2`, rank/dim, alpha and dropout as
PEFT configuration.  It also distinguishes one-adapter *performant* fused
QKV/FC1 from canonical separate adapters, a choice MLite must state in its
artifact mapping rather than hide in a linear wrapper.
[Megatron Bridge PEFT guide](https://docs.nvidia.com/nemo/megatron-bridge/0.3.0/training/peft.html)

## MLite audited baseline

The current native Qwen3-MoE Lite implementation has the intended ownership
separation:

* `primitive/modules/lora.py` owns `LoraConfig`, `(rank, alpha, dropout,
  target_modules)`, frozen-base/trainable-adapter accounting, TP/SP-aware
  adapter math, `LinearLoRA`, and per/shared-expert grouped adapters.
* `primitive/modules/gqa.py` attaches deltas beside its QKV and output
  projections; `primitive/modules/experts.py` attaches shared grouped deltas
  beside local expert FC1/FC2.  The base linear remains the base linear
  (`_VanillaColParallelMatmul`, `te.Linear`, or `te.GroupedLinear`).
* `model/qwen3_moe/lite/protocol.py` normalizes model-owned configuration,
  builds the model, then freezes non-adapter parameters **before** choosing
  its currently implemented `dist_opt`, FSDP2, or no-optimizer inference
  path.  It has no M-FSDP branch at this baseline.  The optimizer receives the
  ordinary model parameter set and sees only the `requires_grad=True` adapter
  tensors.
* `model/qwen3_moe/lite/lora_adapter.py` gathers/slices TP layouts and maps
  fused QKV/gated FC1 and experts to PEFT names.  It validates rank, alpha and
  targets on load, and writes `adapter_model.safetensors` plus PEFT metadata.

This is substantially safer than name-based generic injection: the primitive
knows whether the base output is column-sharded, whether a row-parallel input
is local, whether sequence parallel is active, and whether adapters are shared
per expert.

## Signed-off design contract for future model coverage

### Boundaries and API

1. Add only a model-owned `lora` configuration: `rank`, `alpha`, `dropout`,
   canonical `target_modules`, and an explicit `adapter_layout` choice for
   fused modules.  `rank == 0` must preserve the exact no-LoRA model path.
2. A target model composes `LinearLoRA`/grouped LoRA next to a known MLite
   linear surface.  The adapter primitive may depend on `ParallelState`; the
   base linear and every optimizer must not depend on model target names.
3. After all chunks are built, freeze all base tensors and assert both
   `trainable_numel > 0` and that every trainable name is an adapter name.
   Construct the optimizer from those normal parameters.  There is no
   `if lora: ...` branch in `dist_opt`, FSDP2, or M-FSDP.
4. Export/import only adapter state in PEFT-compatible safetensors plus an
   `adapter_config.json`; validate base-model identity/revision, rank, alpha,
   target set, fused-layout convention, and model architecture before loading.
   A training-resume checkpoint separately contains adapter optimizer state;
   it must not silently contain or update frozen base optimizer state.

### Parallel layout contract

| Surface | Adapter layout and required behavior |
| --- | --- |
| Column parallel / `_VanillaColParallelMatmul` / `te.Linear(parallel_mode="column")` | `B` follows the output shard.  `A` is replicated, except an explicitly rank-partitioned formulation with the matching all-gather/reduction autograd path.  SP must gather/reduce-scatter exactly as the base linear requires. |
| Row parallel `te.Linear(parallel_mode="row")` | `A` consumes the local input shard and the resulting delta is reduced exactly once on the TP group; never all-gather a row-parallel input merely to make LoRA convenient. |
| `te.GroupedLinear` MoE | Choose per-expert or shared-across-local-experts explicitly.  Adapter tensor ownership follows EP/ETP local-expert ownership; routed token `splits` are part of the primitive contract. |
| PP | Each pipeline chunk creates/freezes/exports only its local adapter tensors.  Pipeline stage is a state-dict placement, not a replica dimension. |
| CP | LoRA changes hidden projections, not token ownership.  CP input/output layouts and attention collectives are untouched; test the real CP model path. |

`TP × PP × EP × CP` is a model/adapter placement problem.  Optimizer choice is
an independent data-parallel state problem.  This yields the intended matrix:

| Optimizer backend | Required LoRA-specific behavior |
| --- | --- |
| `dist_opt` | No LoRA special case.  Shard optimizer state only for trainable adapters using the existing DP groups/checkpoint metadata. |
| FSDP2 | No LoRA special case.  FSDP materializes/shards ordinary adapter parameters according to its existing wrap/mesh policy; frozen base remains excluded from gradients and optimizer groups. |
| M-FSDP | No LoRA special case.  M-FSDP placement/state ownership must derive from adapter `Parameter` placement, never from target module strings. |

### Weight update and rollout resynchronization

The inference side must receive an adapter artifact, not a transient merged
training weight by default.  At a synchronization point, rank 0 (or the
existing checkpoint owner) exports canonical PEFT adapter tensors after TP/EP
gather; the rollout worker verifies base revision and adapter metadata, loads
the adapter, and applies it to its inference model.  The base remains immutable
and can be cached across policy versions.  Optional merge is an explicit
offline/export operation `W <- W + scale * B @ A`, with a numerical
merge/unmerge test; it is not the training resync protocol.

This prevents a vLLM/SGLang worker from accidentally receiving TP-local shards
or a full, duplicate base model.  It also gives a clear future choice: a
rollout engine that cannot load PEFT adapters is unsupported until an explicit
merge/export implementation and its memory budget are accepted.

## Validation gates before implementation acceptance

The gates follow the MLite skills contracts: `basic.constitution` requires a
checkable Megatron/HF reference and end-to-end evidence;
`primitive.design` requires a primitive contract and replaceability;
`primitive.parallel.tp`, `primitive.optimizer.distopt`, and
`primitive.optimizer.fsdp` require explicit sharding/ownership validation;
`application.verl` requires an end-to-end VERL precision check.

1. **CPU unit:** fixed-tensor `base + scale * B @ A`, zero-initialized B
   identity, frozen-base/no-base-grad invariant, config aliases, invalid
   rank/target rejection, and PEFT config/tensor consistency.
2. **Distributed primitive:** TP=1 reference versus TP=2 for column, row and
   SP paths; check forward, input grad, A grad, B grad, and shard/export
   reconstruction.  For MoE, check EP local-expert routing and shared/per-
   expert adapter semantics.
3. **Backend matrix:** one real optimizer step and save/load/resume for each
   of `dist_opt`, FSDP2 and M-FSDP, asserting no frozen tensor is in an
   optimizer group and that the adapter update equals a single-rank reference.
4. **Model topology:** tiny Qwen3-MoE forward/backward at TP2+PP2+EP2+CP2
   (or documented physical proxy) with LoRA enabled; `rank=0` must match the
   no-LoRA reference.  This is a GPU/Slurm gate, not a login-node test.
5. **Artifact/resync:** PEFT export/import round-trip across different TP
   degree where supported, base-revision mismatch rejection, and a real
   training-to-rollout adapter refresh.  Compare logits before/after reload
   and, where merge is offered, merged versus adapter logits.
6. **VERL E2E:** short SFT and one RL/GRPO policy update with real adapter
   refresh.  Report loss/reward plus trainable/base parameter counts; do not
   treat a rollout without the new adapter as a valid end-to-end result.

## Explicit non-goals and risks

* No generic PEFT monkey-patching of arbitrary model graphs; each supported
  MLite model declares its target mapping.
* No QLoRA, DoRA, multi-adapter routing, AdaLoRA, or quantized optimizer in the
  first increment.
* Do not confuse MLA low-rank architecture dimensions with PEFT LoRA.
* The present baseline's name-based freezing must remain constrained to model
  construction where names are controlled; a future generalization should use
  an adapter-parameter registry rather than broad string matching.
* Fused QKV/FC1 semantics and MoE shared-adapter semantics are externally
  observable in PEFT export.  Their mapping needs per-model tests, not only a
  `target_modules` string test.

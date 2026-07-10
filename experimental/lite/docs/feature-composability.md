# Feature Composability in Megatron Lite

This document is a source-based architecture study, not an implementation or a
performance claim. It studies how Muon, Megatron-FSDP (M-FSDP), CUDA Graphs,
FP8, THD packed sequences, static context parallelism (CP), and dynamic CP can
coexist without creating a model-by-feature implementation matrix. No GPU code
was run for this study.

The upstream reference is NVIDIA Megatron-LM `fd1121b8` (`dev`, observed on
2026-07-10). The MLite baseline is `69ea18d0`. Three MLite implementations that
were still on delivery branches at the time of the study are called out as
**in flight**, never as baseline behavior:

- standalone M-FSDP at `e4e878f6`;
- compact Muon + dist-opt lowering at `62404d4a`; and
- closed Hopper blockwise profiles at `10280ae2`.

## Decision summary

The seven names are not seven peer user options. The following ownership
domains are useful for reading the source and locating coordination seams; they
are not a proposed public configuration schema:

| Domain | Features | State it owns |
| --- | --- | --- |
| Optimizer algorithm | AdamW, Muon | parameter selection, update math, optimizer state schema |
| Parameter backend | replicated, dist-opt, FSDP2, M-FSDP | parameter materialization, gradient reduction, master/state sharding, offload |
| Precision | BF16, FP8 recipe and weight storage | compute contexts, quantizers, amax/scale state, compute/master dtype contract |
| Sequence layout | BSHD, THD, static CP, dynamic CP | token layout, positions/masks, `PackedSeqParams`, per-microbatch CP group |
| Execution capture | eager, partial/layer CUDA Graph, full iteration, optimizer graph | static addresses, graph signature, warmup/replay order, graph-safe hooks |

The support and reuse units are deliberately different:

1. A **primitive** is an independently validated unit of math, parallel
   semantics, or runtime behavior. Reusing one does not promise that it composes
   with every other primitive.
2. A **`(model_name, impl)` pair** is the unit of curated assembly and
   end-to-end support. One model may have several implementations, but MLite
   should not create an implementation for every feature-product cell.
3. An implementation config contains only a small set of policies that change
   the user contract, such as FP8 strategy, optimizer strategy, or
   recompute/offload policy. THD layout, CUDA Graphs, fusion, and overlap must
   not become five independent public feature axes merely because the source
   analysis separates their owners.
4. A semantics-preserving refinement such as CUDA Graph capture, fusion, or
   overlap becomes part of the implementation's default path only after
   correctness and performance qualification over its stated envelope. Its
   `off` path is a correctness oracle, A/B control, and debugging aid rather
   than an ordinary user-selected implementation mode.

Support is therefore explicit and selective. A needed combination is assembled
in a concrete model implementation and qualified end to end; an unneeded or
unqualified combination remains unsupported even when no mathematical hard
conflict appears in the pairwise analysis.

## Classification used below

- **Hard conflict** means the mechanisms cannot preserve both contracts at the
  same time. No permanent hard conflict was found among these seven features.
- **Needs coordination** means the features share state, a process group, a
  shape/address contract, or an ordering boundary.
- **Orthogonal** means neither feature needs to inspect the other at the
  surveyed implementation stage. It is local evidence, not a timeless or
  transitive support promise. Both may still depend on a generic lower layer,
  such as the parameter backend.

“Current guard” is reported separately. A source-level rejection is evidence
of the current implementation envelope, not proof of a mathematical conflict.

## Pairwise interaction matrix

Every one of the 21 pairs is covered here. “MCore” means the surveyed
`fd1121b8` snapshot; “MLite branch” means an explicitly identified in-flight
implementation.

| Pair | Mechanism classification | Interaction and source evidence | Current envelope |
| --- | --- | --- | --- |
| Muon × M-FSDP | **Needs coordination** | Muon orthogonalizes a logical 2-D matrix, including TP-aware Newton-Schulz ([`emerging_optimizers.py:185-209`][mcore-muon-ns]); a sharding backend owns when that matrix is materialized. The surveyed MLite M-FSDP code accepts an optimizer factory after main-parameter grouping ([`fused_ops.py:21-34`][mlite-mfsdp-factory]), but a concrete Muon integration still needs an explicit gather/NS/reshard or distributed-NS contract. | MCore rejects emerging optimizers with both Torch-FSDP2 and M-FSDP ([`arguments.py:1883-1890`][mcore-emerging-fsdp-guard]). The MLite branch exposes a construction seam but does not prove Muon through it. |
| Muon × CUDA Graph | **Needs coordination** | Forward/backward graphing can remain optimizer-agnostic, but optimizer graphing is a separate capability. MCore only marks Adam `capturable` ([`optimizer/__init__.py:545-553`][mcore-adam-capturable]) and wraps `optimizer.step` separately from the FWD/BWD schedule ([`training.py:3871-3890`][mcore-graph-wrappers]). | FWD/BWD graphing is not inherently blocked by Muon. The current optimizer-graph implementation is Adam-only; Muon optimizer capture is unqualified. |
| Muon × FP8 | **Needs coordination** | FP8 compute may use a high-precision authoritative master that Muon updates. The contract must identify compute storage, main gradient, master owner, and gather dtype ([`contract.py:59-80`][mlite-parameter-contract]). Muon must never orthogonalize an opaque FP8 storage shard. | MCore explicitly rejects FP8/FP4 *parameter gather* for LayerWise Muon while allowing FP8 compute with parameters persisted in BF16 ([`arguments.py:1896-1900`][mcore-muon-fp8-guard]). |
| Muon × THD | **Orthogonal** | THD changes token layout and attention metadata ([`packed_seq_params.py:11-30`][mcore-packed-seq]); Muon selects and updates parameters. Neither implementation reads the other's state. | No pair-specific lowering is justified at this stage. This is not a support claim: a selected model implementation must still qualify the complete lifecycle, and adding a refinement such as graph capture invalidates this scoped orthogonality evidence. |
| Muon × static CP | **Needs coordination** | Muon math only needs the parameter's TP metadata, but its parameter backend must reduce gradients over the correct DP×CP ownership domain. MCore sizes LayerWise/DistOpt layouts from `dp_cp` ([`training.py:1996-2015`][mcore-muon-layout]); MLite represents `dp_cp_group` explicitly ([`state.py:13-35`][mlite-parallel-state]). | Not a Muon algorithm conflict. The optimizer lowering, not Muon, must own CP-aware gradient and master-parameter placement. |
| Muon × dynamic CP | **Needs coordination** | Dynamic CP changes the CP subgroup per microbatch but keeps a fixed outer DP×CP rank pool ([`model_parallel_config.py:83-103`][mcore-dynamic-cp-config]). Muon should remain unaware; the runtime must guarantee per-token loss normalization and reduce the accumulated gradient over the stable optimizer ownership domain. | MCore does not reject this pair directly, but dynamic CP requires per-token loss and its scheduler ([`arguments.py:1560-1585`][mcore-dynamic-cp-guards]). No dedicated Muon qualification was found. |
| M-FSDP × CUDA Graph | **Needs coordination** | Both own parameter addresses and hook timing. MCore enables a graph-safe M-FSDP mode and moves full-iteration all-gather out of the wrong phase ([`training.py:2346-2361`][mcore-mfsdp-cg-setup]). TE graph capture later reinstalls parameter pre-forward hooks manually ([`training.py:4045-4056`][mcore-cg-manual-hooks]). | Supported only for the qualified scopes. Partial TE capture still requires persistent buffers and correct hook phase; “CUDA Graph on” cannot imply every scope works. |
| M-FSDP × FP8 | **Needs coordination** | M-FSDP owns high-precision masters, low-precision materialization, all-gather buffers, and FP8 transpose-cache lifetime. The upstream implementation has explicit FP8 buffer/quantization paths ([`param_and_grad_buffer.py:3112-3261`][mcore-mfsdp-fp8-buffers]). | MCore supports selected combinations. MLite's in-flight blockwise contract fixes FP32 master/state and BF16 parameter all-gather ([`hopper_blockwise.py:52-61`][mlite-blockwise-param-contract]); a backend must consume that record rather than infer from tensor classes. |
| M-FSDP × THD | **Orthogonal** | M-FSDP shards parameters and optimizer state; THD owns input packing and attention boundaries. MLite's shared packer produces model forward kwargs before optimizer behavior is involved ([`protocol_utils.py:54-88`][mlite-thd-forward]). | No direct pair guard was found. A concrete implementation still needs end-to-end qualification; adding CUDA Graph or dynamic CP invalidates the pair-only evidence and requires requalification. |
| M-FSDP × static CP | **Needs coordination** | CP ranks participate in parameter-gradient ownership. MLite FSDP2 already builds its mesh from `dp_cp_group` ([`wrap.py:80-95`][mlite-fsdp2-dpcp]); the in-flight M-FSDP branch similarly selects dense DP from `dp_cp_group` ([`config.py:97-112`][mlite-mfsdp-groups]). | MCore warns that TP/CP and FSDP want different `CUDA_DEVICE_MAX_CONNECTIONS` settings on Hopper and earlier ([`arguments.py:1636-1651`][mcore-fsdp-cp-warning]). Correctness can compose; performance policy needs an explicit choice. |
| M-FSDP × dynamic CP | **Needs coordination** | Dynamic CP reassigns ranks between data and context work per microbatch, while M-FSDP assumes a stable parameter-shard and gradient-reduction ownership domain. A viable design must separate the stable parameter group from the selected attention CP subgroup. | MCore currently rejects the combination with the exact guard “Dynamic context parallelism not supported with Megatron FSDP” ([`arguments.py:1560-1566`][mcore-dynamic-cp-guards]). This is a current lowering conflict, not evidence of permanent impossibility. |
| CUDA Graph × FP8 | **Needs coordination** | This is a real shared-state interaction. Local replay restores the FP8 recipe/group, controls first-microbatch weight-cache updates, and performs delayed-scaling reduction after backward ([`cuda_graphs.py:620-680`][mcore-cg-fp8-replay]). TE capture passes recipe, weight caching, per-layer enablement, and the TP/CP amax group ([`cuda_graphs.py:2542-2580`][mcore-te-cg-fp8]). | Supported for qualified TE/MCore scopes and recipes. A model implementation's graph setup must consume its selected precision strategy; it must not invent a second FP8 switch. |
| CUDA Graph × THD | **Needs coordination** | Graph replay requires a stable tensor/non-tensor signature; THD carries variable `cu_seqlens`, maxima, and optional metadata. MCore requires max padding for THD graphing ([`transformer_config.py:3173-3186`][mcore-thd-cg-guard]) and derives a bounded microbatch count ([`cuda_graphs.py:2291-2365`][mcore-thd-cg-bound]). | Bounded/padded THD is supportable. An implementation that cannot bound THD must report graph capture as not applicable with a reason, or fail loudly if it had declared the input applicable; silent truncation or silent eager replay is invalid. |
| CUDA Graph × static CP | **Needs coordination** | CP group identity and local shapes become part of the graph signature. FP8 graph capture also includes CP in the amax reduction group ([`cuda_graphs.py:2571-2578`][mcore-te-cg-fp8]). | Static CP can compose when the group and shapes do not change between capture and replay. |
| CUDA Graph × dynamic CP | **Needs coordination** | A per-microbatch CP subgroup changes group identity, local shape, and THD metadata. The current TE helper binds `dp`, `dp_cp`, and `pp` groups at construction ([`cuda_graphs.py:1822-1837`][mcore-te-helper-groups]); therefore dynamic CP needs a graph bank/signature keyed by local CP size and group, not one global graph. | The argument layer requires THD max padding when graphing dynamic packing ([`arguments.py:1620-1634`][mcore-thd-cg-args]), but the surveyed helper has no complete per-microbatch graph-bank contract. Treat the combination as unsettled, not silently supported. |
| FP8 × THD | **Orthogonal** | Precision coverage is attached to semantic GEMM sites, whereas THD changes sequence representation. The in-flight MLite FP8 strategy names attention projections, dense MLPs, and MoE experts without referring to BSHD/THD ([`hopper_blockwise.py:28-49`][mlite-blockwise-sites]). | Orthogonal only at the surveyed pre-graph stage. Kernel-specific shape support still belongs to primitive evidence, and adding graph capture or another refinement requires the model implementation to requalify the complete path. |
| FP8 × static CP | **Needs coordination** | Delayed-scaling FP8 may reduce amax over TP×CP, while parameter storage/all-gather remains optimizer-owned. MCore exposes `tp_only_amax_red` and explicitly documents the TP-CP domain ([`transformer_config.py:644-651`][mcore-fp8-cp-config]). | Supported for qualified recipes/operators. MLite's first in-flight Hopper strategies deliberately require `cp=1` ([`config.py:97-112`][mlite-blockwise-current-guards]); that is an initial strategy boundary, not a general FP8 limitation. |
| FP8 × dynamic CP | **Needs coordination** | Dynamic CP selects a per-microbatch CP group inside attention ([`transformer_engine.py:1797-1818`][mcore-te-dynamic-cp]); any non-local FP8 scale reduction must use a group consistent with that execution. Blockwise local scales reduce, but do not erase, operator eligibility and parameter-ownership concerns. | No complete MLite integration exists. A delayed-scaling strategy needs dynamic group handling; a local-scale strategy still needs explicit coverage evidence. |
| THD × static CP | **Needs coordination** | The two features intentionally meet at one boundary: THD owns packing and `cu_seqlens`; CP owns zigzag rank partition. MLite performs the split in the shared protocol-forward helper ([`thd.py:192-230`][mlite-thd-cp-split]), matching the `primitive.module.thd`/`primitive.parallel.cp` skill contract. | MLite baseline supports static THD+CP paths. The split must not be duplicated inside model functions. |
| THD × dynamic CP | **Needs coordination** | Dynamic CP depends on packed variable-length work and carries `local_cp_size` plus `cp_group` in `PackedSeqParams` ([`packed_seq_params.py:18-30`][mcore-packed-seq]). Its scheduler is forced to `default_dynamic_cp` ([`model_parallel_config.py:502-515`][mcore-dynamic-cp-validation]). | MCore has an implementation. MLite baseline has no dynamic-CP groups or scheduler and reports a fixed `cp_range` ([`runtime.py:242-259`][mlite-fixed-cp-range]). |
| static CP × dynamic CP | **Needs coordination** | Dynamic CP does not run beside an unrelated second CP topology: the configured DP×CP domain is the maximum rank pool, and runtime chooses subgroup sizes from it ([`parallel_state.py:921-948`][mcore-dynamic-cp-groups]). | Static `context_parallel_size` remains the maximum geometry. A concrete implementation must keep one owner for CP groups and treat dynamic CP as a sequence-layout strategy, not a second parallel dimension. |

### Matrix conclusion

The count is **18 needs-coordination, 3 orthogonal, and 0 permanent hard
conflicts**. The absence of a hard-conflict cell is not permission to enable all
cross products. Several pairs are actively rejected by current code, and even
an orthogonal pair is supported only after a selected `(model_name, impl)` has
qualified the full path in which it appears.

## How Megatron `dev` composes the features

### Configuration and validation flow

Megatron uses a layered but partly duplicated flow:

1. `arguments.py` parses legacy and current flags, normalizes CUDA Graph enums,
   derives DP/CP sizes, mutates implied features, and rejects CLI combinations.
   Examples include CUDA Graph migration ([`arguments.py:675-712`][mcore-cg-args]),
   FSDP implying dist-opt ([`arguments.py:1239-1258`][mcore-mfsdp-args]), and
   emerging optimizer routing ([`arguments.py:1872-1909`][mcore-muon-args]).
2. `TransformerConfig` owns model/compute behavior and repeats library-level
   validation for programmatic callers. CUDA Graph backend and module coverage
   are typed fields ([`transformer_config.py:1018-1076`][mcore-cg-config]); its
   validator checks graph scope, MoE shapes, recompute, offload, and THD bounds
   ([`transformer_config.py:2788-2953`][mcore-cg-validation]).
3. `DistributedDataParallelConfig` owns parameter/gradient communication and
   M-FSDP graph-mode fields ([`distributed_data_parallel_config.py:84-121`][mcore-ddp-config],
   [`distributed_data_parallel_config.py:218-245`][mcore-ddp-mfsdp-config]).
4. `OptimizerConfig` owns algorithm/state behavior, including Muon, FP8-aware
   optimizer fields, and separate optimizer graphing ([`optimizer_config.py:258-340`][mcore-optimizer-config],
   [`optimizer_config.py:401-436`][mcore-optimizer-cg-config]).
5. The training assembler copies validated fields into each owner and installs
   runtime wrappers and hooks.

This is more disciplined than scattering every feature through model classes,
but it is still one concrete implementation's procedural assembly and
validation. It is not evidence that the same fields should become MLite-wide
user axes or a universal capability system.

### Supported and explicitly guarded combinations

| Combination | `dev` result | Exact evidence |
| --- | --- | --- |
| Muon + M-FSDP/FSDP2 | Rejected | “Emerging optimizer does not support ...” ([`arguments.py:1887-1890`][mcore-emerging-fsdp-guard]) |
| Muon + FP8 compute + BF16 parameters | Allowed by the guard | Error text rejects only FP8/FP4 parameter gather and recommends blockwise/MXFP8 compute with BF16 parameters ([`arguments.py:1896-1900`][mcore-muon-fp8-guard]) |
| M-FSDP + Adam/SGD | Supported | M-FSDP forces distributed optimizer and rejects other algorithms ([`arguments.py:1239-1258`][mcore-mfsdp-args]) |
| M-FSDP + CUDA Graph | Scope-dependent support | graph-safe mode and full-iteration all-gather relocation ([`training.py:2351-2360`][mcore-mfsdp-cg-setup]) |
| Dynamic CP + M-FSDP | Rejected | exact dynamic-CP guard ([`arguments.py:1560-1566`][mcore-dynamic-cp-guards]) |
| THD/dynamic packing + CUDA Graph | Bounded support | max-alignment guard ([`arguments.py:1620-1634`][mcore-thd-cg-args]) |
| Full-iteration graph + per-module scopes | Rejected | exact scope guard ([`arguments.py:2126-2128`][mcore-full-graph-scope]) |
| Full MoE graph + dropless/dynamic shapes | Rejected | requires drop-and-pad capacity ([`transformer_config.py:2854-2868`][mcore-cg-validation]) |
| Optimizer graph | Separate Adam path | Adam is capturable and the wrapper encloses `step()` only ([`optimizer/__init__.py:545-553`][mcore-adam-capturable], [`training.py:3885-3890`][mcore-graph-wrappers]) |

### The actual CUDA Graph × FP8 × dist-opt interaction

This combination is not implemented by three independent booleans:

1. The TE graph helper is constructed with the already-built model and
   optimizer ([`training.py:3968-3977`][mcore-te-helper-build]).
2. Capture consumes the FP8 recipe, layer enablement, weight-cache policy, and
   TP/CP amax group ([`cuda_graphs.py:2542-2580`][mcore-te-cg-fp8]).
3. Capture warmup mutates gradients and optimizer state, so the helper clears
   model buffers, optimizer gradients, MoE metrics, and temporary tensors
   before training resumes ([`cuda_graphs.py:2611-2656`][mcore-te-reset]).
4. Dist-opt/M-FSDP pre-forward all-gather hooks cannot remain inside the wrong
   graph phase. After capture, the training loop re-enables hooks and asks each
   graphed layer to install manual pre-forward hooks
   ([`training.py:4045-4056`][mcore-cg-manual-hooks],
   [`cuda_graphs.py:2712-2720`][mcore-helper-manual-hooks]).
5. Optimizer graphing, if requested, is still a separate graph around
   `optimizer.step()`.

This is the strongest evidence for explicit coordination interfaces in MLite:
precision state, parameter hooks, and graph state have different owners, but
the runtime assembler orders their hand-off.

### What Megatron does cleanly

- Typed dataclass fields separate transformer, distributed, and optimizer
  ownership.
- CUDA Graph backend, coverage, and optimizer graph have distinct internal
  ownership rather than one overloaded boolean.
- Important unsupported combinations fail before training with actionable
  error text.
- The production training assembler owns hook/capture ordering; graph support
  is not a test-only wrapper.
- Dynamic CP carries its subgroup in per-microbatch sequence metadata instead
  of mutating one global CP group.

### Where Megatron becomes muddy

- CLI validation and dataclass validation repeat many rules. They can drift;
  for example, the dynamic-CP CLI block still checks the deprecated
  `enable_cuda_graph` field while later code validates the normalized
  `cuda_graph_impl` through THD padding rules
  ([`arguments.py:1560-1563`][mcore-dynamic-cp-guards],
  [`arguments.py:1620-1634`][mcore-thd-cg-args]).
- Validation mutates other feature settings: M-FSDP turns on `use_custom_fsdp` and
  distributed optimizer, while an emerging optimizer turns
  `use_distributed_optimizer` back off in favor of LayerWise
  ([`arguments.py:1239-1258`][mcore-mfsdp-args],
  [`arguments.py:1872-1886`][mcore-muon-args]). The final state depends on
  validation order.
- Eligibility is often inferred from broad flags and parameter-name patterns.
  The Muon path still tags QKV and experts by names
  ([`optimizer/__init__.py:777-799`][mcore-muon-name-routing]). That is practical
  in MCore's controlled model tree but is the wrong boundary for MLite
  primitives.
- Central assertions describe current support, but they do not explain which
  component owns the missing behavior. This encourages adding another
  cross-feature `if` instead of keeping the concrete implementation boundary
  reviewable.

MLite should borrow concrete ownership lessons, fail-loud guards, and production
hook ordering. It should not copy the full validator or generalize it into an
MLite-wide dispatcher.

## MLite's current state and composition rules

### Rules extracted from every skill category

| Skill category | Composition invariant used by this design |
| --- | --- |
| `basic` | Pick the strongest checkable reference, freeze unrelated variables, use the smallest falsifiable proxy, and require end-to-end evidence before delivery ([`basic.constitution`][skill-constitution], [`basic.find_reference`][skill-reference]). |
| `primitive` | Every primitive declares math/parallel semantics, shape/dtype/rank rules, state/process-group ownership, valid and invalid combinations, and adjacent composition tests ([`primitive.principle`][skill-principle], [`primitive.contract`][skill-primitive-contract]). Selection must reject hidden coupling ([`primitive.select_for_compose`][skill-select]). |
| `model-compose` | A model selects already-validated primitives and preserves runtime/model/primitive boundaries; model precision is validated after composition ([`model_compose.build_model`][skill-build-model]). |
| `application` | Runtime success is not correctness evidence; build/train/save/load and independent precision comparison remain required ([`application.runtime`][skill-runtime]). |
| `perf` | Performance evidence is invalid without a stable workload and precision evidence; fusion must name coupled primitives, retain an unfused reference, and rerun precision plus measurement after each candidate change ([`perf.measure`][skill-measure], [`perf.fusion`][skill-fusion], [`perf.optimize`][skill-optimize]). |
| `insight` | Add a primitive progressively from principle to proxy to distributed composition to performance, and split work rather than widening one procedure indefinitely ([`insight.progressive_primitive_design`][skill-progressive], [`insight.scope_control`][skill-scope]). |

These skills establish two different proof levels. A primitive proves its own
invariants and selected adjacent composition; `model_compose.build_model` then
requires a model-level precision result after concrete assembly. Neither proof
level promises arbitrary combinations. Unsupported combinations stay explicit
rather than taking an unmeasured fallback.

### Baseline inventory

| Surface | Baseline behavior | Composition consequence |
| --- | --- | --- |
| Registry | The runtime resolves a concrete protocol from `(model_name, impl)` and fails when that pair is not registered ([`registry.py:20-68,134-140`][mlite-model-registry]). | The existing pair is the natural support-promise boundary. Multiple curated impls fit without creating one per feature combination. |
| Common config | `MegatronLiteConfig` already owns `model_name`, `impl`, parallel and optimizer records, while implementation-specific keys enter model-local `impl_cfg` ([`config.py:24-47`][mlite-runtime-config]). | Keep `ImplConfig` narrow: only policies that alter this implementation's user contract belong there. Do not mirror every internal primitive or refinement as a public key. |
| Build lifecycle | Runtime resolves the protocol from `(model_name, impl)`, builds its model/optimizer, loads weights, executes post-load replacement, reloads optimizer masters, then returns the handle ([`runtime.py:170-259`][mlite-build]). | A qualified refinement that depends on final modules, masters, or hooks must be installed by this concrete build path after those objects exist and before the first production step. |
| THD + CP | Shared protocol code pads and CP-splits THD at the immediate forward boundary ([`protocol_utils.py:54-100`][mlite-thd-forward]); primitive code owns zigzag split/reconstruction ([`thd.py:71-147`][mlite-thd-zigzag]). | This is the good precedent: model protocols call a shared primitive; attention consumes already-local metadata. |
| Static groups | `ParallelState` creates fixed TP/EP/CP/PP/DP/DP×CP groups ([`state.py:57-155`][mlite-static-groups]). | Dynamic CP cannot be represented by the current state alone. Runtime also reports a fixed `cp_range`, confirming that it is not implemented. |
| FSDP2 | FSDP2 owns `fully_shard`, mixed precision, prefetch, and DP×CP mesh; its optimizer currently accepts Adam/AdamW only ([`wrap.py:138-194`][mlite-fsdp2-wrap], [`optimizer.py:285-340`][mlite-fsdp2-adam]). | Optimizer algorithm and parameter backend are not yet independently selectable on this path. |
| dist-opt | The wrapper constructs MCore DDP/optimizer and explicit process groups, but the model protocol still passes a `model_name` that is stored and never consumed ([`megatron_wrap.py:198-240`][mlite-distopt-model-name]). | This is dead interface and leaked upper-layer knowledge. Do not generalize it into feature or model allowlists; remove it in the implementation that next touches the seam. |
| FP8 remnants | Models contain FP8 branches, but the surveyed Qwen3-MoE production protocol passes `fp8=False` ([`qwen3_moe/protocol.py:174-186`][mlite-qwen-fp8-off]). | Test-fed, production-unreachable branches are debt, not support. A selected FP8 strategy must replace or activate a production path and remove the dead alternative. |
| CUDA Graph / dynamic CP | No baseline config, controller, capture/replay, dynamic subgroup registry, or scheduler exists under `megatron.lite`. | Forwarding MCore flags would not implement either behavior. They need concrete integration and qualification in a selected model implementation before being declared applicable. |

The static layering test already enforces the desired direction and explicitly
checks that primitives are model-name agnostic
([`test_layering_contracts.py:190-235`][mlite-layering-test]). New model
implementations and refinements must preserve that direction.

### In-flight precedents and what is reusable

#### Closed Hopper FP8 strategies

The in-flight implementation has the strongest reusable contract:

- `PrecisionImplementation` is immutable and combines a closed recipe, typed
  semantic coverage, and one `ParameterContract`
  ([`contract.py:59-80`][mlite-parameter-contract]).
- Only two named Hopper strategies exist; recipe, target, and storage mode cannot
  be arbitrarily multiplied ([`hopper_blockwise.py:28-49`][mlite-blockwise-sites],
  [`hopper_blockwise.py:131-159`][mlite-blockwise-profiles]).
- Runtime rejects ad-hoc precision keys and unsupported adjacent features
  before model allocation ([`config.py:82-121`][mlite-blockwise-current-guards]).
- Coverage is typed by semantic site and object identity; names are diagnostic,
  not selection rules ([`hopper-blockwise-fp8-contract.md:204-227`][mlite-fp8-coverage-design]).

Reusable within implementations that select this FP8 user policy: immutable
contracts, exact coverage sealing, separate compute/master ownership, closed
strategies, and fail-loud preflight.

Not reusable as a universal abstraction: `ParameterContract` alone does not
describe graph signatures, sequence layout, optimizer algorithm math, or a
model implementation's support envelope. It must remain local to the concrete
precision/parameter integration that consumes it.

#### Standalone M-FSDP optimizer factory

The in-flight M-FSDP implementation narrows its own config and process groups
([`config.py:26-112`][mlite-mfsdp-config]) and accepts an optimizer factory over
already-created parameter groups ([`fused_ops.py:21-34`][mlite-mfsdp-factory]).

Reusable lesson: keep parameter sharding/communication separate from update
math, and supply process groups explicitly rather than reading model names or
global MCore state. The surveyed factory is an implementation fact, not a
recommendation for an MLite-wide optimizer factory.

Limit: “already-sharded parameters” is not sufficient for Muon. A concrete
Muon+M-FSDP implementation must explicitly supply gather/NS/reshard or
distributed NS for the logical matrix. Otherwise the factory is syntactically
pluggable but semantically incomplete.

#### Compact Muon lowering

The in-flight Muon work improves the boundary in two ways:

- the metadata adapter knows only a caller-provided expert classifier and
  preserves module-owned TP/QKV metadata ([`muon_routing.py:13-32`][mlite-muon-routing]);
- one build path owns validation, config lowering, metadata tagging, DDP layout,
  and optimizer construction, and rejects partial direct construction
  ([`megatron_wrap.py:107-128`][mlite-muon-lowering],
  [`megatron_wrap.py:272-345`][mlite-muon-compose]).

Reusable within a selected implementation: a backend-specific lowering
transaction that validates before mutation and consumes semantic metadata.

Limit: the current lowering remains pinned to one MCore contract and initializes
residual global MCore state. It is a backend adapter, not the common feature
model.

## Architecture alternatives

### Alternative A: Megatron-style centralized feature validation

Add every internal feature field to `MegatronLiteConfig.__post_init__` and grow
pairwise assertions there. This starts small, but the common config soon knows
model construction, optimizer hooks, per-microbatch metadata, and every current
feature list. It becomes a hidden all-in-one implementation and remains
order-dependent. Reject this alternative.

### Alternative B: one global combination registry

Expose names such as `muon_mfsdp_fp8_thd_cp_graph` and make each name a tested
bundle. This hides the Cartesian product in names without removing it from
implementation or evidence. The two closed Hopper FP8 strategies are valid
because they freeze tightly coupled precision choices that change the user
contract; they do not justify system-wide combination profiles. Reject this
alternative.

### Alternative C: curated model implementations with qualified refinements

This is the recommendation, and it already matches MLite's registry boundary:

| Unit | Responsibility | What it does not promise |
| --- | --- | --- |
| Primitive | Preserve one math/parallel/runtime contract and its own validation evidence. | Arbitrary composition with every other primitive. |
| `(model_name, impl)` | Select and explicitly assemble concrete primitives; own forward, backward, grad finalization, optimizer step, save/load, and performance qualification. | One implementation per Cartesian-product cell. |
| `ImplConfig` policy | Represent a small choice that changes this implementation's user contract, such as FP8, optimizer, or recompute/offload strategy. | Independent switches for every layout, kernel, capture, fusion, and overlap detail. |
| Qualified refinement | Improve the concrete implementation without changing its semantics, over a measured applicability envelope. | A normal user-facing mode or a silent fallback path. |

The implementation may keep narrow records at real ownership boundaries, such
as the in-flight FP8 `ParameterContract`, but MLite should not introduce a
global capability spec, compiler, abstract factory, or manifest that decides
all models and features. A little explicit assembly duplicated across model
implementations is cheaper and more reviewable than a shared helper that
branches on `impl`, model names, or feature lists.

### What belongs in `ImplConfig`

The test is whether a choice changes the user's contract for that specific
implementation:

- FP8 recipe/storage strategy changes numerical and checkpoint behavior, so a
  closed choice such as `hopper_blockwise_bf16_weight` or
  `hopper_blockwise_fp8_weight` may be a user policy.
- Optimizer strategy and recompute/offload policy change update or memory
  behavior, so they may be user policies within the implementation's supported
  set.
- THD packing, a particular fusion, overlap schedule, or CUDA Graph capture
  scope is normally implementation machinery. It should not become an
  independent public axis merely because it has a separate internal owner.

This rule does not require deleting every existing `impl_cfg` field in this
research task. It is the boundary for future implementation evolution.

### Qualification and defaulting of refinements

A refinement enters the default implementation only when all of the following
hold over a stated envelope:

1. it preserves the implementation's semantics through forward, backward,
   gradient finalization, optimizer step, and save/load;
2. it is performance-qualified with a fixed workload, warmup, repeats, and
   memory evidence, and is monotonically no worse within that envelope; and
3. its applicability can be decided and reported without asking a primitive to
   know the model or implementation name.

CUDA Graph capture, fusion, and communication overlap follow this rule. Once
qualified, they are default implementation behavior. Their `off` path remains
available only as a correctness oracle, controlled A/B, or debugging tool. A
new refinement invalidates the old composition proof: all lifecycle and
performance evidence must be rerun for the refined stage.

Applicability is observable per implementation and stage:

- `enabled`: the input is inside the qualified envelope and the refinement ran;
- `partial`: only a documented subgraph or stage is qualified, and the boundary
  is reported; or
- `not-applicable`: the input is outside the envelope, with an actionable
  reason.

If an input was declared applicable and capture or setup then fails, execution
must raise an error. Silently continuing in eager mode would turn a qualified
default into unmeasured behavior.

## Initial implementation boundaries and handoff

The pairwise matrix is a risk map for implementations that are actually needed;
it is not a backlog to populate. Near-term work should use these boundaries:

| Workstream | Correct placement | Surveyed or initial boundary | Do not build |
| --- | --- | --- | --- |
| Hopper blockwise FP8 | Closed user policy inside the selected `ImplConfig`; reuse its concrete precision and parameter contracts. | The in-flight strategies cover attention projection, dense MLP, and experts; require CP=1 and PP=1 and exclude CUDA Graph/dynamic CP in their first contract. FP8-weight storage needs stronger checkpoint/master evidence. | A global recipe × target × storage matcher or a cross-feature profile. |
| Muon + M-FSDP | Explicit assembly in each model implementation that needs it, reusing stable Muon math and parameter-sharding primitives. | Supply bounded full-matrix gather → NS → reshard or distributed NS; start without optimizer graph or FP8 parameter gather; static CP waits for DP×CP reduction evidence and dynamic CP remains unsupported. | A universal optimizer factory or an implementation merely because this matrix has no hard-conflict cell. |
| THD + static CP | Existing shared THD/CP primitives called explicitly by model protocol forward paths. | Preserve one owner for packing and zigzag partition; validate the selected model implementation end to end. | Model-local copies of split logic or a helper that dispatches on model names. |
| Dynamic CP | Add only to a model/runtime implementation with a demonstrated workload need. | Requires max-aligned THD, per-token loss, a dynamic group registry, and scheduler. M-FSDP and CUDA Graph remain not applicable until separately integrated and qualified. | A second independent CP dimension or a global sequence profile. |
| CUDA Graph | Semantics-preserving performance refinement of a concrete implementation. | A first qualification may be partial and narrow, for example fixed-shape BSHD attention with eager unselected regions. FP8, THD, PP, M-FSDP hooks, and dynamic CP expand the envelope only after full requalification. | A normal `cuda_graph=on/off` user policy, a named global graph profile, or silent eager fallback after declared-applicable capture failure. |

The current BF16/eager implementation remains the correctness reference while
graph support is unqualified. It is not a second product profile. Likewise,
Muon×THD and M-FSDP×THD need no pair-specific lowering, but that observation
does not admit either pair until a selected implementation has end-to-end
evidence.

## Implementation and review guardrails

1. **No model names below model composition.** Primitive selection uses semantic
   contracts. Paths and names are diagnostics only.
2. **No dead compatibility surface.** Production-unreachable branches, exports,
   factories, or config fields must be removed; tests do not make them live.
3. **No hidden all-in-one helper.** Share stable concrete primitives, subgraphs,
   and model-independent runtime code. If a helper branches on implementation,
   model, or feature lists, move that assembly back to the concrete model
   implementation.
4. **No silent fallback.** Report `enabled`, `partial`, or `not-applicable` with
   reasons. A declared-applicable setup failure is an error, not eager success.
5. **One owner per state item.** FP32 master, quantizer state, CP group, graph
   buffers, and optimizer state each have exactly one owner.
6. **Validate before mutation; refine after construction.** Backend assembly
   validates its concrete assumptions before wrapping modules; graph setup waits
   until weights, masters, and hooks are final.
7. **Keep graph scopes separate.** Forward/backward graph support never implies
   optimizer graph support.
8. **Treat orthogonality as scoped evidence.** It belongs to one implementation
   stage. Any new refinement invalidates it and triggers full requalification.
9. **Evidence follows the MLite skills.** Primitive evidence remains independent;
   a supported `(model_name, impl)` additionally needs production end-to-end and
   performance evidence.

## Validation plan for later implementations

This study is intentionally zero-GPU. A later implementation should build only
the evidence required by an actual model/workload need:

1. CPU/static: narrow `ImplConfig` policy parsing and rejection, layering tests,
   dead-export scans, and checks that shared helpers contain no model/impl/feature
   dispatch.
2. Primitive proxy: eager BF16 reference for each changed primitive's
   forward/backward/update contract, plus its selected adjacent composition.
3. Concrete implementation: real `build_model` → `protocol.forward` → backward →
   grad finalize → optimizer step → save/load, non-skip and with the strongest
   independent reference available.
4. Refinement qualification: run the same lifecycle with the candidate enabled,
   compare against the `off` oracle, then measure fixed-workload step time,
   throughput, and memory. Record the applicability envelope and status.
5. Interaction expansion: test only demanded pairs/triples from the matrix, with
   unrelated variables frozen. At minimum, any demanded CUDA Graph×FP8×parameter
   backend, THD×dynamic-CP×graph, or Muon×M-FSDP×precision path needs explicit
   hook/state/ownership evidence.
6. Invalidation: after every new capture, fusion, overlap, or other refinement,
   rerun forward through save/load plus performance; old orthogonality and
   end-to-end results do not carry over automatically.

## Source links

### Megatron `dev` (`fd1121b8`)

[mcore-muon-ns]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/emerging_optimizers.py#L185-L209
[mcore-emerging-fsdp-guard]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L1883-L1890
[mcore-adam-capturable]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/__init__.py#L545-L553
[mcore-graph-wrappers]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L3871-L3890
[mcore-muon-fp8-guard]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L1896-L1900
[mcore-packed-seq]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/packed_seq_params.py#L11-L30
[mcore-muon-layout]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L1996-L2015
[mcore-dynamic-cp-config]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/model_parallel_config.py#L83-L103
[mcore-dynamic-cp-guards]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L1560-L1585
[mcore-mfsdp-cg-setup]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L2346-L2361
[mcore-cg-manual-hooks]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L4045-L4056
[mcore-mfsdp-fp8-buffers]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py#L3112-L3261
[mcore-fsdp-cp-warning]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L1636-L1651
[mcore-cg-fp8-replay]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L620-L680
[mcore-te-cg-fp8]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L2542-L2580
[mcore-thd-cg-guard]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_config.py#L3173-L3186
[mcore-thd-cg-bound]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L2291-L2365
[mcore-te-helper-groups]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L1822-L1837
[mcore-thd-cg-args]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L1620-L1634
[mcore-fp8-cp-config]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_config.py#L644-L651
[mcore-te-dynamic-cp]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/extensions/transformer_engine.py#L1797-L1818
[mcore-dynamic-cp-validation]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/model_parallel_config.py#L502-L515
[mcore-dynamic-cp-groups]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/parallel_state.py#L921-L948
[mcore-cg-args]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L675-L712
[mcore-mfsdp-args]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L1239-L1258
[mcore-muon-args]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L1872-L1909
[mcore-cg-config]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_config.py#L1018-L1076
[mcore-cg-validation]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/transformer_config.py#L2788-L2953
[mcore-ddp-config]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/distributed/distributed_data_parallel_config.py#L84-L121
[mcore-ddp-mfsdp-config]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/distributed/distributed_data_parallel_config.py#L218-L245
[mcore-optimizer-config]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/optimizer_config.py#L258-L340
[mcore-optimizer-cg-config]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/optimizer_config.py#L401-L436
[mcore-full-graph-scope]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/arguments.py#L2126-L2128
[mcore-te-helper-build]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/training/training.py#L3968-L3977
[mcore-te-reset]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L2611-L2656
[mcore-helper-manual-hooks]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/transformer/cuda_graphs.py#L2712-L2720
[mcore-muon-name-routing]: https://github.com/NVIDIA/Megatron-LM/blob/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1/megatron/core/optimizer/__init__.py#L777-L799

### MLite baseline (`69ea18d0`)

[mlite-parallel-state]: ../megatron/lite/primitive/parallel/state.py#L13-L35
[mlite-thd-forward]: ../megatron/lite/model/protocol_utils.py#L54-L100
[mlite-fsdp2-dpcp]: ../megatron/lite/primitive/optimizers/fsdp2/wrap.py#L80-L95
[mlite-thd-cp-split]: ../megatron/lite/primitive/parallel/thd.py#L192-L230
[mlite-fixed-cp-range]: ../megatron/lite/runtime/backends/mlite/runtime.py#L242-L259
[mlite-runtime-config]: ../megatron/lite/runtime/backends/mlite/config.py#L24-L47
[mlite-build]: ../megatron/lite/runtime/backends/mlite/runtime.py#L170-L259
[mlite-model-registry]: ../megatron/lite/model/registry.py#L20-L140
[mlite-thd-zigzag]: ../megatron/lite/primitive/parallel/thd.py#L71-L147
[mlite-static-groups]: ../megatron/lite/primitive/parallel/state.py#L57-L155
[mlite-fsdp2-wrap]: ../megatron/lite/primitive/optimizers/fsdp2/wrap.py#L138-L194
[mlite-fsdp2-adam]: ../megatron/lite/primitive/optimizers/fsdp2/optimizer.py#L285-L340
[mlite-distopt-model-name]: ../megatron/lite/primitive/optimizers/megatron_wrap.py#L198-L240
[mlite-qwen-fp8-off]: ../megatron/lite/model/qwen3_moe/lite/protocol.py#L174-L186
[mlite-layering-test]: ../tests/unit/runtime/test_layering_contracts.py#L190-L235

### MLite in-flight source snapshots

[mlite-mfsdp-factory]: https://github.com/ISEEKYAN/Megatron-LM/blob/e4e878f657e0e175d336a45ca903dd74529915d0/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/fused_ops.py#L21-L34
[mlite-mfsdp-groups]: https://github.com/ISEEKYAN/Megatron-LM/blob/e4e878f657e0e175d336a45ca903dd74529915d0/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/config.py#L97-L112
[mlite-mfsdp-config]: https://github.com/ISEEKYAN/Megatron-LM/blob/e4e878f657e0e175d336a45ca903dd74529915d0/experimental/lite/megatron/lite/primitive/optimizers/mfsdp/config.py#L26-L112
[mlite-parameter-contract]: https://github.com/ISEEKYAN/Megatron-LM/blob/10280ae2f6a95c7e6f2f13f4a0ee6ecdf9ddefb8/experimental/lite/megatron/lite/primitive/precision/contract.py#L59-L80
[mlite-blockwise-param-contract]: https://github.com/ISEEKYAN/Megatron-LM/blob/10280ae2f6a95c7e6f2f13f4a0ee6ecdf9ddefb8/experimental/lite/megatron/lite/primitive/precision/hopper_blockwise.py#L52-L61
[mlite-blockwise-sites]: https://github.com/ISEEKYAN/Megatron-LM/blob/10280ae2f6a95c7e6f2f13f4a0ee6ecdf9ddefb8/experimental/lite/megatron/lite/primitive/precision/hopper_blockwise.py#L28-L49
[mlite-blockwise-profiles]: https://github.com/ISEEKYAN/Megatron-LM/blob/10280ae2f6a95c7e6f2f13f4a0ee6ecdf9ddefb8/experimental/lite/megatron/lite/primitive/precision/hopper_blockwise.py#L131-L159
[mlite-blockwise-current-guards]: https://github.com/ISEEKYAN/Megatron-LM/blob/10280ae2f6a95c7e6f2f13f4a0ee6ecdf9ddefb8/experimental/lite/megatron/lite/runtime/backends/mlite/config.py#L82-L121
[mlite-fp8-coverage-design]: https://github.com/ISEEKYAN/Megatron-LM/blob/10280ae2f6a95c7e6f2f13f4a0ee6ecdf9ddefb8/experimental/lite/docs/hopper-blockwise-fp8-contract.md#L204-L227
[mlite-muon-routing]: https://github.com/ISEEKYAN/Megatron-LM/blob/62404d4ab9802d2bd53f20f0ceeff1508bc6f72d/experimental/lite/megatron/lite/primitive/optimizers/muon_routing.py#L13-L32
[mlite-muon-lowering]: https://github.com/ISEEKYAN/Megatron-LM/blob/62404d4ab9802d2bd53f20f0ceeff1508bc6f72d/experimental/lite/megatron/lite/primitive/optimizers/megatron_wrap.py#L107-L128
[mlite-muon-compose]: https://github.com/ISEEKYAN/Megatron-LM/blob/62404d4ab9802d2bd53f20f0ceeff1508bc6f72d/experimental/lite/megatron/lite/primitive/optimizers/megatron_wrap.py#L272-L345

### MLite skills

[skill-constitution]: ../skills/basic/constitution.md
[skill-reference]: ../skills/basic/find-reference.md
[skill-principle]: ../skills/primitive/principle.md
[skill-primitive-contract]: ../skills/primitive/contract.md
[skill-select]: ../skills/primitive/select-for-compose.md
[skill-build-model]: ../skills/model-compose/build-model.md
[skill-runtime]: ../skills/application/runtime.md
[skill-measure]: ../skills/perf/measure.md
[skill-fusion]: ../skills/perf/fusion.md
[skill-optimize]: ../skills/perf/optimize.md
[skill-progressive]: ../skills/insight/progressive-primitive-design.md
[skill-scope]: ../skills/insight/scope-control.md

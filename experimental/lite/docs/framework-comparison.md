# Distributed Training Framework Landscape

This document compares veScale, NVIDIA NeMo AutoModel, NVIDIA NeMo Megatron
Bridge, and Megatron Lite (MLite) across functionality, performance evidence,
and developer experience. The projects have different centers of gravity, so
the goal is not to declare one universal winner. The goal is to identify the
right boundary for MLite and the ideas that are worth adopting.

## Snapshot and evidence policy

The comparison was researched on 2026-07-10 against these public snapshots:

- [veScale `main` at `20cf5c7`](https://github.com/volcengine/veScale/tree/20cf5c7),
  its [legacy documentation](https://volcengine.github.io/veScaleWeb/guide/introduction.html),
  the [2025 eager-SPMD paper](https://arxiv.org/abs/2509.07003), and the
  [2026 veScale-FSDP paper](https://arxiv.org/abs/2602.22437).
- [NeMo AutoModel v0.5.0](https://github.com/NVIDIA-NeMo/Automodel/tree/v0.5.0)
  and its [official documentation](https://docs.nvidia.com/nemo/automodel/latest).
- [Megatron Bridge v0.5.0](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/v0.5.0)
  and its [official documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/).
- MLite at repository commit `69ea18d07`, using the local source, tests, and
  committed benchmark records linked below.

Evidence is classified as follows:

- **Source-backed** means the inspected snapshot contains the implementation or
  a runnable entry point.
- **Documented** means an official project document describes the behavior, but
  this review did not reproduce it.
- **Published** means a project paper or performance page reports a result. It
  is not an independent reproduction.
- **Measured** is reserved for a committed MLite benchmark record with a real
  job identifier, environment, configuration, and result.

No new GPU run was made for this study. In particular, published numbers from
different models, precisions, GPU generations, batch sizes, and parallel layouts
must not be used as a cross-framework ranking.

## Executive view

| Project | Best fit | Strongest differentiator | Main limitation for an MLite user |
| --- | --- | --- | --- |
| veScale | Research and design reference for eager SPMD, topology-stable RNG, distributed checkpointing, and flexible sharding | Single-device semantics over a custom DTensor/DModule stack; the newer public direction adds structure-aware `RaggedShard` FSDP | The old full framework moved to `legacy/`; the current public `main` says only a small piece is open sourced, has no release, and the auto-plan pages still say “Coming Soon” |
| NeMo AutoModel | Day-0 training and fine-tuning of Hugging Face models with PyTorch-native distributed execution | Native Hugging Face model/checkpoint path plus FSDP2/DTensor and a broad recipe/component ecosystem | Day-0 compatibility is broader than day-0 optimized TP/CP/kernel coverage; the fastest paths remain architecture- and environment-dependent |
| Megatron Bridge | Production Megatron-Core training, Hugging Face interoperability, and reusable recipes | Broad bidirectional HF↔Megatron conversion, model providers, training recipes, and Megatron-Core performance features | More concepts and dependencies than a small runtime; extending a frontier architecture usually requires both a model provider and conversion mappings |
| MLite | Fast incubation of native high-performance model implementations and explicit runtime/primitives inside Megatron-LM | Small replaceable primitives, typed model protocols, explicit runtime tiers, and paired Core/Bridge validation surfaces | Experimental source-only package, narrow model/recipe coverage, and less operational tooling than either NeMo project |

The closest same-domain comparison is Megatron Bridge. AutoModel is the clearest
reference for Hugging Face-native ease of use. veScale is most useful as a
design reference, but its legacy and current public surfaces must be kept
separate.

## Functional comparison

| Capability | veScale | NeMo AutoModel | Megatron Bridge | MLite |
| --- | --- | --- | --- | --- |
| Primary abstraction | Legacy: extended DTensor + DModule + DeviceMesh in eager SPMD. Current public direction: RaggedShard DTensor/FSDP | Hugging Face-compatible `nn.Module` implementations over PyTorch DTensor/DeviceMesh | Megatron-Core model providers plus an HF↔Megatron bridge and training layer | Native model protocols over explicit runtime, model, and primitive layers |
| Model onboarding | Generic PyTorch modules can be parallelized with sharding plans; public examples focus on nanoGPT, Llama, and Mixtral | HF models load without checkpoint conversion; optimized implementations and plans are added incrementally | A supported family gets a model provider, mapping registry, conversion verification, and optionally recipes | A family gets a typed config, native model, protocol, and per-model HF mapping |
| Current model breadth | The current public repository is not a maintained model zoo | Broad HF fallback plus optimized LLM, VLM, diffusion, retrieval, and custom-MoE paths | Broad LLM/VLM/audio/omni/diffusion bridge and recipe catalog | Five native families in the source registry: Qwen3 MoE, Qwen3.5 MoE, Kimi K2, GLM5, and DeepSeek V4; `qwen3` is a Qwen3-MoE compatibility alias |
| Data/optimizer sharding | Legacy DDP and Megatron-derived ZeRO 2+ distributed optimizer; current RaggedShard work targets flexible FSDP | FSDP2/HSDP and Megatron FSDP | Megatron DDP/distributed optimizer and Megatron FSDP | Megatron distributed optimizer (`dist_opt`) and FSDP2 backends |
| Model parallelism | Legacy TP, SP, PP, n-D mesh, and an EP implementation; graph-eager and manual PP modes | TP/SP, PP, CP, EP, and DP composition | TP, PP, VPP, CP, EP, and ETP through Megatron-Core | TP, PP, VPP, CP, EP, and ETP in `ParallelConfig`; SP is used with TP |
| Automatic planning | The public Auto TP/SP and Auto PP pages still say “Coming Soon”; manual/plan-zoo paths are documented | Configuration-driven plans, HF TP plans, and AutoPipeline; not a general whole-program parallel-plan search | Tuned recipes select known-good configurations; not a general automatic planner | Typed explicit configuration and automatic PP layout balancing; not a general search planner |
| Checkpoint/resume | Legacy DCP-based save/load, DP/TP/PP resharding, async save, plan caching, load balancing, and broadcast | DCP plus sharded or consolidated HF Safetensors; optimizer state in DCP; mesh/topology resharding | Megatron distributed checkpoints plus parallelism-aware, per-parameter streaming HF import/export | DCP/FSDP2 and Megatron dist-checkpoint paths, optimizer/RNG resume, topology reshard tests, and per-model HF Safetensors load/export; no async save in the inspected source |
| Training workflows | Framework primitives and a few training examples; no current public recipe catalog | Pretraining, SFT, LoRA/QLoRA, knowledge distillation, QAT, tool calling, diffusion, retrieval, and speculative-model recipes | Pretraining, SFT, LoRA/DoRA, inference, ModelOpt flows, and production recipes | Pretrain-ready runtime contract plus VERL and Miles examples for SFT/GRPO; no general dataset/recipe catalog |
| RL integration | The repository describes internal LLM/RL use, but the inspected public surface is limited | HF-compatible models can be used by external RL stacks; RL is not the primary built-in trainer | Integration APIs exist, while NeMo RL owns the full post-training stack | Runtime `export_weights` and `to()` tiers are designed for RL; committed VERL/Miles adapters exercise actor training and weight refresh |
| Precision/performance stack | Eager-dispatch and communication optimizations; topology-stable distributed RNG requires the patched stack in the legacy design | TE, DeepEP, FlexAttention, torchao FP8, fused losses, prefetch, and model-specific kernels | Megatron-Core kernels, Transformer Engine, communication overlap, FP8/MXFP8/NVFP4, recompute, and packed sequence support | Native TE/custom primitives, fused losses, specialized attention/MoE modules, deterministic mode, recompute, offload, and packed THD paths |
| Extension model | Register operator sharding rules, module plans, or custom PP schedules | Compose independent components while retaining the HF API | Add a provider/bridge mapping and optionally a recipe | Register a small model protocol or runtime backend; shared behavior belongs in replaceable primitives |
| Public maturity | Public surface is in transition: old system under `legacy/`, new system only partly open | Released, packaged, documented, and actively expanding | Released, packaged, containerized, extensively documented, and tied to pinned MCore snapshots | Explicitly experimental, importable from source, and not yet integrated into repository-level packaging |

### Important capability nuances

1. **“Day-0” is not the same as “fully optimized.”** AutoModel can retain the
   HF model and checkpoint format immediately, while optimized TP plans, CP
   hooks, fused kernels, and model-specific recipes arrive separately. MLite
   should adopt this distinction in its own support reporting.
2. **veScale auto planning is a direction, not a current public baseline.** The
   legacy docs describe experimental planning, while the dedicated Auto TP/SP
   and Auto PP pages contain no implementation guide. It should not be scored as
   a delivered advantage.
3. **MLite source is ahead of its introductory prose.** The source registry has
   five native model families, while older introductory text still describes an
   initial Qwen-only drop. Capability reporting should be generated from the
   registry rather than maintained by hand.
4. **Bridge and MLite optimize different ownership boundaries.** Bridge owns the
   broad interoperability/recipe product. MLite owns native experimental models,
   small primitives, and a runtime contract that can use Bridge as a backend and
   comparison reference.

## Performance evidence

### The only direct comparison in the inspected evidence

The committed [MLite benchmark record](../examples/bench/README.md) contains a
paired MLite, legacy `mbridge`, and real Megatron Bridge run. All three rows used
the same synthetic stream and configuration: 8× H100 80 GB, Qwen3.5 MoE reduced
to 8 layers and 8 experts, BF16, sequence length 1024, four microbatches,
Megatron distributed optimizer, five warmup steps, and ten measured steps.
Slurm job `12624917` completed with exit code `0:0`.

| Runtime | Average step time | Tokens/s (8 GPUs) | Peak memory | Model TFLOP/s/GPU |
| --- | ---: | ---: | ---: | ---: |
| MLite native | 309.433 ms | 105,896.935 | 14.324 GB | 80.444 |
| Legacy `mbridge` reference | 332.201 ms | 98,639.089 | 17.987 GB | 74.931 |
| Real Megatron Bridge backend | 334.936 ms | 97,833.496 | 16.403 GB | 74.319 |

Within this reduced, paired workload, MLite had 7.61% lower step time, 8.24%
higher throughput, and 12.67% lower peak memory than the real Bridge row. These
numbers do **not** establish full-model or at-scale superiority.

The paired performance run reported loss agreement within `atol=0.05` and
`rtol=0.005` (`max_abs_diff=0.000500`). A separate deterministic job
`12630675` established bitwise loss and grad-norm parity plus weight/logit
fingerprints against the **legacy `mbridge`** path. It did not establish strict
bitwise parity against the real Megatron Bridge row. Performance and precision
claims must keep those two references distinct.

### Published results that are not cross-comparable

| Project | Representative published evidence | Why it cannot rank the four projects |
| --- | --- | --- |
| veScale | The 2025 paper reports up to 2.2× speedup over systems including TorchTitan and a 78.4% code-complexity reduction. The 2026 veScale-FSDP paper reports 5–66% higher throughput and 16–30% lower memory than its evaluated FSDP baselines | Paper-selected models, baselines, scales, and implementation snapshots differ from every MLite run; the newer FSDP result is also narrower than the legacy full-framework claim |
| NeMo AutoModel | The official [performance summary](https://docs.nvidia.com/nemo/automodel/latest/performance/performance-summary) reports Qwen3 MoE 30B on 8× DGX-H100, BF16, sequence length 4096, at 12,040 tokens/s/GPU and 277 model TFLOP/s/GPU | Full model, different sequence/batch/precision/kernel stack, and per-GPU metric versus MLite's reduced Qwen3.5 workload |
| Megatron Bridge | The official [26.06 performance summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) reports Qwen3 30B-A3B on 16× DGX-H100, FP8, sequence length 4096, at 8,826 tokens/s/GPU and 203 model TFLOP/s/GPU | Different GPU count, precision, model, batch, and full-model workload; MoE results also use forced-balanced, token-dropless routing |
| MLite | The paired result above reports complete step-time, throughput, memory, TFLOP, loss, and environment fields | It is reproducible evidence for one reduced proxy only; it is not a substitute for a full-model scaling curve |

The next credible four-way performance study would need one common HF revision,
identical tokens and loss semantics, matched BF16/FP8 modes, the same hardware,
the same warmup/repeat protocol, complete memory and quality metrics, and a
declared treatment of routing balance. Until then, feature and usability
decisions should not be justified with unmatched throughput numbers.

## Ease-of-use comparison

| User journey | veScale | NeMo AutoModel | Megatron Bridge | MLite |
| --- | --- | --- | --- | --- |
| Install and first run | Legacy quick start builds patched PyTorch and TorchDistX, then installs from source or a locally built image | `pip`/`uv` package, NGC image, YAML recipe, `automodel` CLI, local and Slurm launch paths | Package or NeMo container, then Python/YAML recipes through `torchrun` or NeMo-Run | Add `experimental/lite` to `PYTHONPATH`, construct typed Python config, and launch through a project/example script |
| Start from an HF model | Generic model code can remain PyTorch-native, but the public legacy flow requires an explicit sharding plan and patched stack | Direct `from_pretrained`; model and checkpoint remain HF-native | `AutoBridge` detects a supported architecture and streams weights into a Megatron provider | HF config and weights are consumed by a registered native protocol; unsupported families need code |
| Change parallelism | Edit DeviceMesh and sharding/PP plans | Edit YAML/device-mesh configuration | Override recipe/`ConfigContainer` model fields | Edit `ParallelConfig`; advanced users can provide an explicit PP layout |
| Inspect before launch | Eager execution is debuggable, but the public project has no current general CLI | YAML/CLI overrides and launcher tooling | Typed config, documented recipe overrides, and profiling/validation guides | Benchmark and integration scripts support dry-run; there is no general `mlite` CLI |
| Add a frontier model | Potentially low model-code change if operator coverage and plans are sufficient; unsupported ops need sharding rules | HF fallback is immediate, then optimized hooks/plans/kernels can be layered in | Add/extend MCore provider plus bridge mappings; recipe work is separate | Implement a native model/protocol and mapping while reusing small primitives; more work than HF fallback but more control |
| Validate correctness | Single-device semantic goal and topology-stable RNG are unusually strong ideas, but rely on the legacy patched stack | Broad tests and checkpoint interoperability; exact topology-invariant claims are model/path specific | Built-in conversion verification and production MCore tests | Independent-reference rules, deterministic fingerprints, paired backends, and repository maintenance skills make the expected evidence explicit |

AutoModel has the shortest path from a new HF checkpoint to a runnable training
job. Megatron Bridge has the strongest production path from HF interoperability
to Megatron-Core performance. MLite has the smallest reviewable implementation
surface for native frontier work, but currently asks the user to assemble more
of the environment and workflow. veScale's abstraction is elegant, while its
public installation and repository transition impose the highest adoption risk.

## MLite/mbridge versus Megatron Bridge

Three names must remain distinct:

- `mlite` is the native MLite runtime and native model implementation path.
- `mbridge` is a legacy external package used by MLite as the validated
  Megatron-Core/distributed-optimizer reference.
- `bridge` is MLite's adapter to the official `megatron.bridge` package.

Calling both reference paths “Bridge” hides an important evidence boundary.

| Area | MLite and legacy `mbridge` | Official Megatron Bridge | Recommended ownership |
| --- | --- | --- | --- |
| Native model implementation | MLite owns native frontier implementations and reusable primitives; legacy `mbridge` materializes MCore models for comparison | Owns broad MCore providers and model integrations | Keep experimental native architecture work in MLite; graduate generally useful MCore/provider work upstream |
| HF weight mapping | MLite has explicit per-model `HFWeights` mappings; `mbridge` supplies a legacy conversion reference | Owns broad, parallelism-aware, per-parameter streaming import/export and verification | Do not rebuild a second general bridge product in MLite; expose official Bridge behind a stable adapter |
| Training runtime | MLite defines a compact pretrain/RL-ready/RL-best contract and can run native, legacy, or official-Bridge backends | Owns a complete configurable training loop, data/checkpoint/logging utilities, and recipes | Keep MLite's backend-neutral contract small; use Bridge when users need its production training layer |
| Recipes and operations | A few benchmark, VERL, and Miles examples | Large recipe/tutorial/performance/container surface | Reuse or interoperate with Bridge recipes rather than cloning the catalog |
| Validation | Same-harness native/reference comparisons and deterministic fingerprints | Conversion verification, checkpoint integrity checks, and broad upstream tests | Make official Bridge the long-term interoperability reference; keep legacy `mbridge` only where it provides unique parity evidence |
| RL | MLite exports HF-format weights and supports offload hooks for external actor/rollout stacks | Bridge exposes integration points, while the full NVIDIA RL workflow belongs to NeMo RL | Treat both as train backends under a separate RL orchestrator; avoid embedding RL policy logic in model primitives |

The complementary product boundary is therefore:

1. **MLite is the native model and primitive incubator.** It should stay small,
   explicit, and optimized for reviewing new architecture work.
2. **Megatron Bridge is the production interoperability and recipe layer.** It
   should be preferred for broad HF conversion, supported-model recipes,
   packaged operation, and production documentation.
3. **The MLite `bridge` backend is the contract between them.** It is useful for
   A/B validation and for users who want MLite's runtime surface over an
   official Bridge model.
4. **Legacy `mbridge` is a reference, not a second product direction.** New
   user-facing documentation should call it `mbridge_legacy` conceptually, and
   new correctness claims should migrate to the official Bridge path when the
   required model support and deterministic controls exist.

## Borrow list for MLite

Priority means expected value to MLite, not implementation authorization. Every
item must preserve the runtime/model/primitive layering: model names and recipe
knowledge must not leak into primitives.

### P0 — high value, low architectural risk

1. **Generate a capability manifest from registries and protocols.** Borrow the
   discoverability of AutoModel's model coverage and Bridge's support tables.
   Report each model's native/fallback status, workflows, optimizer backends,
   parallel dimensions, HF import/export, and validation level. This would have
   prevented the current source-versus-introductory-doc drift. Acceptance can be
   CPU-only: schema validation plus a test that every registered protocol
   resolves and declares its capabilities.
2. **Standardize benchmark evidence as a machine-readable manifest.** Borrow
   the complete configuration tables from both NeMo projects and enforce the
   local [`application.bench`](../skills/application/bench.md) contract: model
   revision, hardware/software, dataset or synthetic stream, precision,
   parallel layout, routing policy, warmup, repeats, throughput, step time,
   memory, quality metric, and correctness artifact. A benchmark missing any of
   those fields should not publish a comparative claim.
3. **Clarify backend names and evidence labels.** Introduce explicit user-facing
   names or aliases such as `mbridge_legacy` and `megatron_bridge`, with a
   deprecation path for ambiguous names. Result files should record the package
   version and implementation identity, not just `bridge`.
4. **Generalize dry-run into a small `plan --explain` surface.** Borrow
   AutoModel's CLI/YAML ergonomics and Bridge's typed configuration, while
   retaining MLite's Python dataclasses. The output should resolve model,
   backend, mesh, optimizer, checkpoint format, runtime tier, unsupported
   combinations, and the exact launch command without importing optional GPU
   packages.

### P1 — valuable, but requires design and GPU evidence

5. **Offer an explicitly labeled HF-native fallback lane.** Borrow AutoModel's
   day-0 path for correctness bring-up and integration, while keeping MLite
   native protocols as the optimized path. A fallback must never silently count
   as native model support or performance evidence. Promotion to “optimized”
   should require an independent reference, end-to-end training, and declared
   parallel/kernel coverage.
6. **Expose one stable HF interoperability API.** Borrow Bridge's `AutoBridge`
   ergonomics and conversion verification. Under that API, prefer delegation to
   official Megatron Bridge for supported MCore models and use MLite's per-model
   mappings for native models. Avoid a second generic conversion registry unless
   it represents MLite-native parameters that Bridge cannot express.
7. **Add asynchronous checkpoint save and publish reshard coverage.** veScale's
   plan caching/async-save ideas and AutoModel's HF-compatible terminal format
   are useful. Build them above the existing checkpoint primitives, with crash,
   resume, optimizer/RNG, topology-change, and bandwidth evidence. Do not infer
   large-scale behavior from a single-rank test.
8. **Make topology-stable RNG a first-class correctness target.** veScale's
   single-device-equivalent random stream is more valuable than copying its
   patched-PyTorch implementation. Start with a bounded contract for dropout and
   routing across supported TP/CP layouts, then use the strongest feasible
   independent reference and deterministic fingerprints.
9. **Define a graduation path from MLite to MCore/Bridge.** For each native
   model, distinguish experimental-only primitives, generally reusable MCore
   capabilities, official Bridge provider/mapping work, and recipe work. This
   prevents permanent duplication while preserving MLite's faster incubation
   loop.

### P2 — investigate, do not promise

10. **Explore recommendation before automatic planning.** A bounded recommender
    can use model shape, memory estimates, and a small catalog of validated
    layouts to explain candidate TP/PP/CP/EP configurations. Full eager automatic
    planning should wait for public, reproducible evidence; veScale's current
    auto-plan documentation is not a sufficient implementation baseline.

## Ideas not to copy directly

- Do not require a long-lived PyTorch fork merely to reproduce veScale's legacy
  RNG or DTensor behavior. Prototype against upstream extension points first.
- Do not import a whole HF model abstraction into MLite primitives. A generic
  fallback belongs above the model layer and must not teach lower-level
  optimizer/checkpoint code about model families.
- Do not duplicate Megatron Bridge's full model zoo, recipe catalog, conversion
  registry, or operational documentation. Integrate it and contribute generic
  improvements upstream.
- Do not market automatic parallelism before the planner, constraints, fallback
  behavior, and performance/correctness evidence are public and reproducible.
- Do not compare published throughput unless model revision, precision, tokens,
  hardware, parallel layout, routing, warmup, repeats, memory, and quality are
  aligned.

## Local MLite evidence map

- [Architecture and layer ownership](architecture.md)
- [Runtime contract and backend identities](runtime.md)
- [Source model registry](../megatron/lite/model/registry.py)
- [Parallel configuration](../megatron/lite/runtime/contracts/config.py)
- [Runtime tiers and backend registry](../megatron/lite/runtime/backends/__init__.py)
- [Optimizer backend registry](../megatron/lite/primitive/optimizers/__init__.py)
- [Checkpoint primitives](../megatron/lite/primitive/ckpt/)
- [Paired benchmark and deterministic evidence](../examples/bench/README.md)
- [VERL integration](../examples/verl/README.md)
- [Miles integration](../examples/miles/README.md)
- [Maintenance skill contracts](../skills/README.md)

# FP8 Training Architecture Study

This document is a source-based design proposal, not an implementation or a
performance claim. The research was completed without running GPU code. The
upstream snapshot is NVIDIA Megatron-LM `cf2f07d7b` (2026-07-09) and
Transformer Engine `8b9968255` (2026-07-08).

## Decision Summary

FP8 should be an **additive capability of existing primitives**, not a new set
of FP8 model implementations. In particular:

- Do not add `QwenFP8Model`, `FP8Attention`, or parallel `*/fp8/` model trees.
- Add one model-neutral precision policy and bind it to existing linear,
  attention, dense-MLP, and expert primitives.
- Select precision by semantic target, recipe, and number format. Model names
  must never appear in the precision primitive.
- Keep parameter storage/all-gather precision and optimizer main-weight
  precision separate from FP8 compute.
- Unmatched targets remain BF16. A requested target that is not covered by a
  precision-aware primitive must fail during model construction rather than
  silently falling back.

This is an additive design: adding a model or adding a precision recipe grows
one axis. It avoids the multiplicative `models x FP8 variants` implementation
matrix.

The recommended first implementation is Hopper blockwise E4M3 for TE linear
GEMMs, initially covering attention projections and dense linears while
keeping core attention and parameters in BF16. MoE grouped linears follow as a
separate gate. MXFP8, FP8 parameter all-gather, and FP8 core attention should
be later stages.

## Terminology: Keep the Axes Orthogonal

The following concepts are independent and should not be collapsed into one
`fp8: bool`:

| Axis | Values | Meaning |
| --- | --- | --- |
| Number format | `e4m3`, `hybrid` | `hybrid` uses E4M3 in forward and E5M2 for backward output gradients. |
| Scaling recipe | `delayed`, `tensorwise`, `blockwise`, `mxfp8`, later `custom` | How scale factors are computed and at what granularity. |
| Compute target | `attention_linear`, `attention_core`, `moe`, `dense` | Which semantic part of a composed model enters an FP8 context. |
| Parameter path | BF16 or FP8 compute parameters; BF16 or FP8 all-gather | Storage and communication, independent of activation/GEMM autocast. |
| Optimizer path | FP32 or explicitly lower-precision main params, gradients, and states | Update precision, independent of the FP8 compute recipe. |

There is no E3M4 FP8 training format in the surveyed Megatron-Core or
Transformer Engine API. TE defines E4M3, E5M2, and HYBRID; pure E5M2 training
is rejected. The likely intended pair is E4M3/E5M2, with HYBRID assigning E4M3
to forward and E5M2 to backward. See
[`recipe/__init__.py:29-50`][te-formats] and
[`transformer_config.py:556-569`][mcore-config].

Likewise, MXFP8 and blockwise FP8 are scaling recipes, not number formats.

## Megatron-Core and Transformer Engine Survey

### Recipe coverage

Megatron-Core exposes `delayed`, `tensorwise`, `mxfp8`, `blockwise`, and
`custom` in [`enums.py:12-19`][mcore-enums]. Its recipe factory maps those
values to TE's `DelayedScaling`, `Float8CurrentScaling`,
`MXFP8BlockScaling`, and `Float8BlockScaling` in
[`fp8_utils.py:554-612`][mcore-recipe].

| Recipe | Scale granularity and state | Default/useful format | Device and important constraints |
| --- | --- | --- | --- |
| Delayed scaling | One scale per tensor, derived from previous-step amax history. Stateful across iterations. | HYBRID by default. | The conservative existing path. Megatron wraps the whole transformer block because entering/exiting per layer breaks amax reduction semantics; first/last BF16 layers are therefore rejected with delayed scaling. |
| Tensorwise current scaling | One current scale per tensor; no amax history. | HYBRID by default. | MCore calls it `tensorwise`. It can carry the FP8 DPA flag. |
| Blockwise FP8 | Default activation and gradient blocks are 1x128; weight blocks are 128x128; scales are FP32 containers, power-of-two by default. | E4M3 by default; HYBRID is allowed. | SM90 Hopper or later. On Blackwell it is emulated with MXFP8 and MXFP8 is preferred. TE explicitly rejects FP8 DPA/MHA for this recipe. |
| MXFP8 | One E8M0 power-of-two scale per 32 consecutive values. Rowwise and columnwise quantizations must both be made from the high-precision source. | E4M3 by default; HYBRID is allowed. | Native on SM100+ Blackwell. Dimensions must meet 32-element block rules. Quantized all-gather is supported. |
| Custom | User factory selected by import path. | Factory-defined. | Useful as an extension point, not an initial MLite delivery target. |

Primary TE details are in
[`DelayedScaling:173-270`][te-delayed],
[`Float8CurrentScaling:286-333`][te-current],
[`MXFP8BlockScaling:337-384`][te-mx-recipe], and
[`Float8BlockScaling:388-457`][te-block-recipe]. The blockwise device matrix
is documented at [`fp8_blockwise_scaling.rst:150-178`][te-block-device]; the
MXFP8 1x32/E8M0 contract is at [`mxfp8.rst:10-57`][te-mx-doc].
MCore's delayed-scaling/first-last-layer rejection is at
[`transformer_config.py:1371-1374`][mcore-fp8-validation].

### What is and is not independently selectable

The normal MCore path opens a TE FP8 autocast around transformer layers. TE
linear and grouped-linear modules under that context use the selected recipe.
For non-delayed recipes, MCore opens the context per layer, which also permits
first/last BF16 layer exclusions
([`transformer_block.py:600-664`][mcore-block-context]).

Core attention is not automatically the same thing as attention projection
linears:

- `fp8_dot_product_attention` and `fp8_multi_head_attention` are separate,
  default-off fields
  ([`transformer_config.py:600-613`][mcore-attn-config]).
- Delayed scaling passes both flags into TE. TE describes DPA as casting the
  high-precision boundary tensors, while full MHA removes those boundaries for
  a standard `linear + DPA + linear` module
  ([`recipe/__init__.py:222-250`][te-delayed-attn]).
- MCore passes DPA, but not MHA, to tensorwise and MXFP8 recipes
  ([`fp8_utils.py:579-590`][mcore-recipe]).
- `Float8BlockScaling` rejects both DPA and MHA
  ([`recipe/__init__.py:435-454`][te-block-attn]).

Therefore, projections can be FP8 while core attention stays BF16. Core
attention can only be enabled for a recipe and backend that support it. A
single `target="attention"` switch would hide this material distinction, so
the proposed MLite target axis splits `attention_linear` from
`attention_core`.

MCore also has an experimental per-module precision file. Ordered glob
matchers can, for example, keep `*.linear_qkv` and `*.linear_proj` in BF16 while
using MXFP8 elsewhere
([`TransformerEngineMixedPrecision.md:47-111`][mcore-mixed-doc]). This allows
independent linear/grouped-linear overrides, but it is not a complete semantic
capability system:

- matching depends on module names;
- FP8 parameter initialization is out of scope;
- CUDA graphs and activation recompute are not verified;
- other MCore decisions still read the global `TransformerConfig` and do not
  observe the override.

Those limitations are explicit at
[`TransformerEngineMixedPrecision.md:26-38`][mcore-mixed-limit]. MLite should
reuse the recipe semantics, but not make glob names its primary public API.

### MoE-specific controls

MCore applies the global FP8 recipe to TE expert GEMMs, with extra MoE
requirements rather than a separate model implementation. It requires suitable
TE versions for MoE and grouped GEMM, and converts the compatibility
`moe_router_padding_for_fp8` flag to the general quantized-routing padding flag
([`transformer_config.py:2258-2289`][mcore-moe-validation]). Its MoE guide
recommends blockwise FP8 on Hopper, MXFP8 on Blackwell, routing-map padding,
and FP8 parameter all-gather only as an additional optimization
([`moe/README.md:463-498`][mcore-moe-guide]).

This supports the capability approach: MoE contributes alignment and dispatcher
constraints, but it does not require an FP8 copy of each model.

### Parameters and optimizer main weights

FP8 compute does not imply FP8 optimizer updates:

1. With ordinary FP8 autocast, model parameters may remain BF16 and are
   quantized for TE GEMMs.
2. `TransformerConfig.fp8_param` asks TE to initialize selected GEMM weights in
   FP8 ([`transformer_config.py:571-576`][mcore-config]).
3. `fp8_param_gather` keeps compute parameters in FP8 and communicates them in
   FP8. MCore restricts it to distributed optimizer, FSDP, or inference paths
   ([`arguments.py:1036-1038`][mcore-param-validation]).
4. Optimizer main weights and states remain separately configurable. The
   default MCore main parameter, main gradient, and Adam state dtypes are FP32
   ([`optimizer_config.py:184-207`][mcore-optimizer-config]). Megatron-FSDP also
   defaults main params to FP32 and requires a main-param representation for
   quantized parameters
   ([`megatron_fsdp README:116-125`][mcore-fsdp-policy]).

MCore specifically coordinates ownership of master weights between MCore and
TE for precision-aware optimization
([`optimizer/__init__.py:563-585`][mcore-master-ownership]). MLite must not
silently change optimizer state precision when FP8 compute is enabled.

## Current MLite Inventory

The current production entrypoint does **not** expose FP8 training. All model
protocols construct BF16 models and explicitly set `fp8=False`, including
Qwen3 MoE
([`qwen3_moe/protocol.py:174-197`][mlite-qwen3-protocol]), Qwen3.5
([`qwen3_5/protocol.py:181-211`][mlite-qwen35-protocol]), Kimi K2
([`kimi_k2/protocol.py:160-186`][mlite-kimi-protocol]), GLM-5
([`glm5/protocol.py:200-226`][mlite-glm-protocol]), and DeepSeek V4
([`deepseek_v4/protocol.py:357-386`][mlite-ds4-protocol]). As a result, direct
model-constructor FP8 branches are not reachable through the MLite runtime.

| Surface | Already present | Missing or unsafe to claim |
| --- | --- | --- |
| Recipe builder | `build_fp8_recipe()` returns TE `DelayedScaling(margin=0, HYBRID)`. | Its `train_config` argument is unused; there is no recipe/format selection, capability check, amax group, or failure contract ([`primitive/utils/__init__.py:7-11`][mlite-recipe]). |
| TP linears | `ColumnParallelLinear` and `RowParallelLinear` use TE modules with BF16 params, so they can participate in TE autocast. | The vanilla LM-head path is intentionally torch matmul and is not FP8-capable. There is no semantic target metadata ([`parallel/linear.py:166-252`][mlite-linear]). |
| GQA | Projection linears and `te.DotProductAttention` are reusable primitives. | The current recipe never enables FP8 DPA/MHA. Projection FP8 and core-attention FP8 are not independently expressible ([`modules/gqa.py:77-171`][mlite-gqa]). |
| MLA/DSA attention | Several projections use reusable TE parallel linears. | MLA also contains direct `nn.Linear` and torch SDPA fallback paths, which TE autocast does not make FP8 ([`attention/mla.py:106-133`][mlite-mla-linear], [`attention/mla.py:175-189`][mlite-mla-core]). |
| Experts | `Experts` uses TE `GroupedLinear` with BF16 params and already implements FP8 token-count padding to a multiple of 16 ([`experts.py:71-102`][mlite-experts-build], [`experts.py:137-230`][mlite-experts-forward]). | The padding is driven by a model-threaded boolean. Qwen3.5, Kimi K2, and GLM-5 explicitly raise for FP8 MoE ([`qwen3_5/model.py:154-178`][mlite-qwen35-guard], [`kimi_k2/model.py:266-291`][mlite-kimi-guard], [`glm5/model.py:381-405`][mlite-glm-guard]); only direct Qwen3 construction reaches the partial path. |
| Model forward | Model classes contain a whole-model `te.fp8_autocast` branch; Qwen3 is representative ([`qwen3_moe/model.py:441-478`][mlite-qwen3-forward]). | The context is one hard-coded recipe for all locations. It cannot express target rules, and the production protocol disables it. Copying this branch into more models would multiply maintenance. |
| BF16 and optimizer | Protocols call `.to(torch.bfloat16)`. The MCore wrapper builds `TransformerConfig(bf16=True, params_dtype=bfloat16)` ([`megatron_wrap.py:256-282`][mlite-mcore-config]). FSDP2 can maintain FP32 master params ([`fsdp2/adamw.py:150-173`][mlite-fsdp-master]). | There is no FP8 parameter initialization/all-gather contract. Enabling autocast must not mutate these defaults. |

### What the block-FP8 resync work proves

Adjacent, not-yet-integrated resync work contains a pure-tensor 128x128 E4M3
checkpoint quantizer and dequantizer. It validates shape divisibility, computes
per-block scales, and emits checkpoint-format values
([`block_fp8.py:13-92`][resync-block]). A DS4 adapter then selects model-specific
weights and ignored layers
([`deepseek_v4/resync.py:25-93`][resync-model]).

The reusable evidence is:

- a CPU-checkable quantize/dequantize oracle for checkpoint weight blocks;
- explicit shape/dtype failure modes;
- the need to distinguish quantized weights from ignored router, norm, embedding,
  and other non-GEMM weights;
- end-to-end resync validation methodology.

It is **not** a training FP8 implementation. It does not create TE activation or
gradient recipes, manage forward/backward row/column representations, or enter
FP8 GEMM contexts. Its model-name and checkpoint-name selection rules must stay
in a model/export adapter and must not be moved into a training precision
primitive.

## Design Invariants from the MLite Skills

The proposal follows the repository's function-model skills:

- `basic.constitution`: choose the smallest reviewable design, keep primitives
  replaceable, and validate against Megatron first
  ([`skills/basic/constitution.md:23-42`][skill-constitution]).
- `primitive.principle`: define shape/dtype/rank rules, forward/backward/update
  equivalence, a precision threshold, and a small proxy before implementation
  ([`skills/primitive/principle.md:20-31`][skill-principle]).
- `primitive.contract`: declare owned modules, public API, state/config,
  placement, valid/invalid combinations, and composition/e2e validation
  ([`skills/primitive/contract.md:22-60`][skill-contract]).
- `primitive.design`: use a checkable reference, state the principle, and make
  selection and replaceability explicit
  ([`skills/primitive/design.md:19-40`][skill-design]).

These produce the following non-negotiable invariants:

1. Enabling a policy must not change tensor shapes, output dtype contracts,
   process groups, parameter ownership, optimizer semantics, or model topology.
2. Unselected targets must follow the existing BF16 path.
3. Every selected target must report a matched precision-aware primitive; no
   silent direct-`nn.Linear`, torch-attention, or unsupported-kernel fallback is
   allowed.
4. A primitive may know semantic capabilities such as `moe` or
   `attention_core`, but it may not know Qwen, Kimi, GLM, or DeepSeek names.
5. Recipe state is owned by the precision capability and is reused, not rebuilt
   on every forward.
6. Parameter storage/all-gather and optimizer main-weight precision require
   their own explicit opt-in and validation.

## API Alternatives

### A. New FP8 primitives plus FP8 model implementations

Example shape:

```python
class FP8ColumnParallelLinear(ColumnParallelLinear): ...
class Qwen35FP8Model(Qwen35Model): ...
```

This is easy to prototype but is rejected. It multiplies model and recipe
variants, duplicates architecture fixes, and puts recipe selection in the model
layer. It also makes mixed BF16/FP8 composition awkward.

### B. Typed semantic policy bound to existing primitives (recommended)

Example user API:

```python
from megatron.lite.primitive.precision import (
    FP8Format,
    FP8Recipe,
    FP8Rule,
    ParameterPrecision,
    PrecisionPolicy,
    PrecisionTarget,
)

precision = PrecisionPolicy(
    rules=(
        FP8Rule(
            targets={
                PrecisionTarget.ATTENTION_LINEAR,
                PrecisionTarget.DENSE,
            },
            recipe=FP8Recipe.BLOCKWISE,
            format=FP8Format.E4M3,
        ),
    ),
    parameters=ParameterPrecision(
        compute="bf16",
        main="fp32",
        all_gather="bf16",
    ),
)

cfg = MegatronLiteConfig(model_name="qwen3_moe", precision=precision)
```

Each target may appear in at most one rule. There is no ordering or implicit
last-match-wins behavior. An absent target is BF16. This makes recipe x target
composition data rather than classes.

`ATTENTION_LINEAR` covers QKV/up/down/output projection GEMMs inside an
attention primitive. `ATTENTION_CORE` covers DPA itself. `MOE` covers routed
and shared expert GEMMs, not the router. `DENSE` covers non-MoE MLP GEMMs. The
embedding, normalization, router, and vocabulary head remain BF16 unless a
future design adds separately reviewed targets.

The public policy belongs on the common `MegatronLiteConfig`, not in every
model's `ImplConfig`, because it is a cross-model primitive capability.

Implementation sketch:

```python
plan = compile_precision_policy(cfg.precision, te_capabilities())
bundle = protocol.build_model(model_cfg, impl_cfg=impl_cfg)
coverage = bind_precision(bundle.chunks, plan)
plan.validate_coverage(coverage)

# Inside reusable primitives, not model implementations:
with self.precision.context(PrecisionTarget.ATTENTION_LINEAR):
    qkv = self.qkv(x)
with self.precision.context(PrecisionTarget.ATTENTION_CORE):
    out = self.core_attn(q, k, v)
```

`bind_precision` installs an explicit controller on precision-aware primitives
after model construction. It does not use a process-global mutable policy or
module-name guessing. Inline operations that do not expose the requested
semantic target make coverage validation fail. Models become eligible by
composing supported primitives, not by adding FP8-specific model code.

Parameter initialization is deliberately excluded from the first version of
post-build binding. When FP8 parameters are later enabled, construction-time
binding must be designed explicitly; it must not be smuggled into the autocast
controller.

### C. Ordered module-name matchers

Example shape:

```yaml
precision:
  - pattern: "*.linear_qkv"
    recipe: blockwise
  - pattern: "*.experts.*"
    recipe: mxfp8
```

This mirrors MCore's experimental precision file and is useful as an import or
compatibility layer. It is not recommended as MLite's primary API because
module renames change behavior, matches occur after initialization, and model
structure leaks into configuration. If supported, it should compile into the
same typed semantic policy and emit the matched module list as evidence.

## Proposed Capability Contract

### Owned surface

The minimal implementation should own:

- `megatron/lite/primitive/precision.py`: enums, immutable policy/rules,
  recipe factory, capability checks, controller, and coverage validation;
- `runtime/backends/mlite/config.py`: one common `precision` field;
- `runtime/backends/mlite/runtime.py`: compile and bind the policy;
- existing reusable primitives: semantic target boundaries only.

Model protocols should only lose their hard-coded `fp8=False` and old global
autocast wiring after their primitive coverage is complete. No new model
implementation or model registry entry is required.

### Shape, dtype, state, and placement

- Inputs and outputs keep their existing shapes and BF16 public dtype.
- TE performs internal FP8 casts; selected primitives must not expose raw FP8
  outputs across residual or pipeline boundaries unless a later contract says
  so.
- Delayed-scaling amax history and recipe objects are persistent controller
  state. They are not rebuilt on every call.
- Amax reduction groups must come from the primitive's existing TP/CP process
  groups, matching MCore's `get_fp8_context` behavior
  ([`fp8_utils.py:614-672`][mcore-context]).
- MXFP8 rowwise/columnwise representations and blockwise alignment are TE-owned
  compute state. The checkpoint resync tensors are not substituted for them.
- The default parameter path remains BF16 model params plus FP32 main params.

### Valid initial combinations

| Target | Delayed/tensorwise | Blockwise | MXFP8 |
| --- | --- | --- | --- |
| Attention projection linears | Later/reference path | Initial target | After Blackwell gate |
| Core attention | Later, fused-attention-only | Unsupported by TE | Later, only after backend/shape validation |
| Dense linears | Later/reference path | Initial target | After Blackwell gate |
| MoE experts | After grouped-linear padding gate | Second target | After Blackwell grouped-linear gate |

The first implementation should reject, with a useful reason:

- `format="e3m4"` or pure E5M2;
- blockwise plus `attention_core`;
- MXFP8 when TE reports no support or the device is below SM100;
- blockwise when TE reports no support or the device is below SM90;
- a selected target implemented by vanilla `nn.Linear`, torch matmul, local
  attention, or another non-TE path;
- FP8 parameter all-gather without a supported distributed optimizer/FSDP path;
- lower-precision main params without the explicit precision-aware optimizer;
- mixed delayed-scaling target rules in the first release, because MCore and
  its per-module override both expose delayed-scaling state restrictions.

## Validation Contract for an Implementation Task

This design-only study ran no GPU validation. A later implementation must leave
the following evidence.

### CPU and static proxy

- Policy parsing and round-trip serialization for every enum and rule.
- Duplicate-target, unsupported-combination, and missing-coverage failures.
- Recipe factory mapping using mocked TE availability.
- A coverage manifest proving that requested targets matched production
  primitives; tests that only instantiate otherwise unreachable model branches
  do not count.
- Layering checks proving the precision primitive imports no model package and
  contains no model-name conditions.

### Primitive GPU reference

Use MCore/TE with the same recipe as the independent reference. Freeze input,
weights, seed, shapes, process groups, and optimizer. Compare:

- output shape and public BF16 dtype;
- forward output within a reviewed FP8 threshold;
- input and weight gradients;
- one optimizer update from the same FP32 main weights;
- BF16 output for every unselected target;
- fail-loud behavior rather than kernel fallback.

The smallest useful cases are one TP linear, one attention projection/core
split, and a reduced expert count. Composition then needs a real model path with
TP and EP before end-to-end signoff. Performance measurement comes only after
the BF16 baselines and precision contract align.

### Composition and end to end

- First model: Qwen3 MoE, because its primitives already have the most complete
  partial FP8 path. Production protocol reachability must be demonstrated.
- Compare against MCore, not MLite loaded through a second path.
- Cover at least the selected target combinations and one TP+EP composition;
  add PP/CP cells where the touched primitive uses those groups.
- Report loss, gradients, updated weights, and logits. A narrow constructor or
  isolated smoke does not prove delivery.
- Run all GPU work through the repository's Slurm environment and report real,
  non-skipped job results.

## Staged Delivery

1. **Policy and fail-loud coverage, CPU only.** Add the typed API, TE capability
   mapping, semantic target boundaries, and static tests. BF16 remains the
   default and no existing config changes behavior.
2. **Hopper blockwise E4M3 for linear GEMMs.** Enable
   `attention_linear` and `dense` on TE-backed primitives. Keep
   `attention_core`, MoE, parameters, all-gather, and optimizer state BF16/FP32.
   Prove a primitive reference and a reachable Qwen3 composition.
3. **Hopper blockwise MoE.** Add grouped-linear and routing-padding validation,
   then enable `moe`. Do not encode a list of supported models; coverage decides
   eligibility.
4. **Model-family expansion by composition.** Migrate inline linears to existing
   reusable primitives where necessary and remove model-level FP8 guards only
   after coverage and reference tests pass. No FP8 model variants are added.
5. **Blackwell MXFP8.** Reuse the same policy with a new recipe value. Validate
   1x32 alignment, row/column representations, grouped experts, and real SM100
   availability.
6. **Parameter/all-gather optimization.** Add construction-time quantized params,
   FP8 all-gather, and optimizer ownership as a separate reviewed contract.
7. **Core attention.** Enable only supported recipe/backend combinations. Keep
   blockwise core attention invalid until TE supports it.

The DS4 BF16-to-block-FP8 resync exporter remains a separate rollout/export
capability. It can share terminology and oracle tests, but not model selection
logic or TE training state.

## Decisions Requested

The architecture decision can be made without GPU data:

1. Approve additive capability option B and reject FP8 model variants.
2. Approve the four semantic targets, especially the split between attention
   projections and core attention.
3. Approve blockwise E4M3 linear GEMMs on Hopper as the first implementation,
   with MoE, MXFP8, parameter all-gather, and FP8 core attention staged behind
   separate evidence gates.
4. Keep common precision policy at the runtime config layer while all
   model-specific eligibility is derived from primitive coverage.

## Source Links

[mcore-enums]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/enums.py#L12-L19
[mcore-config]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/transformer_config.py#L556-L576
[mcore-attn-config]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/transformer_config.py#L600-L613
[mcore-fp8-validation]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/transformer_config.py#L1371-L1374
[mcore-recipe]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/fp8_utils.py#L554-L612
[mcore-context]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/fp8_utils.py#L614-L672
[mcore-block-context]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/transformer_block.py#L600-L664
[mcore-mixed-doc]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/extensions/TransformerEngineMixedPrecision.md#L47-L111
[mcore-mixed-limit]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/extensions/TransformerEngineMixedPrecision.md#L26-L38
[mcore-moe-validation]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/transformer_config.py#L2258-L2289
[mcore-moe-guide]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/transformer/moe/README.md#L463-L498
[mcore-param-validation]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/training/arguments.py#L1036-L1038
[mcore-optimizer-config]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/optimizer/optimizer_config.py#L184-L207
[mcore-master-ownership]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/optimizer/__init__.py#L563-L585
[mcore-fsdp-policy]: https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/distributed/fsdp/src/README.md#L116-L125

[te-formats]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/transformer_engine/common/recipe/__init__.py#L29-L50
[te-delayed]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/transformer_engine/common/recipe/__init__.py#L173-L270
[te-delayed-attn]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/transformer_engine/common/recipe/__init__.py#L222-L250
[te-current]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/transformer_engine/common/recipe/__init__.py#L286-L333
[te-mx-recipe]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/transformer_engine/common/recipe/__init__.py#L337-L384
[te-block-recipe]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/transformer_engine/common/recipe/__init__.py#L388-L457
[te-block-attn]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/transformer_engine/common/recipe/__init__.py#L435-L454
[te-block-device]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/docs/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.rst#L150-L178
[te-mx-doc]: https://github.com/NVIDIA/TransformerEngine/blob/8b9968255eb879e6e390f427836906b29aad64d2/docs/features/low_precision_training/mxfp8/mxfp8.rst#L10-L57

[mlite-recipe]: ../megatron/lite/primitive/utils/__init__.py#L7-L11
[mlite-linear]: ../megatron/lite/primitive/parallel/linear.py#L166-L252
[mlite-gqa]: ../megatron/lite/primitive/modules/gqa.py#L77-L171
[mlite-mla-linear]: ../megatron/lite/primitive/modules/attention/mla.py#L106-L133
[mlite-mla-core]: ../megatron/lite/primitive/modules/attention/mla.py#L175-L189
[mlite-experts-build]: ../megatron/lite/primitive/modules/experts.py#L71-L102
[mlite-experts-forward]: ../megatron/lite/primitive/modules/experts.py#L137-L230
[mlite-qwen3-forward]: ../megatron/lite/model/qwen3_moe/lite/model.py#L441-L478
[mlite-qwen35-guard]: ../megatron/lite/model/qwen3_5/lite/model.py#L154-L178
[mlite-kimi-guard]: ../megatron/lite/model/kimi_k2/lite/model.py#L266-L291
[mlite-glm-guard]: ../megatron/lite/model/glm5/lite/model.py#L381-L405
[mlite-qwen3-protocol]: ../megatron/lite/model/qwen3_moe/lite/protocol.py#L174-L197
[mlite-qwen35-protocol]: ../megatron/lite/model/qwen3_5/lite/protocol.py#L181-L211
[mlite-kimi-protocol]: ../megatron/lite/model/kimi_k2/lite/protocol.py#L160-L186
[mlite-glm-protocol]: ../megatron/lite/model/glm5/lite/protocol.py#L200-L226
[mlite-ds4-protocol]: ../megatron/lite/model/deepseek_v4/lite/protocol.py#L357-L386
[mlite-mcore-config]: ../megatron/lite/primitive/optimizers/megatron_wrap.py#L256-L282
[mlite-fsdp-master]: ../megatron/lite/primitive/optimizers/fsdp2/adamw.py#L150-L173

[resync-block]: https://github.com/ISEEKYAN/Megatron-LM/blob/a89f1bd4f8d4fb6c28d4caceac1ed25642e75522/experimental/lite/megatron/lite/primitive/quantization/block_fp8.py#L13-L92
[resync-model]: https://github.com/ISEEKYAN/Megatron-LM/blob/a89f1bd4f8d4fb6c28d4caceac1ed25642e75522/experimental/lite/megatron/lite/model/deepseek_v4/lite/resync.py#L25-L93

[skill-constitution]: ../skills/basic/constitution.md#L23-L42
[skill-principle]: ../skills/primitive/principle.md#L20-L31
[skill-contract]: ../skills/primitive/contract.md#L22-L60
[skill-design]: ../skills/primitive/design.md#L19-L40

# DeepSeek-V4.1-Flash on Megatron-Lite

40-layer CED MoE model with Engram n-gram memory. `model/deepseek_v41/` is a thin
assembly over shared primitives; reusable logic lives in `primitive/`
(`optimizers/{sinkhorn,headwise_muon}`, `modules/{engram_lookup,router_replay}`,
`quantization/*`, `parallel/*`).

## Semantics that differ from DeepSeek-V4
- **mHC shift**: attention consumes `pre_mix`, FFN consumes `attn_pre`, the block
  returns `ffn_pre`; reusing the current block's coefficients is wrong.
- **CSA2 quantization**: main KV uses group 16 with E4M3 scales, the indexer uses
  group 32 with E8M0; mixing them is wrong.
- **Contiguous CP**: `pack_routed_experts` / `pack_r3_replay_mask` must pass
  `contiguous=True`; the zigzag default misroutes silently.
- **Optimizer routing** is by logical matrix shape, never parameter name: `wq_a`
  is one shared matrix, `wq_b` is 64 independent ones, `wkv` is one shared latent
  K/V head. Sinkhorn (Algorithm 1): K=11, tau=1e-3, eps=1e-20, momentum=0.95,
  gamma=0.18, Engram LR 5x, no warm start.
- **Post-training scope**: the indexer stays frozen and out of the optimizer;
  Engram supports frozen (FP8 only) and trainable (persistent FP32 master) under
  one switch, and is never offloaded.

## mHC execution
mHC uses the original pure PyTorch implementation, including FP32 accumulation,
source-to-destination residual orientation and the shifted layer boundary.
The attempted Core-kernel delegation was reverted: making these operations depend
on an optional Core installation broke standalone CPU execution and CED tests.
V4.1 mHC execution does not require `megatron.core`.

## Official reference
Oracle comparisons load the pinned upstream source from `DS41_REFERENCE_DIR` at
test time and verify SHA-256; no official source is vendored into this repo.

## Assembly example
```python
import os
from megatron.lite.model.deepseek_v41.lite.protocol import ImplConfig, build_model, build_model_config
config = build_model_config(os.environ["DS41_MODEL_DIR"])
bundle = build_model(config, impl_cfg=ImplConfig(device="meta"))
print(sum(p.numel() for p in bundle.chunks[0].parameters()))
```
Meta construction inspects assembly; execution needs a tokenizer-derived Engram
map and materialized weights. Select post-training trainability explicitly.
Pure data parallel text training uses PyTorch DDP with the explicit V4.1 Muon
optimizer. SFT normalization counts valid tokens across all replicas; successful
steps update routing biases from their combined load statistics. Engram storage
stays on the model device in both frozen and trainable modes. Changing vision
trainability requires rebuilding the DP bundle; external staged vision is not
yet supported with DP.

The two-GPU regression preserves all 40 layers with reduced dimensions in the
floating diagnostic mode. It checks two optimizer steps, unequal token counts,
replica equality, and comparison with a single-process global batch. This does
not establish full-size or native quantized training support. TP/VPP/ETP remain unsupported. Text-only PP2 model forward/backward is
available separately, with the limits described below.


## Contiguous context parallel text training
Use `ParallelConfig(cp=2)` in an initialized two-rank world, keeping TP, EP,
PP and VPP at one. The protocol receives the complete packed batch, shifts
labels and loss weights within each document, then assigns one contiguous
interval per rank. Uneven lengths pad transport only. Each rank visits every
document, including empty intersections, and hashes complete document history
before selecting its Engram rows. CSA2 queries and selections are local; window
KV, compressor groups and shared KV remain document-global.

Communication uses the shared differentiable CP gather and DDP gradient
averaging. This correctness path materializes full-document KV; it does not
provide fused sparse attention's memory or throughput characteristics. CP
modality/replay inputs and CP combined with other parallel dimensions are
rejected. In particular, CP>1 with EP>1 raises
`CP_AND_EP_NOT_SIMULTANEOUSLY_SUPPORTED`. TP/VPP/ETP, PP sizes other than 1 or 2, and custom pipeline
layouts raise `V4.1_UNSUPPORTED_PARALLELISM`, listing the rejected settings.
Floating tests retain the 40-layer assembly, all three CSA2 modes,
frozen indexers, and nonzero frozen/trainable Engram tables. The strict reference
preserves local operator shapes and the gather backward reduction order;
parameter contributions are checked before DDP averaging as well as after it.


## Expert parallel text training
Set `ImplConfig(parallel=ParallelConfig(ep=2), optimizer="muon", ...)` after
initializing the distributed world. Each rank allocates only its contiguous
expert interval; module names retain global expert indices. Both EP=1 and EP>1
use the shared `TokenDispatcher`. This assembly selects native all-to-all;
the primitive also retains its DeepEP transport and participation checks.
Native checks use `ep_group`; DeepEP checks use its buffer's `tp_ep_group`.
Empty expert chunks preserve dispatch autograd edges and every rank participates
in both forward and backward collectives.

Dense parameters use DDP. Expert gradients sum only over replicas of the same
expert and use the dense DP loss scale. The clipping norm counts each dense
parameter once plus all expert shards; nonfinite gradients and failed optimizer
candidates prevent publication on every rank. Invoke the bundle's
`finalize_grads` after the microbatch loop and before the optimizer step, as the
MLite runtime does.

The EP regression uses an independent serial global batch and exact comparisons.
Serial autograd supplies each expert projection's inputs and output gradients;
the reference concatenates them in source-token order before its weight-gradient
GEMM, matching the EP reduction order. It first reconstructs the unmodified
serial gradients exactly. This avoids comparing separately rounded microbatch
GEMMs with one combined GEMM. FP64 accumulation diagnostics retain their raw
residuals and also check exact agreement after rounding to FP32 masters.
The test alternates an empty receiving rank with active experts on both ranks,
and independently checks nonfinite-gradient skips, candidate rejection, missing
participants, and EP2 with two replicas per expert. These are reduced-dimension
floating tests, not full-size native-quantized or combined TP/CP/PP validation.


## Weight export
The registered protocol accepts the Verl engine's `export_dtype`, `cpu` and
`buffer_max_size_bytes` options. `export_dtype` casts active plain FP32/FP16/BF16
weights; encoded Engram FP8 tables/scales and inactive archival payloads retain
exact bytes. `cpu=True` returns CPU tensors; otherwise tensors use the model device.
Conversion copies use the buffer budget, and HF save reuses the shared safetensors
shard writer with that shard budget. A single named tensor is indivisible and may
exceed the budget; this is not a hard bound on the returned tensor's memory.
The default save accepts the engine's precreated empty directory. Existing nonempty
checkpoints are not overwritten. With no options, the lossless archival save remains
byte-streamed. Deployment conversion options such as `target` and `resync_config`
are rejected by name; quantized rollout conversion is not implemented here.

For native SFT, the protocol supplies the global next-token denominator; the
Verl adapter does not apply its token-count scale a second time. An external
loss callback owns its objective normalization, with the existing microbatch
multiplier cancelling the runtime's microbatch averaging.

HF checkpoint saves and online weight exports retain trainable masters and
archival bytes even when the engine has a rollout resync format configured.
Both engine paths use the protocol's `HF_SAVE_SUPPORTS_RESYNC=False` capability
to omit deployment conversion options and emit native HF weights. This does
not implement quantized rollout conversion; consumers must accept native weights.
Direct unsupported export keywords still raise `TypeError` naming the keyword.

## Validation loss and release caveats

Forward-only runtime loss is `sum(microbatch_loss / num_microbatches)`, matching
training's backward scaling, for both PP and non-PP dispatch. Native SFT runs
`prepare_microbatches` in validation as well as training: it only prepares the
shared next-token denominator and does not require gradients. With unequal token
counts this yields the token-weighted objective, rather than a mean of local
means. External loss callbacks retain ownership of normalization and bypass this
hook. The VERL adapter multiplies caller-normalized loss contributions by the
microbatch count, so runtime averaging recovers their sum. Its inference callback
returns zero loss and collects per-microbatch outputs separately; token outputs
are not combined through the runtime scalar loss.

Returning only the last validation microbatch loss was inherited from main.
Excluding forward-only execution from the preparation hook was introduced with
this integration. Both behaviors are corrected together without changing the
training gradient scale or external callback metrics.

Known limitations retained for this release:

- **CP-AUX-SCALE:** the generic train-step auxiliary-loss hook still supplies
  `1 / num_microbatches`, assuming CP=1. It does not apply the CP group-size
  multiplier. CP text-path checks do not establish correctness of nonzero
  auxiliary losses under CP>1; that combination remains unvalidated.
- **hf-resync-unsupported:** deployment resync conversion remains unsupported.
  Online export and HF save omit `target` / `resync_config` for V4.1; direct
  unsupported export keywords raise `TypeError`, not a conversion result.
- **online-export-contract-test-gap:** the previous engine-to-export coverage
  gap is addressed by `test_v41_engine_online_export_resync_contract`. It consumes
  native weights through the runtime and real V4.1 exporter with configured
  resync, plus supported/legacy capability controls. It is a CPU contract test,
  not an end-to-end rollout synchronization or quantized-consumer acceptance.
- **R3-WARN-0 / R3-EV-003:** `changed=0` is a warning, not an error; replay
  evidence is logged once per driver, not once per step.
- **R3-EV-001 / R3-EV-002:** cross-model contracts use `_TinyChunk` and do not
  establish full-model execution liveness. Record mode has no `VOID` gate;
  replay with no observed routes is rejected by `R3_REPLAY_VOID`. Review found
  these checks neither fabricated nor unconditionally passing; their scope
  remains narrower than full-model integration.
- **CP/PP-REPLAY-GATE / CP-EP-EXCL:** unsupported CP/PP replay paths and CP+EP
  combinations fail fast, rather than silently misrouting. PP>1 record mode
  raises `NotImplementedError`; PP2 also rejects distributed optimizer config.
- **ds41-no-compose-skill / ds41-doc-skill-gap:** model-composition and
  documentation skill coverage remains incomplete. Workflow checks do not
  establish end-to-end model acceptance.
- **mhc-pytorch-path / hasattr-gates:** mHC retains the pure PyTorch path;
  optional protocol behavior uses capability/attribute gates with compatibility
  defaults. These are integration limitations, not fused-kernel performance or
  exhaustive capability-validation claims.

## Resident Engram row owners

`ImplConfig(shard_engram=True)` is the default. Each DP/CP group rank stores
only its contiguous row interval, including with EP enabled and CP=1. Uneven
tables use boundaries `total_rows * rank // owner_count`; no padding rows are
added to persistent storage. `shard_engram=False` selects replicated tables for
small diagnostic comparisons. Existing parallelism restrictions still apply.

`trainable_engram=False` keeps only FP8 weights and E8M0 scales. Setting it to
`True` adds a persistent local FP32 master; its FP32 gradient has the same local
shape and aliases `main_grad`. Owner masters stay outside dense DDP buckets.
Lookup backward sums owner requests, gradient finalization normalizes once, and
Sinkhorn reduces logical-table statistics across owners. Local training
checkpoints retain this ownership and require the same topology on resume.
HF export currently assembles complete tables and is not a bounded-memory
export for the official dimensions.

The transport reuses NVIDIA-NeMo/Automodel `8a646a739d30ed99ac022e39541e7db8c35ab2db`
and the verified Qwen3.8 owner implementation: fixed-capacity All-to-All,
source-rank request order, stable sorting, count agreement, and received-ID
validation. Sorted-owner validation uses row boundaries to support nondivisible
tables. FP8/scale byte transport and the existing FP32 index gradient arithmetic
remain separate from the floating PLE lookup.

For the official two tables (384006168 and 384016682 rows, width 256), the
maximum persistent storage per rank is below, in GiB. These are arithmetic
bounds, excluding optimizer state, activations and temporary buffers.

| Owners | FP8 + scales | FP32 master, if trainable | FP32 main gradient |
| --- | ---: | ---: | ---: |
| 1 | 188.833133 | 732.443666 | 732.443666 |
| 8 | 23.604142 | 91.555459 | 91.555459 |
| 16 | 11.802071 | 45.777730 | 45.777730 |
| 32 | 5.901036 | 22.888865 | 22.888865 |

Distributed tests use reduced tables and check logits, losses, gradients and
updates at zero tolerance, with matched physical parameter/bucket layouts for
trainable comparisons. The reference gathers rows and requests independently
of the production All-to-All. This does not establish official-scale training
capacity or bitwise equivalence across different DDP bucket geometries.

## Text-only PP2 model forward/backward

In an initialized two-rank world, use:

```python
ImplConfig(
    parallel=ParallelConfig(pp=2),
    pipeline_split_layer=20,
    text_only=True,
    optimizer=None,
    # Set device, dtype, token_map and quantized mode as usual.
)
```

The model uses Megatron-core's pipeline layout API. Stage 0 owns text layers
0–19, the embeddings, both Engram tables (layers 1 and 14), and the inactive
vision encoder/aligner. Stage 1 owns layers 20–39 and the final norm/head.
`pipeline_split_layer` is explicit, but values other than 20 (including 15) raise
`V4.1_PP_CSA2_PAYLOAD_UNSUPPORTED`: those cuts need additional CSA2 owner state.
At layer 20, CSA2 recreates its KV/index state, and the saved CED pair is exactly
the layer-19 hidden/pre-mix pair.

The pair travels as one three-dimensional FP32 tensor through the model's
`set_input_tensor` interface. FP32 preserves native pre-mix values and gradients
even when residual activations are BF16. The bundle publishes `pipeline_dtype`,
and the runtime passes it to the shared pipeline schedule. Other models keep the
existing BF16 communication default. This follows NVIDIA/Megatron-LM
`dev@68447eeaae0b8b5300dc5e3ae8d91d8a0753fae6`, fetched
2026-09-14T04:11:46Z, where P2P receive buffers use `config.pipeline_dtype`.

Only text-only PP2 at 20/20 is covered, using the floating diagnostic mode.
PP with EP/CP, PP optimizer training, and distributed HF checkpoint assembly are
not validated. Multimodal PP remains rejected by `V4.1_PP_TEXT_ONLY`; this does
not add pipeline support for staged backward callbacks. Both stages contain
text layers, so neither is an encoder-only stage. The upstream no-image backward
and encoder-only CP scaling cases are reference context, not multimodal PP
validation evidence.

The dedicated two-GPU test runs the real runtime and shared P2P schedule with
variable-length packed microbatches, nonuniform loss weights, and frozen/trainable
resident Engram tables. It compares logits, token-normalized losses and local
parameter gradients with the complete model at zero tolerance. This establishes
model forward/backward behavior, not full-size or end-to-end training support.

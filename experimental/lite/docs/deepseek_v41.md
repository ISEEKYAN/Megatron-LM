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
not establish full-size or native quantized training support. `build_model` still
rejects PP > 1 (and CP/VPP/ETP); local pipeline range helpers are not a supported
PP runtime.


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


## Tensor parallel text training

After distributed initialization, select
`ImplConfig(parallel=ParallelConfig(tp=2), optimizer="muon", quantized=False, ...)`.
The initial integration requires a TP-only world: DP/EP/CP/PP/VPP/ETP combinations
remain rejected. Column shards own attention's `wq_a`, `wq_b`, `wkv`, `wo_b`,
compressor projections and the vocabulary head. Shared primitive collectives
gather projection outputs and sum input gradients. Grouped `wo_a`, ETP=1
experts, embeddings and attention operations remain replicated.

FP32 masters and native FP32 weight gradients remain local during forward and
backward. At each optimizer step, a primitive gathers logical masters and
gradients, then publishes each rank's updated slice. Muon and Sinkhorn operate
on complete matrices: `wq_a` remains one shared matrix and `wq_b` retains its
per-head matrix grouping. Optimizer momentum is currently replicated and the
step temporarily materializes full matrices. This implementation reduces model
parameter storage and projection work, without claiming sharded optimizer memory.

The regression retains all 40 reduced-width layers, all three CSA2 modes,
CED/mHC, frozen indexers and both Engram modes for two real optimizer steps.
It records native FP32 trajectory differences, then requires exact agreement
with an independent single-process full-matrix FP64 oracle after FP32 publication.
Only projection accumulation precision changes for that diagnostic; the real
TP collectives and full-matrix optimizers still execute. A separate sensitivity
probe replays measured gradient perturbations through an identical full-matrix
optimizer to check the resulting parameter and next-forward differences. This
probe is not the independent parity reference. The tests also require real
collective calls, reduced local parameter counts and FP32 gradient low bits.
Native quantized/full-size training is not established by these floating tests;
FP8 column shards must preserve complete 32-row blocks.

HF export gathers full logical tensors once. All TP ranks participate in save;
TP rank zero writes the checkpoint. Load slices full tensors exactly once for
TP owners, and TP1 loads them without slicing. Tests save at TP2 and reload at
TP1 with bitwise parameter/buffer checks, then reload at TP2 to detect double
slicing. Optimizer state serialization retains the full logical matrix layout.

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

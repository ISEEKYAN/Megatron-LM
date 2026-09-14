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
rejects PP > 1 (and TP/CP/VPP/ETP); local pipeline range helpers are not a supported
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

HF checkpoint saves retain trainable masters and archival bytes even when the
engine has a rollout resync format configured. The engine uses the protocol's
HF-save capability to keep deployment conversion options out of this native
checkpoint path. Direct unsupported export options still fail explicitly.

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
- **HF-RESYNC:** direct deployment resync calls can pass unsupported keywords to
  the fixed V4.1 protocol signature and raise `TypeError`. Native HF checkpoint
  save is separate and preserves masters; deployment resync is not supported.
- **PP replay and optimizer:** PP>1 record mode raises `NotImplementedError`;
  the PP2 assembly rejects distributed optimizer configuration.
- **Replay evidence:** zero changed routes produces a warning. Record mode has
  no replay-equivalent execution probes, and the runtime unit seam does not
  establish an end-to-end R3 integration assertion. Replay with no observed
  routes is rejected by `R3_REPLAY_VOID`.
- **Workflow coverage:** skill routing and model-composition guidance remain
  incomplete (there is no model-compose leaf). Existing workflow checks are not
  end-to-end acceptance; cross-model R3 tests use `_TinyChunk` and do not prove
  execution through every full model assembly.

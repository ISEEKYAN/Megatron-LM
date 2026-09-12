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

## Official reference
Oracle comparisons load the pinned upstream source from `DS41_REFERENCE_DIR` at
test time and verify SHA-256; no official source is vendored into this repo.

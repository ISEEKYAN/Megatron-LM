# Hy3 architecture and Megatron Lite mapping

This document freezes the public `tencent/Hy3` architecture before the native
Megatron Lite implementation is changed.  It follows the staged model-porting
workflow used by the existing Qwen3 MoE implementation: establish an
independent reference, map capabilities to reusable primitives, then implement
weights and distributed validation.

## Reference freeze

| Input | Revision or digest |
| --- | --- |
| Hugging Face repository | `tencent/Hy3@716aa7241bd6d95896be4ebfc761162a9c4d49ef` |
| `config.json` SHA-256 | `663036ceca3d8a178cd772739566c262caffdecebaed6c1d76b464d729bb2951` |
| `model.safetensors.index.json` SHA-256 | `9594f1a9419e62ca7afca51bb644f38ef19039374f7812449381ccf42f0ef79b` |
| Reference implementation | Transformers `hy_v3` (`transformers_version=5.6.0`) |

Only the public configuration and weight index are needed for the structural
contract.  The 598 GB checkpoint is deliberately not copied into the source
tree.

## Architecture manifest

| Area | Hy3 contract |
| --- | --- |
| Decoder | 80 causal decoder layers plus one MTP layer; layer 0 is dense and layers 1-79 are sparse |
| Hidden/normalization | hidden size 4096; RMSNorm epsilon `1e-5`; untied embedding and LM head |
| Attention | GQA with 64 query heads, 8 KV heads and head dimension 128; bias-free projections; Q/K RMSNorm before RoPE |
| Position | default RoPE, theta `11158840`, maximum sequence length 262144 |
| Dense MLP | SwiGLU, intermediate size 13312, no bias |
| Routed MoE | 192 experts, top-8, expert intermediate size 1536, SwiGLU |
| Router | fp32 logits, sigmoid scores; add persistent `expert_bias` only for expert selection; gather un-biased sigmoid scores, normalize selected scores, then multiply by 2.826; no auxiliary loss in the Transformers reference |
| Shared expert | one always-on, ungated SwiGLU MLP with intermediate size 1536; add to routed output (the released config disables fp32 combination) |
| MTP | one layer at checkpoint layer index 80 with `enorm`, `hnorm`, `eh_proj`, a sparse decoder layer and `final_layernorm` |
| Vocabulary | 120832 tokens; separate embedding and output head |

The checkpoint index contains 47,138 tensors.  Sparse layers use
`mlp.router.gate.weight`, `mlp.expert_bias`, `mlp.shared_mlp.*` and per-expert
`gate_proj`/`up_proj`/`down_proj`; layer 0 instead has the three dense MLP
weights.  The MTP layer has the same attention and sparse-MoE names plus its
four MTP-specific tensors.

## Primitive mapping

| Hy3 capability | Megatron Lite implementation | Decision |
| --- | --- | --- |
| GQA, Q/K norm, RoPE, TP/CP/THD | `primitive.modules.GQAttention` | Reuse directly |
| Routed expert compute | `primitive.modules.Experts` | Reuse directly |
| EP/DeepEP dispatch and combine | `primitive.modules.TokenDispatcher` | Reuse directly |
| Sigmoid top-k selection | `primitive.modules.SigmoidTopKRouter` | Extend its generic contract to persist checkpoint expert bias and accept Hy3 field names |
| Dense/shared SwiGLU | `primitive.modules.SwiGLUMLP` | New generic primitive; no Hy3-specific behavior |
| MTP mechanics | `primitive.modules.MTPBlock` and `MTPDecoderLayer` | Reuse directly |
| TP/SP/CP/EP/PP/VPP and THD | `primitive.parallel` and the Qwen3-MoE protocol shape | Reuse directly |
| HF/DCP checkpoint machinery | `primitive.ckpt` | Reuse; Hy3 contributes only its WeightSpec |

No MLA, GDN, DSA, attention sink, attention gate, or KV-mirror behavior is
present in the released architecture.  Model code therefore only selects the
dense/sparse layer type and composes the primitives above; it must not contain
its own attention, router, dispatch, or parallel communication implementation.

## Staged validation

1. CPU/static: config invariants, registry resolution, exact router semantics,
   layer-type selection, WeightSpec coverage, and synthetic HF/native/HF tensor
   round-trip.
2. Single GPU (Slurm): a tiny Transformers `HYV3ForCausalLM` reference versus
   the native model, checking router choices, each layer output, logits, loss,
   and gradients.
3. Distributed (Slurm): TP2, EP2, CP2+THD, PP2 and a CP2xEP2 combination, with
   explicit assertions that routed experts and the requested communication
   paths did not fall back.
4. Training/IO (Slurm): checkpoint load/export/reload, DCP resume, deterministic
   optimizer steps, and a short loss-decreasing SFT run.

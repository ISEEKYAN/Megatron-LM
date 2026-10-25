# Deployment-aligned DS4.1 LoRA

This example builds `impl="vllm"`, runs actual training steps on real tokenized
text, and writes adapter-only safetensors through the normal Lite runtime and
Core distributed Adam optimizer. The base model and Engram tables stay frozen. Attention, shared
experts and routed experts use the same `LoraConfig` and adapter owners as the
Lite implementation. Deployment forwards call native vLLM LoRA operations with
explicit adjoints; no vLLM kernel source is copied here.

Use a compatible ISEEKYAN/vllm build at
`4d69bb528888017c7998b97972405c70076533f4` (`v62365/lora-overlap`), based on
vllm-project/vllm#62365 with the native LoRA/DP-filler fixes and the BI activation
op. Stock vLLM is insufficient. Use the native FP8 wo_a model configuration,
sparse FlashMLA, MegaMoE and the paired BI settings. Mega attention is outside
the validated scope.

With `experimental/lite` on `PYTHONPATH`, run the example through a distributed
launcher, for example on a single eight-GPU host:

```bash
torchrun --nproc-per-node=8 experimental/lite/examples/deepseek_v41/lora_vllm.py \
  --model /path/to/native-model --ep 8 --layers 20 40 \
  --token-ids /path/to/real-text-token-ids.json --steps 1 \
  --output /path/to/adapter.safetensors
```

The input is a JSON array of token IDs from real text; each step uses shifted
next-token labels. The example reports the optimizer result and writes the
post-update adapter factors on rank0, after collective parameter synchronization.
It is a minimal training/export example, not a DAPO recipe. Omitting `layers` from a
`LoraConfig` selects the last twenty layers (or all layers of a smaller model).
The default target set covers `wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, shared experts
and routed experts. Use `target_modules` for a subset; dropout is unsupported.
The wo_a delta consumes the native prequantization BF16 inverse-RoPE result;
the base projection retains its FP8 path. Routed down deltas use the native
BF16 activation that already includes routing weights, so combine does not
multiply those weights again.

After building, use normal runtime `forward_backward`, `zero_grad`, and
`optimizer_step` calls. `MegatronLiteEngine.get_per_tensor_param` exports adapter
factors after parameter synchronization; the VERL connector packages versioned
manifests and installs them through the existing bucket transport. Configure
`hf_overrides.mlite_base_manifest_sha256` to the frozen base manifest SHA256 and
keep the base preloaded. Each publication uses VERL's fixed adapter identity;
policy versions, tensor hashes and receiver ACKs identify the actual generation.
The receiver rejects a wrong base, corrupt or partial factors, or skipped policy
versions. The reference policy uses `engine.disable_adapter()`.

Validated numerical scope: rank 32 / alpha 32, all targets, actor EP8/PP1/CP1 and
native rollout DEP4 x 2; FP8 wo_a + sparse FlashMLA + MegaMoE. The two-step checks
compare actual response-token FP32 log-probability bit patterns in rollout,
old-policy scoring and grad-enabled actor scoring at each of three generations.
These checks establish update/publication correctness, not convergence, PP/CP
LoRA parity or multiple concurrent adapters. The full 40-layer/384-expert release
was additionally checked
with rank32 adapters on layers20–39 under KV-only hybrid rollout: three policy
generations, 384 actual response tokens, 24 receiver ACKs and two finite nonzero
updates, with identical FP32 response LP bits. Long-response DAPO throughput and
reward improvement remain separate validations.

Deployment LoRA currently requires `attention_core="flashmla"`; the optional MegaAttention core remains available without LoRA.


For long-context jobs that cannot keep both bases on GPU, set VERL's
`actor_rollout_ref.rollout.custom.ds41_lora_sleep_mode=weights_cpu` explicitly.
This selects vLLM's native level-1 sleep, which backs up and releases base weights,
and restores both `weights` and `kv_cache` before generation. The default remains
`kv_only`. Account for the native CPU backup in the per-node host-memory budget;
this setting does not offload the actor or bypass memory-admission checks. Never
use level 2 to discard the frozen base for adapter-only publication.

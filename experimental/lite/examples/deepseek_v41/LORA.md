# Deployment-aligned DS4.1 LoRA

This example builds `impl="vllm"` on the normal Lite runtime and Core distributed
Adam optimizer. The base model and Engram tables stay frozen. Attention, shared
experts and routed experts use the same `LoraConfig` and adapter owners as the
Lite implementation. Deployment forwards call native vLLM LoRA operations with
explicit adjoints; no vLLM kernel source is copied here.

Use a compatible ISEEKYAN/vllm build at
`ad140fc150a7216ed3dd4a8d3df4b37e4de6fe33` (branch
`v3/ds41-lora-dpfix`), including the BI activation op introduced by
`8a7fa58d768ac5db565470aff8d33e6fcee75484` and FP8 wo_a LoRA support from
`013286420`. Stock vLLM is insufficient. Use the paired native FP8 wo_a model
configuration, sparse FlashMLA, MegaMoE, and the BI settings documented for the
base implementation. Mega attention is outside the validated scope.

With `experimental/lite` on `PYTHONPATH`, run the example through a distributed
launcher, for example on a single eight-GPU host:

```bash
torchrun --nproc-per-node=8 experimental/lite/examples/deepseek_v41/lora_vllm.py \
  --model /path/to/native-model --ep 8 --layers 0 4
```

For the 40-layer E16 proxy, pass `--layers 0 40`. Omitting `layers` from a
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
LoRA parity, multiple concurrent adapters, or full 384-expert 40-layer training.

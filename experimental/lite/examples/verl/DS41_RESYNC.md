# DS4.1 native deployment resync

This adapter is stacked on the bounded row checkpoint exporter. Keep
`experimental/lite` and `experimental/lite/examples/verl` on every worker's
`PYTHONPATH`. It requires the vLLM layerwise reload API and DS4.1 native loaders
(reference vLLM `a7fda4c88bfc421d31e33acc5e01e86ebe467ad8`, verl
`3efe38c759c14622fd1b2c9e3679f2d02f86bdac`). No verl source changes are required.

For a DS4.1 MLite actor, add these overrides to the existing engine recipe:

```text
actor_rollout_ref.actor.engine.model_name=deepseek_v41
actor_rollout_ref.actor.engine.resync_format=mxfp4
+actor_rollout_ref.actor.engine.resync_config.expert_dtype=fp4
+actor_rollout_ref.actor.engine.impl_cfg.optimizer=muon
+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=verl_mlite.rollout.deepseek_v41.DeepseekV41WorkerExtension
actor_rollout_ref.rollout.pipeline_model_parallel_size=1
```

Retain the DS4.1 optimizer configuration required by the training recipe.
`mxfp4` is the existing engine format selector; protocol callers may also use
`target='vllm'`. Both `save_hf_weights` and `export_hf_weights` use the same mixed
checkpoint encoding: FP4 experts, FP8 linear/Engram weights, E8M0 scale siblings,
BF16 plain weights, and FP32 router/sink/mHC controls. `expert_dtype='fp8'`,
partial exports, and arbitrary plain export dtypes are rejected by both entries.

Online tensors carry versioned row-offset metadata in their names. The worker
extension must be installed **before initial model construction**: it records
original vLLM loader metadata before kernel repacking, copies Engram row
intersections into resident hash-head shards, and defers model finalization
until the whole generation arrives. Use this extension instead of the default
BF16-to-quantized receiver. Save writes ordinary full-shape HF safetensors and
config, without transport names or training-master sidecars; it is a deployment
snapshot, not a lossless optimizer/training resume checkpoint. Use target=None
for the latter.

All exporting ranks must drain the stream. Consumers must copy borrowed row
payloads before advancing. The export budget includes row packing and iterator
handoff overlap; a non-row matrix whose codec workspace cannot fit is rejected.
The receiver's resident weights and per-layer vLLM repacking allocations are
separate from the exporter staging budget.

Validation scope is reported in the PR: payloads and scaled row proxies do not
establish full-size rollout quality. Multi-node, large EP, CUDA graph refit,
CPU offload, speculative drafting, LoRA, and alternative expert kernels require
separate validation. The extension rejects drafting and adapter-only sync.

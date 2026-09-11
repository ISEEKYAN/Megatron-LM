# V4.1 single-rank semantics

The modules in `megatron.lite.model.deepseek_v41.lite` implement CSA2 ownership,
shifted hyper-connections, Engram computation and modality-specific expert
selection. They are building blocks; a distributed model factory, sharded tables
and optimizer-step publication are separate integration work.

- `CSA2Attention(config, layer_id)` returns output and explicit `AttentionState`.
  Layers 2/8/14/20 own KV; layers 24/28/32/36 reindex the layer-20 candidate pool;
  intervening layers reuse it. Indexer parameters are frozen and attach no loss.
- `DeepseekV41Block.forward_with_state(hidden, pre_mix, state)` returns all three
  values. Preserve this complete boundary during checkpointing and transport.
- `EngramTable(values, scales, trainable=False)` stores only FP8 values/scales.
  The same switch set to true adds a persistent FP32 master with STE gradients.
  FP32 master representation is a port choice, not an official training recipe.
  Call `refresh_storage()` after an update. Tables stay on device; no offload.
- `ModalityRouter` reuses the existing DS4 router's score/reduction policy,
  disables auxiliary loss, and returns detached per-modality statistics.
  Accumulate these externally and call `update_bias()` once per optimizer step.
  Forward and recomputation do not mutate the selection biases.
- `packed_forward` runs each unpadded THD sequence with independent state.
  It is a correctness path, not fused packed attention or distributed CP.

The three `ds41_*` quantization modules are the minimal codec dependencies for
these operators. They reuse existing MXFP4 and block-FP8 primitives; Core is an
environment dependency and is not copied or modified by this change. Native
FP8 linear requires CUDA. Disabled-quantization diagnostics do not establish
native quantized-kernel parity.

This change is stacked on PR #212, commit
`fd18aa0d02bdfbeb8ff1daf035e46dd9d9911371`, and must merge after #212.
The oracle tools, pinned reference and fixture manifest come from that base.

Focused cases are parameterized in
`tests/unit/model/test_deepseek_v41_semantics.py`. The shifted-HC test independently
derives two consecutive blocks with 2/3/4 unequal copies and nonzero residual
updates. The CED test executes the pinned official `Block.hc_pre`, `RMSNorm`,
`Compressor.forward` and the pre-RoPE portion of `Indexer.forward`, using C's
fixture dimensions, weight recipe and oracle recorder. It compares `x20`,
`latent20` and `index_k20` from the actual MLite block/attention call, at FP32 and
BF16 with sequence lengths 1 and 9, with zero tolerance. A hook ends the official
indexer after `k_norm`; this tests the three CED equations, not the subsequent
native RoPE/FP4 kernels or a complete official model forward.

```text
x20       = attn_norm_20(hc_pre(h20, p20))
latent20  = compressor_norm_20(compressor_wkv_20(x20))
index_k20 = k_norm_20(wk_20(latent20))
```

Run the focused cases with:

```bash
PYTHONPATH=experimental/lite OMP_NUM_THREADS=1 python -m pytest -c /dev/null \
  --confcutdir=experimental/lite --rootdir=experimental/lite -q \
  experimental/lite/tests/unit/model/test_deepseek_v41_semantics.py
```

The D-only review diff is against `fd18aa0d0`. Regression acceptance retains
`d8e010069` (mainparent job 18354480) as the baseline and compares both
failure/error set differences against it. Test skips remain
skips, and added passing semantic cases do not erase baseline failures.

## Text-only model assembly

`DeepseekV41Config` reads the nested release JSON without flattening or dropping
inactive metadata. `deepseek_v41` is registered with the `lite` protocol. The
single-rank model composes all 40 D blocks, both Engram modules, embedding, final
shifted-HC contraction, norm and head. `ImplConfig.token_map` is required for
Engram execution; real token IDs must use the tokenizer-derived map, while the
C reduced fixture deliberately uses its specified identity map. `quantized=False`
is an explicit floating diagnostic. Quantized execution uses D's native FP8
linears and group32 FP4 numerical/STE expert providers; it does not claim native
FP4 GEMM performance. CPU diagnostics are not GPU parity evidence.

`model.parameter_bindings()` enumerates actual live parameter objects once,
including frozen indexers. Each binding exposes `owner`, `attribute`, `role`,
`tensor`, and the query projection's `head_count`. Consumers must inspect these
objects and roles instead of inferring optimizer groups from checkpoint names.
`bind_checkpoint(model, records, store=...)` validates coverage and active
shape/dtype/scale layout, then links keys to those same owners, headers and store.
The C manifest exercises all 3,204 reduced entries with explicit
`allow_missing_mtp=True`; production loads require the complete MTP key set. A separate meta allocation
checks all 96,085 release keys from the A2 contract, including 2,401 MTP keys;
that key-only check does not claim real release-header or payload inspection.

`save_model` / protocol `save_hf_weights` stream lossless active masters and
unchanged archival payloads. Plain master weights have no quantization scale
siblings, following C's plain-export decoder contract. Frozen Engram storage
keeps its FP8 values/scales. Trainable tables export FP32 masters and regenerate
resident storage on reload; callers must use `refresh_storage()` after accepted
optimizer steps. Original nested config is retained. This is a training export,
not a deployment quantization conversion. Saving requires the inactive MTP
bytes to be present; it cannot fabricate checkpoint data for unimplemented modules.

The model exposes live `vision`, `aligner`, `image_start`, `image_end` and
`image_newline` parameters through the same checkpoint binding contract.
`encode_image(patches, n_vit_h, n_vit_w)` is differentiable. The aligner pads
right/bottom and unfolds channel-first cells; 2D RoPE uses height then width.
`image_data.ImageConfig` defaults to patch size 14, downsample ratio 3, 1024
image tokens and 295936 minimum pixels. `prepare_image_inputs` consumes decoded
PIL images and token IDs, returning expanded IDs, token types and sample-local
`ImageInput` spans; URL loading and tokenization remain caller responsibilities.
`forward(..., images=..., token_types=...)` replaces spans before HC expansion,
propagates modality masks through MoE/Engram, and rejects spans crossing packed
sample boundaries. `merge_image_embeddings(images, h)` returns a new tensor.
The generic `PackedBatch` protocol, trainability policy, optimizer routing and
three-stage external vision scheduling require the training integration.
`model.mtp` remains archival and `forward_spec` rejects DSpark execution.

Independent vision/processor tests read hash-checked reference files from
`DS41_REFERENCE_DIR` (default `/tmp/ds41-fixture-reference`). No new official
source is vendored. Reduced FP32/BF16 checks compare forward and every input/
parameter gradient; mutation checks exercise spatial order, padding, RoPE,
patch order, delimiters and the tower/aligner boundary. These are reduced
numerical semantics checks, not full-size model acceptance.

The protocol accepts unpadded `PackedBatch`, restarts CSA2/Engram state per sample,
shifts labels and masks within each sample and masks terminal targets. It returns
loss/log-probabilities and honors loss-context temperature/entropy. Distributed
construction, optimizer creation, and routing replay explicitly require their
separate integrations. No distributed or full-size training claim is made here.

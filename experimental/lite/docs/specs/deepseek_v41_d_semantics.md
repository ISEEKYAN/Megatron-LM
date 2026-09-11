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

Focused cases are parameterized in
`tests/unit/model/test_deepseek_v41_semantics.py`:

```bash
PYTHONPATH=experimental/lite OMP_NUM_THREADS=1 python -m pytest -c /dev/null \
  --confcutdir=experimental/lite --rootdir=experimental/lite -q \
  experimental/lite/tests/unit/model/test_deepseek_v41_semantics.py
```

The delivery branch starts at `d8e010069`. Regression acceptance compares both
failure/error set differences against that exact parent. Test skips remain
skips, and added passing semantic cases do not erase baseline failures.

# DS4 resync: vLLM layerwise-reload lifecycle × MLite IPC bucket analysis

Reference: `mlite-2604-ds4-vllm020-thin` site, fetched 2026-07-13 — `vllm/model_executor/model_loader/reload/layerwise.py`, `vllm/model_executor/models/deepseek_v4.py`.

Evidence chain: parent TASK-1.1.12 r1-r11 (128-card colocated resync), receiver peak 53-58 GiB on vLLM worker TP0, bucket size 128 MiB-2 GiB not correlated (r9).

## 1. Layer completion key (vLLM reload lifecycle)

`initialize_layerwise_reload(model)` walks `model.modules()` and, for each submodule that owns loadable parameters:

1. Saves live kernel tensors in `LayerReloadingInfo.kernel_tensors`.
2. Restores parameters/buffers onto the **meta** device from captured metadata.
3. Wraps every parameter `weight_loader` with `online_process_loader`.

Each wrapped loader call:

- Buffers `(param_name, bound_args)` in `info.loaded_weights`.
- Adds `get_numel_loaded(original_loader, bound_args)` to `info.load_numel`.
- When `info.load_numel >= info.load_numel_total` (and the module is not `Attention` / `MLAAttention`), runs `_layerwise_process`:
  - materializes meta tensors on device,
  - replays buffered loads through the original loader,
  - runs quant repacking,
  - copies back into the original kernel tensor storage,
  - **resets** the layer info (freeing staging).

`finalize_layerwise_reload` processes any still-partial modules (attention last) and resets stragglers.

**Completion is per `nn.Module`, not per IPC bucket and not per HF `layers.N` prefix.** A single HF decoder layer maps to many vLLM submodules (`fused_wqa_wkv`, MoE experts, norms, MHC tensors, …).

HF names are normalized before routing: `DeepseekV4ForCausalLM.load_weights` applies `WeightsMapper` (`layers.` → `model.layers.`, `.scale` → `.weight_scale_inv` or expert `.weight_scale`, etc.) then `AutoWeightsLoader` dispatches into child modules.

## 2. MLite bucket HF name order

Export path (`export_hf_weights` → `export_resync_weights`):

- Native params iterate model order; PP>1 streams stage buckets over `pp_group` (stage 0 fully, then stage 1, …).
- Within a stage, tensors are layer-major in `named_parameters` order.
- `export_resync_weights` emits `(weight, scale)` pairs for quantized tensors using official V4-Flash names such as `layers.0.attn.wq_a.weight` / `layers.0.attn.wq_a.scale` (no `model.` prefix — mapper adds it on the vLLM side).

IPC sender (`_SyncBucketProducer` in `verl_mlite/compat.py`) packs the flat export stream into **byte-bounded** ZMQ buckets (`update_weights_bucket_megabytes`, historically 2048 → 128 in r6-r9). Buckets were allowed to span HF layer boundaries whenever the byte cap fired mid-layer.

## 3. Why staging did not release (r1-r11 root cause)

With layerwise reload active, each `model.load_weights(bucket)` call may touch multiple HF layers. Every touched vLLM submodule enters deferred staging until **its** `load_numel_total` is satisfied. When buckets interleave tensors from layers *N*, *N+1*, *N+16* (PP stage boundary) or split one layer across buckets with other layers inserted, **many submodules stay partial simultaneously**. Their `loaded_weights` tensors remain live until `_layerwise_process` — this is the observed ~37.5 GiB “arrival” copy stacked on top of the ~37.5 GiB kernel footprint (53-58 GiB total, independent of bucket size).

r10 (`MLITE_RECV_DIRECT_LOAD=1`, bypass lifecycle) proved memory headroom exists (**no OOM**) but failed on fused-parameter loaders (`load_merged_column_weight` missing) — lifecycle is still required for DS4 fused params.

r11 (`sleep_level=2` via Hydra) failed fast because `sleep_level` is not a veRL config key; vLLM was already at auto sleep level 2 per r9 logs.

## Fix (implemented)

**Receiver (primary):** `LayerClusterBuffer` in `verl_mlite/rollout/layer_cluster.py` — accumulate IPC buckets and call `model.load_weights` once per HF layer cluster (`layers.N` / `mtp.N` / embed / norm / head), so layerwise reload completes submodule staging before the next layer begins.

**Sender (secondary):** `_SyncBucketProducer` flushes when the HF layer cluster key changes, even under the byte cap, so ZMQ frames are layer-aligned when possible.

r12 (128-card) should reuse the existing `smoke128-d6aa45604-R-r*.submit.sh` family with `.bak` runtime patches reverted.

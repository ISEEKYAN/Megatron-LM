# DS4 fp8 resync: verl-project/verl#6473 vs our verl_mlite Fix-A

Reference: `verl-project/verl` PR **#6473** "[megatron] feat: support DeepSeek V4 GRPO",
merged 2026-07-14, merge commit `cf8005fdadabe99b3620a4b86a960a914d948b37`.
Read as a reference to check our Fix-A (IPC bucket offset byte-alignment) is not
incomplete for real DS4 fp8 resync. We are **not** switching to the upstream path
wholesale (our stack is verl_mlite-customized: mlite engine + self-built resync +
THD-PP fix + DSA + vLLM 0.25); this note only harvests what upstream teaches.

## 1. Why upstream does not hit our byte-alignment crash (the decisive answer)

Question that motivated this read: "upstream did not pad the IPC bucket offset, so how
does verl DS4 Megatron RL run without the `.view(dtype)` storage_offset crash?"

Code-level answer:

- **#6473 does not touch `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`.**
  It is not in the PR file list. The sender still does `offset += weight.nbytes` with no
  alignment padding, and the receiver still does `buffer[offset:offset+size].view(dtype)`.
  The mechanical defect our CPU test reproduces is present in upstream main too
  (independently confirmed against main HEAD `5b2bfe7c…`, tests/utils/test_bucketed_weight_transfer.py
  mixes only fp32/bf16/fp16 — no fp8, no odd numel — so it never trips).
- Upstream avoids the crash **architecturally, not by fixing the offset**: the trainer
  ships **BF16** weights over the IPC bucket, and fp8 quantization happens on the **vLLM
  receiver side**. See `vllm_rollout/utils.py::_update_weights` (FP8 branch):
  `load_quanted_weights(param_updates, self.model_runner, …)` with the comment
  "Convert bf16 weights to fp8 format before loading", plus
  `prepare_quanted_weights_for_loading` / `process_quanted_weights_after_loading`.
- Consequence: the upstream resync bucket **never carries a raw fp8 (itemsize-1) tensor**,
  so `offset` never lands odd and no subsequent BF16/FP32 `.view(dtype)` ever misaligns.
  This is exactly the "pure BF16 stream stays aligned → green" case our CPU test
  (`test_resync_bucket_byte_alignment.py`) already proves.
- The only place upstream transports already-fp8 bytes is `iter_deepseek_v4_weights`
  (init / HF-checkpoint load path): expert weights of dtype int8 / float8_e8m0fnu are
  `.view(torch.uint8)` (itemsize 1) so their own reconstruction is trivially aligned.
  This is not the RL resync hot path.

### Root cause of the divergence

- **Ours** (`megatron/lite/model/deepseek_v4/lite/resync.py::export_resync_weights`):
  quantizes bf16 → fp8 / mxfp4 on the **sender/trainer side** (`quantize_block_fp8`,
  `quantize_mxfp4`) and ships the **already-quantized fp8 tensor + a scale tensor** into
  the bucket. itemsize-1 fp8 tensors with odd numel ⇒ odd `offset` ⇒ the next non-fp8
  tensor's `.view(dtype)` crashes. This is the 128-real-weight crash.
- **Upstream #6473**: quantizes on the **receiver side**; bucket is fp8-free.

So Fix-A (pad `offset` to an 8-byte boundary) is *necessary* for our architecture and
*correct* — upstream simply never needs it because it never ships fp8. Fix-A stays fully
compatible with our sender-side-quant stack (CPU-test proven, byte-lossless).

## 2. What #6473 does beyond offset alignment (audit of `vllm_dsv4_fp8_utils.py`, 352 lines)

Relevant to correctness of an fp8 DS4 reload, i.e. things Fix-A alone does **not** cover:

1. **Scale key naming → `weight_scale_inv`.** `_map_weight_name_for_vllm` renames a
   trailing `.scale` to `.weight_scale_inv` (and `.shared_experts.w2`→`.down_proj`,
   `layers.`/`embed.`→`model.` prefix, `head.weight`→`lm_head.weight`). vLLM 0.25's DS4
   fp8 linear params are `weight` / `weight_scale_inv` / `weight_scale`
   (`_prepare_linear_params_for_loading` binds exactly those three names via
   ModelWeightParameter / BlockQuantScaleParameter).
   → **Our `resync.py::_scale_name` emits `".scale"` (line 18), not `weight_scale_inv`.**
   Unless our verl_mlite receiver renames it, our block scales would not bind to the vLLM
   fp8 params. This is the single highest-value supplement Fix-A misses.
2. **Dense (non-expert) fp8 block-scale cache + TP-sharded re-apply.**
   `cache_deepseek_v4_dense_fp8_scales` / `reload_deepseek_v4_dense_fp8_scales` /
   `_copy_scale_shard` capture dense e8m0 scales and re-apply them after the standard load
   (which otherwise drops them), sharding along input/output dim by `tp_rank`/`tp_size`.
   Relevant because rollout TP=8 shards dense fp8 scales; if our receiver relies on vLLM's
   default load it may lose them.
3. **MoE w13/w2 param restore with fused shapes** (`_restore_moe_params_for_loading`):
   w13 = (E, 2*inter, hidden//2), w2 = (E, hidden, inter//2), scales along //32, as uint8
   params. Confirms the expert-weight fused layout vLLM 0.25 expects.
4. **Expert fp8 + e8m0-scale → uint8 transport** (`iter_deepseek_v4_weights`) — the
   alignment-safe way to move already-fp8 bytes if we ever ship them.
5. **No IPC-offset block padding anywhere** — re-confirmed. Upstream's answer to the
   alignment problem is "don't put fp8 in the bucket", not "align the offset".

## 3. Recommendation (decision surface — architecture call is bayan's)

- **P1 — keep sender-side quant, apply Fix-A + scale-key supplement (recommended, matches
  bayan 01:05 "apply on our compatible stack, don't wholesale switch").**
  - (a) Fix-A: pad the IPC bucket `offset` to 8 bytes (`offset = (offset + 7) & ~7`) at the
    sender, receiver reads the padded offset unchanged. Locus is verl
    `bucketed_weight_transfer.py` (cluster/site verl, upstream write-scope) or, to stay in
    our repo, Fix-C monkeypatch in `experimental/lite/examples/verl/verl_mlite/`.
  - (b) Scale-key supplement: ensure the block scale reaches vLLM as `weight_scale_inv`
    (either emit it from `resync.py::_scale_name`, or rename in the verl_mlite receiver).
    **Must be confirmed against the actual verl_mlite receiver before changing `resync.py`
    blindly** — the currently-green BF16 proxy path must not regress.
  - (c) Verify dense fp8 scales survive TP=8 load (item 2) — receiver-side inspection.
  - Verification is a GPU job (live vLLM receiver); gated on bayan (D2 + GPU rule).
- **P2 — adopt upstream architecture (ship bf16, quantize on receiver).** Structurally
  immune to the alignment bug, but a large change to our resync + pulls in verl's
  `load_quanted_weights` fp8 receiver machinery ⇒ close to the "wholesale switch" bayan
  ruled out on compatibility grounds. Recorded as the explanation for "why upstream never
  crashes" and a possible future simplification, not the near-term path.

Bottom line: Fix-A is correct and sufficient for the *mechanical* alignment crash on our
architecture; the *additional* thing #6473 reveals is the **`weight_scale_inv` scale-key
contract** (and dense-scale TP re-apply), which is a numerical-correctness concern
orthogonal to alignment and must be verified against our receiver, not patched blind.

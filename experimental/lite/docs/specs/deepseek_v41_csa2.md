# DeepSeek-V4.1 CSA2 operator migration contract

This specification compares the existing MLite DS4 CSA with the released V4.1
inference operators. It does not change DS4 production behavior or claim an
implemented CSA2 training path. Decisions apply to a future V4.1 implementation.

## Evidence and scope

Baseline: MLite commit `26e9bf64faf06be04686f199f8eecbfd07861a9c`,
`experimental/lite/megatron/lite/primitive/modules/attention/csa.py` (D below).
Official reference (O): [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py)
at revision `df42c109f1defefcbfcedbe7d905718a12266e40`, SHA-256
`4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
The matching config SHA-256 is
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.
Pinned local bytes were inspected; the web fetch failed, so no assertion of
current remote-main freshness is made. Source symbols are authoritative when
line numbers in earlier briefs refer to a different baseline.

Reuse the companion owner/consumer contract `deepseek_v41_owner_consumer.md`,
weight mapping `contracts/deepseek_v41/{weights,config}.json`, and optimizer
contract `specs/deepseek_v41_optimizer.md` when those independent changes land.
Do not recreate their complete layer/key/optimizer manifests here.
The latest owner contract correctly states that official inference has no CED
implementation: H20 and paired pre_mix are a proposed training boundary.
In particular O:1267 does not imply layer-20 compressor consumes raw H20:
Block's attention pre-mix and normalization still intervene.

## Operator decisions

**Keep DS4** means retain an implementation facility with the stated contract;
**Official** means change V4.1 semantics; **Equivalent** means the displayed
algebra agrees, not that different kernels/dtypes are bitwise identical.

| Operator / boundary | Existing DS4 | Released V4.1 | Decision and evidence |
|---|---|---|---|
| Layout adapter | SBH→BSH, output transpose in DeepseekV4CSAAttention | BSH input | **Keep DS4** layout adapter; a transpose followed by its inverse preserves token/head identity. Packed sequence boundaries must remain explicit. |
| Q low rank | q_norm(wq_a(x)), then wq_b | Same sequence, O:770–772 | **Equivalent** matrix and RMSNorm composition with matched weights/epsilon; no extra gain or head reduction. |
| Q head RMS | Additional q*rsqrt(mean(q²)+eps), D:416–418 and 770–772 | No normalization between wq_b and RoPE, O:771–772 | **Official** remove this operation for V4.1. The earlier `_per_head_rms` name denotes this semantic operation; current baseline has inline expressions. Probability counterexample below disproves equivalence. |
| Local KV | kv_norm(wkv(x)), shared latent head | Same, O:702–706 | **Equivalent** unquantized projection/norm algebra; keep one KV head, not one per Q head. Quantization differs below. |
| RoPE branch | theta and YaRN selected by ratio>1, D:402–410, 752–764; boundary/index helpers also use >1 | Any nonzero ratio enables compressed theta and YaRN, O:679–697 | **Official** ratio!=0 everywhere. ratio=1 is global unpooled KV, not pure SWA. |
| RoPE pair layout | Adjacent pairs in trailing 64 features; cos/sin duplication is just storage, D:122–139 | Complex adjacent pairs in trailing 64, O:391–405 | **Equivalent** (a+ib)(c+is)=(ac-bs)+i(as+bc). CPU extracted-function comparison below. |
| RoPE scaling | Linear blend of interpolated/extrapolated frequencies | Same correction floor/ceil and ramp, O:369–388 | **Equivalent** for factor=16, original length=65536, beta_fast=32, beta_slow=1; no added amplitude/mscale correction. Branch choice remains a required change. |
| Compressor allocation | Every ratio>1 layer; none for ratio=1 | Only KV sources 2,8,14,20, O:653–660 | **Official** Full owns KV; Reindex reads KV and computes new selection; Reuse reads both. Reuse the companion 40-layer assignment. |
| Compression ratio 2 | Generic gate pooling but includes learned APE | FP32 wkv/wgate, per-feature softmax over each two-token group, then cast and RMSNorm, O:438–487 | **Official** preserve exact dtype/order and remove APE; do not assume DS4 BF16 linear plus FP32 softmax matches FP32 projections. |
| Compression ratio 1 | No compressor | BF16 wkv then RMSNorm, no gate, O:452–470 | **Official** instantiate this at owner 20; no pooling/APE/state for partial groups. |
| Overlapping source entries | ratio=4 uses coff=2 and previous/current group overlap, D:149–183, THD overlap helper | Nonoverlapping ratio-token groups, O:478–487 | **Official** remove overlapping source entries and doubled projection dimensions from CSA2. This is not permission to deduplicate local-window vs global attention entries. |
| Absolute position embedding | Learned compressor.ape added to gate, D:154–180 | No compressor APE | **Official** remove APE in main and index paths; no invented checkpoint keys. |
| Compressed positions and visibility | Group start positions; completed-group causal limits | Position j*ratio, visible count floor((query_position+1)/ratio), O:547–551, 576–581, 750–757 | **Equivalent** for nonoverlapping groups with sequence-local positions. At ratio=2, query 0 sees zero groups, query 1 sees group 0. Trailing incomplete group stays invisible. |
| Index K origin | Independent indexer.compressor(x), D:315–317, 585–590 | k_norm(wk(main pre-RoPE compressed latent)); only KV owners own wk/k_norm, O:531–555 | **Official** remove independent compression; derive K before main latent is overwritten by RoPE/quantization. Reindex consumes shared index K, never recompresses hidden state. |
| Index Q transform | wq_b(q_low), RoPE, rotate_activation, D:605–610 | wq_b(qr), RoPE, FP4 quantization, O:558–560 | **Official** remove DS4 rotation from released path. Orthogonal rotation could preserve exact dot products if applied to both sides, but does not prove equivalence through quantization. |
| Index score | Per-head ReLU dot score, weighted head sum via fused backend | sum_h ReLU(q_h·k) weights_h, weights_proj(x)*128^-1/2*32^-1/2; TP sum, O:563–568 | **Official** enforce complete scaling once across caller/kernel. Shared formula alone does not certify the existing fused ABI. |
| Two-level candidates | No CSA2 candidate publisher/consumer state | Layer 20 publishes block mask; later index sources restrict scores, O:583–589, 598–623 | **Official** block size 8, top 2048 blocks by max score; pin newest reachable block, drop -inf blocks; then top 512 positions. Empty prefix must publish empty selection. |
| Top-K sharing | Recomputed layer-local selection or ratio-specific deterministic path | Sorted positional indices, -1 for unreachable, global offset; Reuse reads shared topk, O:591–594, 721–737 | **Official** owner-scoped publication. Candidate mask and integer Top-K are discrete, not differentiable tensors; indexer training requires its separately specified auxiliary objective. |
| SWA and global concatenation | Local causal window plus compressed entries | Same conceptual concatenation, O:774–778 | **Equivalent** only with identical selected entries, offsets and masks. Local K remains layer-specific, including Reuse layers. Window=128. |
| Attention logits/sink | Sparse kernel with head_dim^-1/2 and sinks | sparse_attn(q,kv,attn_sink,indices,512^-1/2), O:780 | **Keep DS4** sparse-kernel abstraction, but require ABI/mask/sink/quantized-input validation. No claim existing backend accepts CSA2 unchanged. |
| Cache quantization | Existing DS4 training/backend precision policy | SWA post-RoPE entire KV FP8; compressed post-RoPE KV FP4 block16/E4M3 scales; index Q/K FP4 block32/E8M0, O:705–706, 554,560,758–760 | **Official** distinguish all three formats; matching BF16 algebra cannot certify quantized forward. Training backward policy remains an implementation obligation. |
| Inverse output RoPE | _project_context uses cos,-sin before grouping, D:460–468 | apply_rotary_emb(o, freqs, True), O:781 | **Equivalent** conjugation equals sine negation. Keep inverse before projection; moving it across arbitrary weights is invalid. |
| Grouped output projection | GroupedLinear einsum ...gd,god→...go, then wo_b | bsgd,grd→bsgr, then flatten and wo_b, O:785–789 | **Equivalent** y[b,s,g,r]=sum_d o[b,s,g,d]W[g,r,d]. Eight groups of 8 heads×512 features, each to 1024; concatenate to 8192 then project to 5120. No across-group sum. |
| State/backward/CP | Packed THD, distributed sparse attention and indexer loss machinery | Global mutable inference caches; inference_mode | **Keep DS4** training/parallel framework, but replace owner-state transport explicitly. Floating KV consumer gradients sum at owner; integer selection has no ordinary derivative. Inference cache assignment is not proof of training/autograd equivalence. |

The owner sets are KV={2,8,14,20}, index={2,8,14,20,24,28,32,36}.
Decoder layers 20–39 have ratio=1, layers 2–19 ratio=2, layers 0–1 ratio=0.
These values select behavior; simply accepting the config does not implement it.
DSpark forward/rollout remains out of scope; preserve MTP weights per the companion
carrier contract, without mapping them to additional backbone layers.

## Reproducible CPU numerical evidence

Run with Python and CPU PyTorch, using the pinned released source above:

```sh
python experimental/lite/tools/validate_deepseek_v41_csa2.py \
  --official-model /path/to/inference/model.py
git diff --check
```

The probe AST-extracts actual MLite rotary helpers and official functions, omitting
GPU imports and decorators. It checks the official source digest before execution.
It tests FP32 vectors [1,8,2,512], including a 64-feature rotary tail, at positions
0,1,127,128,65535,65536,131071,1048575. The corrected caller selects theta=160000
and YaRN for ratios 1 and 2, theta=10000 without YaRN for ratio 0. The production
caller is still unchanged: this is a candidate semantic check, not integration.

Measured with CPU PyTorch 2.12.1: corrected output vs official maximum absolute
error **2.384185791015625e-7** for each ratio, using atol=rtol=1e-6. Inverse rotation
also reconstructs the input within those bounds. DS4 ratio=1 negative control
errors by position are 0, 0.23721886, 4.32865286, 5.39011955, 5.13199329,
4.43960762, 6.29173470, 4.02702570. Ratios 0 and 2 agree within 2.39e-7.

For q=[2,0], K=identity(2), scale=1/sqrt(2), the extra per-head RMS changes
softmax probabilities by **0.0733711123**. This is a counterexample, not a model
quality metric. The two grouped einsums agree exactly on a float64 reduced-size
fixture with eight groups; a head-group permutation negative control fails.
The index identity above supplies the dimension-independent proof.

These checks passed with exit code 0 and marker `CSA2_CPU_SEMANTIC_PROBES_OK`.
They do not exercise fused GPU kernels, cache decoding, two-level Top-K,
quantization, CP, full-model numerical parity, or backward. Those require later
implementation-stage tests against an independent reference. In particular,
no training pass or FP32-gradient claim follows from inference source analysis.

# Incremental RL weight synchronization

## Decision summary

MLite should prototype **lossless, versioned incremental synchronization**, but
should not replace the current full-weight path yet. The current path already
matches the Megatron reference: the validated Qwen3.5 harness measured 11.837 s
for MLite versus 11.903 s for mbridge. Incremental synchronization therefore has
to beat a seconds-scale, GPU-resident baseline; the obsolete 100 s measurement
is not a valid benefit baseline.

The first implementation candidate is adaptive per export bucket:

1. send only a version manifest when the target bytes are unchanged;
2. send the current dense bucket when compression is not profitable;
3. otherwise send either an exact bitmap plus replacement values for the local
   GPU transport, or a bytewise XOR compressed with zstd for object storage;
4. always retain full synchronization as recovery and compatibility fallback.

Both incremental codecs reconstruct the exact target representation. Top-k,
low-rank, and quantized deltas are intentionally deferred because they change
the rollout policy and require a separate numerical approval. The empirical
decision starts from exported BF16 weights rather than FP32 optimizer master
weights, then separately serializes the same checkpoint pairs into the target
block-FP8 representation used for inference.

<!-- RESULT_VERDICT_BEGIN -->
The evidence supports an opt-in lossless prototype, not a production-path
replacement. Two sampled RL adjacent-step windows were bitwise unchanged in
their analyzed rollout target: one after a nonzero-gradient DAPO warm-up update
and one after a zero-advantage GRPO batch. A nonzero SFT adjacent-step control
changed 49.461% of BF16 values, so this does not establish a universal per-step
sparsity rule. The Qwen3.5 ten-update DAPO windows changed 9.755% and 10.881%;
the later window was denser, and neither reproduces Cognition's reported
over-99% reduction.

For the same full Qwen3.5 10 -> 20 delta, canonical block-FP8 serialization
reduced changed-byte density from 9.755% in BF16 to 1.294% of the mixed FP8
target. Manifest-only unchanged tensors plus bitmap/replacement values cost
13.132% of the FP8 target, or 6.845% of the original BF16 bytes. Applied to the
validated 69.321 GB transfer, the payload lower bound falls from 11.04 GB for
BF16 to 4.75 GB for FP8. Changes are broad across layers: 31 of 40 layers are
needed to cover 80%, and the top four contribute only 11.19%. The next decision
should therefore authorize only the adaptive lossless encode/apply prototype
and its bucket-level wall-time benchmark. Production integration and lossy
top-k or quantized deltas remain gated.
<!-- RESULT_VERDICT_END -->

## Empirical protocol

The CPU-only analyzer is
[`analyze_checkpoint_delta.py`](../examples/verl/scripts/analyze_checkpoint_delta.py).
It streams one safetensor at a time and reports, for every tensor:

- delta L-infinity, L2, and relative L2 norms;
- exact changed-element density plus densities above `1e-8` through `1e-3`;
- an absolute-delta magnitude histogram;
- exact bitmap-plus-replacement and 32-bit COO payload estimates;
- nonzero bytes in the bitwise XOR; and
- zstd level-3 bytes for independent per-tensor XOR frames;
- actual E4M3 weight plus FP32 block-scale target bytes; and
- family, shallow/middle/deep, per-layer, and layer-concentration aggregates.

The zstd total is conservative relative to a production bucket stream because
each tensor starts a separate frame. Bitmap and COO estimates exclude names,
shapes, version manifests, checksums, and kernel launch cost, so they are payload
lower bounds rather than wall-time predictions.

Two evidence sets are kept separate:

- **True one-step window.** Qwen3-8B-Base DAPO calibration run, BF16, learning
  rate configured to peak at `1e-6`, base to `global_step_1`. The scheduler's
  logged step-1 actor rate was `1e-7` during its ten-step warm-up. The run logged
  a nonzero gradient norm and completed `update_actor` before saving. The
  four-rank FP32 FSDP checkpoint was merged to HF BF16 on a CPU Slurm partition
  before comparison.
- **Training-stage windows.** Qwen3.5-35B-A3B, DAPO, BF16 exported HF weights,
  learning rate `1e-6`, adjacent stored checkpoints `10 -> 20` and `20 -> 30`
  from one Megatron/mbridge run. Each window contains ten optimizer updates; it
  shows early-to-late behavior but must not be presented as a single-step
  density.

The Qwen3.5 run used a controlled DAPO pipeline with DAPO-Math-17k, AIME-2024,
train batch 32, generation batch 96, eight responses per prompt, token-mean
loss, and TP1/PP1/CP1/EP8. The run's Megatron leg was selected because the
historical MLite leg only retained step 10 and therefore cannot form an adjacent
checkpoint pair.

### Aggregate results

<!-- RESULT_TABLE_BEGIN -->
| Window | Updates | Dense GiB | Exact changed | `>1e-5` | L-infinity | L2 | Relative L2 | XOR nonzero bytes | XOR+zstd / dense | Unchanged+bitmap / dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B base -> 1 | 1 | 15.256 | 0% | 0% | 0 | 0 | 0 | 0% | n/a | manifest only |
| Qwen3.5 10 -> 20 | 10 | 65.392 | 9.7550% | 2.3056% | 6.104e-5 | 0.55882 | 2.277e-4 | 4.9441% | 9.8623% | 15.9255% |
| Qwen3.5 20 -> 30 | 10 | 65.392 | 10.8811% | 3.3970% | 6.104e-5 | 0.74595 | 3.039e-4 | 5.5088% | 10.6415% | 17.0516% |

The one-step result is not an empty training event: the run logged actor
gradient norm 1.286 and 25.98 s in `update_actor` before saving. It shows why
delta detection must use the exported target dtype. The optimizer's FP32 state
can move while the rollout-visible BF16 bits remain unchanged. Zstandard was
not available in that analysis environment, but an unchanged manifest is
strictly smaller than compressing an all-zero XOR payload.

The final column sums a zero-byte payload for unchanged tensors and a bitmap
plus BF16 target values for changed tensors. It excludes manifests and kernels.
Choosing the byte-minimum of bitmap, COO, and dense per tensor would improve
the two Qwen3.5 lower bounds only slightly, to 15.785% and 16.918%.
<!-- RESULT_TABLE_END -->

For BF16 with `N` elements and `k` exact changes, the payload-only ratios are:

- bitmap plus target values: `ceil(N / 8) + 2k` bytes, approximately
  `6.25% + k/N` of dense BF16;
- 32-bit index plus target value: `6k` bytes, approximately `3k/N` of dense;
- XOR plus zstd: measured directly, because its ratio cannot be inferred from
  element density alone.

Thus COO only beats dense BF16 below 33.3% density, a bitmap can still be
smaller below 93.75%, and COO beats a bitmap below 3.125%. These are byte
crossovers, not performance gates: GPU scatter cost can make a theoretically
smaller payload slower.

### Parameter-family results

Family classification exists only in the analysis layer. It is not a runtime
dispatch mechanism and must not leak model names into a primitive.

<!-- FAMILY_TABLE_BEGIN -->
| Window | Family | Dense GiB | Exact changed | `>1e-5` | L-infinity | Relative L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base -> 1 | embedding + head | 2.318 | 0% | 0% | 0 | 0 |
| base -> 1 | attention | 2.813 | 0% | 0% | 0 | 0 |
| base -> 1 | router | n/a | n/a | n/a | n/a | n/a |
| base -> 1 | expert | n/a | n/a | n/a | n/a | n/a |
| 10 -> 20 | embedding | 0.947 | 0.6814% | 0.1122% | 3.052e-5 | 5.404e-5 |
| 10 -> 20 | head | 0.947 | 0.7172% | 0.1367% | 6.104e-5 | 3.950e-5 |
| 10 -> 20 | attention | 2.392 | 10.4236% | 3.8375% | 6.104e-5 | 2.139e-4 |
| 10 -> 20 | router | 0.039 | 14.3340% | 5.1497% | 6.104e-5 | 2.574e-4 |
| 10 -> 20 | expert | 60.234 | 10.1450% | 2.3434% | 6.104e-5 | 2.778e-4 |
| 10 -> 20 | norm | <0.001 | 0.5794% | 0.2408% | 3.052e-5 | 2.496e-6 |
| 10 -> 20 | vision | 0.832 | 0% | 0% | 0 | 0 |
| 20 -> 30 | embedding | 0.947 | 0.8297% | 0.2162% | 6.104e-5 | 7.803e-5 |
| 20 -> 30 | head | 0.947 | 0.8261% | 0.2361% | 6.104e-5 | 5.593e-5 |
| 20 -> 30 | attention | 2.392 | 11.3890% | 4.8163% | 6.104e-5 | 2.813e-4 |
| 20 -> 30 | router | 0.039 | 15.2466% | 6.1305% | 6.104e-5 | 3.262e-4 |
| 20 -> 30 | expert | 60.234 | 11.3246% | 3.4855% | 6.104e-5 | 3.712e-4 |
| 20 -> 30 | norm | <0.001 | 0.6360% | 0.2945% | 6.104e-5 | 3.283e-6 |
| 20 -> 30 | vision | 0.832 | 0% | 0% | 0 | 0 |

Router tensors have the highest density, but they are only 0.039 GiB. Experts
hold 92.1% of the analyzed bytes and therefore dominate payload. Every
trainable family becomes denser in the later window; the frozen vision family
remains unchanged. Head, norm, and visual weights are kept distinct rather than
being folded into embedding, attention, or dense MLP by substring matching.
<!-- FAMILY_TABLE_END -->

<!-- HISTOGRAM_BEGIN -->
All 399 tensors in the true one-step window are in the zero-density bin. The
Qwen3.5 per-tensor exact-density distribution is:

| Per-tensor exact density | 10 -> 20 tensors | 10 -> 20 dense-byte share | 20 -> 30 tensors | 20 -> 30 dense-byte share |
| --- | ---: | ---: | ---: | ---: |
| 0 | 449 | 1.2721% | 449 | 1.2721% |
| (0, 1%] | 62 | 2.8975% | 57 | 2.8975% |
| (1%, 5%] | 16 | 0.0003% | 20 | 0.0001% |
| (5%, 10%] | 70 | 31.1468% | 39 | 14.4661% |
| (10%, 25%] | 362 | 64.6776% | 393 | 81.3555% |
| (25%, 50%] | 61 | 0.0057% | 66 | 0.0086% |
| (50%, 100%] | 6 | <0.0001% | 2 | <0.0001% |

Among elements that changed exactly, the absolute BF16 delta distribution is:

| Absolute delta | 10 -> 20 changed-element share | 20 -> 30 changed-element share |
| --- | ---: | ---: |
| (0, `1e-7`] | 0.0218% | 0.0195% |
| (`1e-7`, `1e-6`] | 3.7625% | 3.3822% |
| (`1e-6`, `1e-5`] | 72.5806% | 65.3789% |
| (`1e-5`, `1e-4`] | 23.6352% | 31.2194% |
| `>1e-4` | 0% | 0% |

The largest per-tensor relative L2 values belong to 4 KiB shared-expert gate
tensors: 2.015e-3 at 10 -> 20 and 2.336e-3 at 20 -> 30. Reporting their sizes
prevents an unweighted top-tensor list from obscuring the byte-dominant expert
weights.
<!-- HISTOGRAM_END -->

### Adjacent-step and block-FP8 follow-up

The follow-up keeps the checkpoint representation explicit. For eligible
attention, dense-MLP, and expert matrices, the analyzer materializes the
**actual serialized block-FP8 target**: E4M3 weight bytes and one FP32
dequantization scale per 128-by-128 block, matching the expansion-and-slice
layout consumed by MLite's
[`*_scale_inv` checkpoint loader](../megatron/lite/model/kimi_k2/lite/checkpoint.py).
Embeddings, routers, norms, heads,
vision tensors, and native specialized GDN tensors remain BF16 passthrough
values in the configured local-DCP projection. Canonical exported
linear-attention names follow the rollout attention path and are quantized.
The reported FP8 changed density is changed serialized bytes divided by all
bytes in that mixed target; it is not a BF16 proxy. Separate bitmaps cover
weight bytes and FP32 scales.
The dequantization scale is `max(abs(block), 1e-4) / 448`; partial edge blocks
are zero-padded for the reduction and the E4M3 weight is cropped back to its
original shape. Packed MoE expert groups keep their leading expert dimension
and quantize each expert over the final two matrix dimensions. Three-dimensional
linear-attention convolution weights remain BF16, matching the loader's
matrix-only rule.

| Checkpoint window | Training event | Updates | Analyzed BF16 GiB | BF16 exact changed | BF16 `>1e-5` | L-infinity | Relative L2 | Block-FP8 changed bytes | Block-FP8 bitmap+values |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B base -> 1 | DAPO, nonzero gradient, warm-up LR `1e-7` | 1 | 15.256 | 0% | 0% | 0 | 0 | 0% | manifest only |
| Qwen3-30B-A3B 1 -> 2 | GRPO, zero-advantage batch | 1 | 8.197 | 0% | 0% | 0 | 0 | 0% | manifest only |
| Qwen3.5-35B-A3B 1 -> 2 | SFT resume smoke, LR `1e-5` | 1 | 9.800 | 49.4609% | 47.0414% | 4.578e-5 | 1.375e-3 | 18.0959% | 28.5390% |
| Qwen3.5-35B-A3B 10 -> 20 | DAPO, LR `1e-6` | 10 | 65.392 | 9.7550% | 2.3056% | 6.104e-5 | 2.277e-4 | 1.2940% | 13.1319% |

The GRPO checkpoint is an honest unchanged fast-path observation, not evidence
of a sparse nonzero optimizer step: its reward, advantage, loss, and gradient
norm were all zero. The SFT pair supplies the nonzero adjacent-step control. Its
local-DCP projection is a quantization control rather than a whole-model wire
estimate. Its quantized matrices changed in 8.0469% of E4M3 weight bytes and
in none of their FP32 scales; including BF16 passthrough tensors gives the
18.0959% mixed-target figure above. Its bitmap-plus-values payload is 17.0809%
of the original BF16 bytes because the complete mixed FP8 target is itself only
59.8511% as large.

The full exported DAPO row is the rollout-format result. Its complete mixed FP8
target is 34.088 GiB, 52.1279% of the 65.392 GiB BF16 export. E4M3 bytes changed
at 1.3467% across the 390 quantized objects; all 2,052,800 FP32 block scales were
bitwise unchanged. BF16 passthrough tensors changed in 0.7009% of their bytes,
giving the 1.2940% mixed-target density. Sending no bitmap for 449 unchanged
tensors reduces bitmap-plus-values from 13.2844% to the 13.1319% shown above.

Thus the sampled RL single-step windows are much sparser than the stored
ten-update DAPO windows, but the high-learning-rate SFT counterexample is not.
The evidence does not establish a universal "one step is sparse" rule across
algorithms, learning rates, or batches. Codec choice must remain per bucket and
measured on the final target bytes.

#### Layer and depth distribution

The tables below use global layer indices and byte-weighted aggregates; the
global row contains embedding, head, and other non-layer tensors. They are
analysis categories, not a layer-selection policy.

| Window | Depth | Dense GiB | BF16 exact changed | BF16 `>1e-5` | Relative L2 | Block-FP8 changed bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SFT 1 -> 2 | shallow | 3.101 | 52.4760% | 49.7362% | 1.552e-3 | 15.5150% |
| SFT 1 -> 2 | middle | 2.879 | 50.2313% | 47.4111% | 1.455e-3 | 12.9008% |
| SFT 1 -> 2 | deep | 2.873 | 47.8465% | 45.3769% | 1.314e-3 | 11.7940% |
| SFT 1 -> 2 | global | 0.947 | 42.1436% | 42.1426% | 9.726e-4 | 42.1436% |
| DAPO 10 -> 20 | shallow | 21.939 | 10.8207% | 2.4667% | 2.890e-4 | 1.4536% |
| DAPO 10 -> 20 | middle | 20.369 | 10.4755% | 2.5422% | 2.822e-4 | 1.4015% |
| DAPO 10 -> 20 | deep | 20.357 | 9.1269% | 2.1924% | 2.457e-4 | 1.2306% |
| DAPO 10 -> 20 | global | 2.726 | 0.4859% | 0.0865% | 1.698e-5 | 0.4859% |

The adjacent SFT update weakens gradually with depth, but it is broad rather
than concentrated. Reaching 80% of transformer-layer exact changes requires 32
of 40 layers; the top four layers, including the unusually dense first layer,
contribute only 11.29%. A selective "send a few layers" policy would therefore
miss most of this update.

The full exported DAPO 10 -> 20 window has the same broad shape in BF16:
shallow, middle, and deep exact densities are 10.8207%, 10.4755%, and 9.1269%,
with relative L2 values 2.890e-4, 2.822e-4, and 2.457e-4. Reaching 80% of its
transformer-layer changes requires 31 of 40 layers, while the top four account
for only 11.19%. Repeating concentration on serialized FP8 changed bytes gives
the same 31 of 40 layers and an 11.18% top-four share. Both representations
therefore reject the few-layer-dominance hypothesis for these samples.

| Window | Parameter family | Dense GiB | BF16 exact changed | BF16 `>1e-5` | Relative L2 | Block-FP8 changed bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SFT 1 -> 2 | embedding | 0.474 | 49.5913% | 49.5911% | 1.374e-3 | 49.5913% |
| SFT 1 -> 2 | attention/QKV | 0.254 | 36.4635% | 30.7994% | 9.976e-4 | 5.3416% |
| SFT 1 -> 2 | GDN | 0.942 | 34.6318% | 29.3289% | 8.758e-4 | 34.6318% |
| SFT 1 -> 2 | router | 0.039 | 48.9394% | 43.3075% | 1.034e-3 | 48.9394% |
| SFT 1 -> 2 | expert MLP | 7.617 | 52.6423% | 50.4034% | 1.638e-3 | 8.1350% |
| SFT 1 -> 2 | norm | <0.001 | 0.8369% | 0.7155% | 5.671e-6 | 0.8369% |
| SFT 1 -> 2 | head | 0.474 | 34.6962% | 34.6944% | 7.693e-4 | 34.6962% |
| DAPO 10 -> 20 | embedding | 0.947 | 0.6814% | 0.1122% | 5.404e-5 | 0.6814% |
| DAPO 10 -> 20 | attention/linear-attention | 2.392 | 10.4236% | 3.8375% | 2.139e-4 | 1.4235% |
| DAPO 10 -> 20 | router | 0.039 | 14.3340% | 5.1497% | 2.574e-4 | 14.3340% |
| DAPO 10 -> 20 | expert MLP | 60.234 | 10.1450% | 2.3434% | 2.778e-4 | 1.3450% |
| DAPO 10 -> 20 | norm | <0.001 | 0.5794% | 0.2408% | 2.496e-6 | 0.5794% |
| DAPO 10 -> 20 | head | 0.947 | 0.7172% | 0.1367% | 3.950e-5 | 0.7172% |
| DAPO 10 -> 20 | vision | 0.832 | 0% | 0% | 0 | 0% |

Expert MLP weights dominate the analyzed shard by size and have its largest
relative L2. FP8 removes most of their small BF16 changes, reducing their
changed-byte density from 52.64% to 8.14%. Router density is high but its
payload is tiny. The GDN, router, norm, embedding, and head rows are BF16
passthrough in this target contract, so their two density columns match by
construction.

The full DAPO window is likewise family-dominated by size, not by a tiny layer
set. Expert MLP contains 60.234 GiB and 95.80% of all exact changed values, at
10.1450% density and relative L2 2.778e-4. Router is denser at 14.3340% but only
0.039 GiB. Attention/linear-attention is 2.392 GiB at 10.4236%; embedding and
head are each 0.947 GiB at 0.6814% and 0.7172%; norms are below 0.001 GiB at
0.5794%; and the frozen 0.832 GiB vision family is unchanged. Block-FP8 reduces
the byte-dominant expert and attention densities to 1.3450% and 1.4235%.
Router, embedding, norm, head, and vision retain their BF16 densities because
they are passthrough families in this contract. Expert MLP still contributes
91.86% of mixed-target FP8 changed bytes because of its size.

### Limitations

- The one-step result is dense Qwen3-8B; the stage result is Qwen3.5 MoE. One
  run per model is enough to reject a universal sparsity assumption, but not to
  establish a production p95 across seeds, recipes, and training phases.
- The retained every-step Qwen3.5 DAPO checkpoint metadata no longer has its
  distributed payload files. This zero-GPU study could not reconstruct that
  desired same-run adjacent pair. The GRPO and SFT adjacent results instead
  read complete local DCP model states; they are representative shard
  distributions, not full canonical exports, and are not used to extrapolate
  whole-model bytes.
- One older SFT resume pair was deliberately rejected: 99.897% exact changes
  and relative L2 near 1.414 indicated reset or mismatched state rather than a
  continuous optimizer step. A successful checkpoint read is not sufficient
  evidence of a valid adjacent update.
- The stage windows accumulate ten updates. Cancellation within a window can
  make cumulative density smaller than the union of per-step changes, while
  repeated small updates can make it larger after BF16 rounding.
- The block-FP8 result uses deterministic CPU serialization of the 128-by-128
  E4M3-weight plus FP32-scale contract. Production enablement still requires
  bitwise comparison against the rollout engine's canonical post-quantization
  artifact and must fall back if its quantizer or eligible-module set differs.
- Payload reduction does not remove all of the current 11.837 s. Export gather,
  mapping, version checks, encode, decode, and apply remain.

## Reference systems and what transfers

[Cognition's SWE-1.7 report](https://cognition.com/blog/swe-1-7) describes a
single trainer feeding rollout clusters on three continents through object
storage. The public architecture sends a compressed delta every K gradient
steps, shows an XOR-plus-zstd data path, reports over 99% transfer reduction,
publishes version manifests, prefetches into CPU memory, and recovers a new
worker from a checkpoint followed by a delta chain. It does **not** publish the
bucket format, crossover policy, checksum protocol, or how that compression
ratio varies by model and step. Its headline ratio is therefore a motivation,
not an MLite estimate.

The same report also describes Muon and replaying the inference-time sampling
distribution from recorded token masks. Those address optimizer behavior and
train/rollout policy consistency; they are orthogonal to the lossless transport
codec proposed here and are not silently imported into this design.

Relevant compression work has different correctness contracts:

- [Deep Gradient Compression](https://research.google/pubs/deep-gradient-compression-reducing-the-communication-bandwidth-for-distributed-training/)
  applies top-k-style sparsification to gradients and needs momentum correction,
  local clipping, momentum masking, and warm-up to preserve convergence.
- [QSGD](https://papers.nips.cc/paper/6768-qsgd-communication-efficient-sgd-via-gradient-quantization-and-encoding)
  quantizes gradients and trades communication for variance and convergence
  time.
- [Error Feedback Fixes SignSGD](https://proceedings.mlr.press/v97/karimireddy19a.html)
  demonstrates why biased lossy compression needs residual error feedback.
- [PowerSGD](https://proceedings.neurips.cc/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html)
  is useful when matrix gradients are low-rank and compression plus all-reduce
  is faster than the optimized dense collective.
- [DeltaZip](https://www.research-collection.ethz.ch/handle/20.500.11850/731441)
  co-designs compressed fine-tuning deltas with multi-model serving. Its
  long-lived base-plus-variant setting is closer to model weights, but not to a
  latency-sensitive update after every RL optimizer step.

These methods justify candidates, not correctness. MLite rollout weights are
state, not an optimization message. Dropping or quantizing a weight update
immediately changes log probabilities and importance ratios. Lossy methods need
same-cache logprob/KL and short-RL evidence in addition to optimizer convergence
arguments.

[NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
require participating ranks to use matching counts and datatypes. Sparse data
must therefore be packed into explicit dense index/mask/value buffers before a
collective or point-to-point transfer; a Python sparse object is not a wire
format.

## Proposed MLite design

### Layering and ownership

The design follows the MLite primitive contract:

- A new generic checkpoint primitive owns delta metadata, lossless codecs,
  checksums, and apply semantics. It knows shapes, dtypes, byte layouts, and
  versions, but no model-family names and no veRL or Miles classes.
- `hf_weights.py` supplies final exported tensors and layout-aware gather hooks.
  Model protocols continue to own Qwen, Kimi, GLM, or DeepSeek mapping.
- Application adapters own transport: veRL CUDA IPC, Miles/SGLang distributed
  broadcast, or object storage.
- A codec registry is selected by measured bucket properties and receiver
  capability. Unsupported combinations fail closed to the existing dense path.

A possible primitive API is:

```python
manifest, payloads = encode_weight_update(
    current=bucket,
    previous=shadow,
    base_version=17,
    target_version=18,
    codecs=("unchanged", "bitmap_values", "xor_zstd", "dense"),
)

apply_weight_update(
    target=rollout_bucket,
    manifest=manifest,
    payloads=payloads,
    current_version=17,
)
```

The manifest includes tensor identity, canonical flat layout, shape, dtype,
base and target versions, codec, payload length, and digests of both encoded
payload and reconstructed target. A duplicate target version is idempotent; a
missing base, gap, stale update, unsupported codec, or digest mismatch requests
a full resync. The receiver publishes the new version only after all buckets
have applied successfully. A receiver that fails after a partial in-place apply
is quarantined until full resync; it must not serve a mixed model.

### Where delta detection runs

The comparison must use the **final target representation**:

1. optimizer step finishes;
2. model-specific mapping and export dtype are resolved;
3. current export bytes are compared with the last acknowledged shadow;
4. the selected payload is emitted and the shadow advances only after ACK.

The initial rollout cadence remains one synchronization per optimizer step.
Increasing Cognition's `K` above one would add policy staleness and is an RL
algorithm change, not a transparent transport optimization.

Comparing FP32 master parameters earlier is incorrect: BF16 rounding can turn a
nonzero master update into an unchanged rollout value, while mapping can split,
concatenate, reorder, or quantize a tensor.

The initial prototype should compare after the existing full gather. This saves
transport and apply bytes without changing gather semantics and gives a clean
performance attribution. A second phase may compare local FSDP/TP/EP shards
against local shadows before gather, exchange change masks, and gather only
selected blocks. That optimization is more valuable but substantially harder
because local coordinates must map to canonical exported coordinates.

The shadow is the main memory cost. It must be bounded and explicit:

- same-cluster prototype: keep the acknowledged canonical shadow on CPU or
  local storage and stage one previous exported bucket at a time; do not retain
  a second full model on every GPU;
- object storage: previous canonical checkpoint or delta base on CPU/local disk;
- offloaded training: evaluate reusing the existing CPU parameter state, but do
  not assume it equals the last acknowledged exported representation.

Reading a full CPU shadow and comparing every byte can erase the transport win;
it is an evidence prototype, not yet a production fast path. The later
pre-gather phase should record exact BF16 replacement masks while local shards
are updated, then map those coordinates through the existing exporter. An
alternative exact format is a changed-block bitmap plus complete replacement
blocks, which needs only previous block digests on the sender, but block
amplification must be measured before selection.

Today the Miles
[`MLiteWeightUpdater.update_weights`](../examples/miles/miles_mlite/weight_update.py)
increments a weight version, pauses and flushes rollout engines, iterates
bounded `_export_weight_chunks`, fans every dense chunk out to colocated and
distributed receivers, waits for their acknowledgements, and resumes serving.
The delta encoder belongs between final chunk export and those transport
adapters. Shadow and base-version state must be receiver-scoped: a new or stale
receiver needs a dense resync without forcing already-current receivers back to
the same base.

### Lossless wire formats

**Unchanged** emits only the authenticated manifest when old and target bytes
are identical. The receiver verifies the base and target digest, advances its
version, and acknowledges without touching weight storage. Advancing the
version is necessary even with an empty payload so that the next incremental
update names the same base on both sides.

**Bitmap plus replacement values** uses one bit per canonical element and packs
the new target values in index order. Applying replacement values, rather than
adding numeric deltas, prevents arithmetic drift and makes replay idempotent.
It is the preferred GPU candidate when exact density is low enough.

**XOR plus zstd** XORs the old and new serialized target bytes and compresses
the result. Applying the decompressed XOR reconstructs the target bits exactly.
It is attractive for object storage and CPU staging, but its CPU encode/decode
cost must be included in wall time. It should not be inserted into the current
11.837 s local path solely because the byte ratio is good.

**Dense** is always available. Codec selection is per bounded export bucket,
using actual encoded bytes and a configured safety margin rather than a global
model-level guess. Metadata, encode time, and apply time are included in the
decision benchmark.

### Interaction with proposed two-slot transport

The separate Miles/SGLang two-slot producer/consumer transport is not
implemented in the current tree. Today `RawHFWeightUpdater` iterates
`_export_weight_chunks` synchronously, accumulates receiver references and
colocated live tensors, and waits for acknowledgements only after every chunk
has been submitted. The incremental prototype must not assume that export and
transport already overlap.

The two proposals should compose without creating another queue. At most two
in-flight slots may exist. A slot owns its exported bucket, encoded payload,
receiver references, and pending shadow transition until every receiver ACKs.
Only then may the sender commit that receiver's new base version and reuse the
slot. A timeout or failed receiver leaves its shadow unadvanced and forces a
dense resync before that receiver serves the target version.

Until the bounded transport lands and is validated independently, the lossless
prototype should encode and apply buckets serially and report its peak live
memory without claiming overlap. The future colocated and distributed legs both
need chunk-order, weight-version, termination, and failure tests. The
acknowledged cross-step shadow remains separately accounted state; it is not one
of the two transient transport slots.

### BF16 to block-FP8 resync

BF16 deltas must never be applied directly to an FP8 rollout tensor. Block-FP8
quantization couples a weight block to its scale, so one BF16 change can alter
the scale and many serialized FP8 bytes.

The safe path is:

1. produce the same canonical block-FP8 target artifact as a full resync;
2. include weight bytes and scale bytes in one versioned block;
3. compare that serialized target with the last acknowledged FP8 artifact;
4. send exact changed blocks or lossless XOR; and
5. verify the reconstructed FP8 bytes before publishing the version.

This adds no error beyond the already validated BF16-to-FP8 quantization. If the
canonical FP8 serializer is unavailable, nondeterministic, or changes version,
the codec must fall back to a full FP8 resync. Exact BF16 density must not be
used to predict FP8 density. The CPU follow-up above now supplies the first
format-level byte measurement; engine-produced artifact equality remains the
integration gate.

### Full calibration and recovery

Lossless replacement and XOR do not accumulate numerical error, but full
snapshots remain necessary for recovery, shadow refresh, serializer upgrades,
and bounded replay time. Rather than hard-coding an arbitrary interval, write a
new full base when any of these is true:

- the delta chain reaches a configured maximum length;
- cumulative encoded bytes exceed a dense snapshot;
- a version or checksum mismatch occurs;
- codec, mapping, dtype, topology, or quantizer identity changes; or
- an operator requests calibration.

Lossy top-k or quantized deltas, if ever approved, additionally require residual
error feedback and a fixed full-calibration cadence. They are a separate
numerical feature, not an optimization flag on the lossless protocol.

## Benefit model and acceptance gates

For the current Qwen3.5 baseline, use 69.321 GB transferred and 11.837 s wall as
the dense reference. If an observed encoded payload ratio is `r`, the payload
estimate is `69.321 * r` GB. Wall time is **not** `11.837 * r`: full-gather,
mapping, encode, synchronization, receiver apply, and fixed handshakes remain.

| Evidence window | Unchanged+bitmap `r` | Estimated transfer | Byte reduction | Optimistic post-gather bound |
| --- | ---: | ---: | ---: | ---: |
| 10 -> 20 | 0.1593 | 11.04 GB | 84.1% | 4.90 s + new overhead |
| 20 -> 30 | 0.1705 | 11.82 GB | 82.9% | 5.00 s + new overhead |

The format-level result changes both the dense denominator and the delta
density for the same 10 -> 20 checkpoint pair:

| Target representation | Dense target / BF16 | Changed bytes / target | Manifest+bitmap / target | Payload / original BF16 | Estimated transfer | Optimistic post-gather bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 100% | 9.7550% | 15.9255% | 15.9255% | 11.04 GB | 4.90 s + new overhead |
| Block-FP8 mixed target | 52.1279% | 1.2940% | 13.1319% | 6.8454% | 4.75 GB | 4.16 s + new overhead |

The FP8 row is a payload lower bound for a new sender-side canonical quantize
and delta path, not an optimization available in today's BF16-send-then-
post-process path. It is 57.0% smaller than the BF16 incremental estimate, but
must pay quantization, detection, and mixed sparse-apply costs that are absent
from the current 11.837 s measurement.

The last column is deliberately optimistic:
`3.591 + r * (11.837 - 3.591)`. It preserves the measured 3.591 s full-gather
time, assumes every other second scales linearly with bytes, and charges no
detect, encode, decode, or sparse-apply cost. It is a lower bound under those
assumptions, not a latency forecast. Fixed handshakes and shadow reads make the
real post-gather prototype slower. Conversely, a future pre-gather change-mask
path can remove work that this bound preserves.

For object storage, the measured XOR-plus-zstd ratios map to 6.84 GB and 7.38 GB
on the same 69.321 GB reference. No object-store latency is projected because
the current seconds-scale same-cluster path is not a valid cross-cluster
bandwidth baseline.

The implementation benchmark must report at least:

- detect, gather, encode, transport, decode, apply, and total wall time;
- encoded bytes and peak live bytes per rank;
- dense fallback rate and codec choice per bucket;
- bitwise target equality for lossless codecs;
- receiver version/failure recovery; and
- same-cache logprob equality plus a short RL loss/reward check before rollout
  delivery.

Recommended staged decision:

1. **Land the analyzer and collect distribution evidence** across at least one
   dense and one MoE run at early, middle, and late stages.
2. **Prototype lossless encode/apply outside the production wire.** Proceed only
   if p95 encoded bytes are at most 25% of dense and encode plus apply cost leaves
   a credible end-to-end win against 11.837 s.
3. **Integrate adaptive unchanged/dense/bitmap transport** into one veRL path
   behind an opt-in flag; require bitwise fast-versus-full equality and bounded
   two-slot memory.
4. **Add object-storage XOR plus zstd** when cross-cluster synchronization is in
   scope; cap checkpoint-plus-delta replay length.
5. **Validate the measured block-FP8 contract against rollout output** before
   enabling DS4 incremental resync. Keep full resync as the only FP8 path until
   bitwise artifact equality and the byte/time gate both pass.
6. **Do not start top-k or quantized-delta implementation** without a separately
   approved numerical experiment.

The 25% byte gate is a proposal for the prototype, not a claimed crossover. It
leaves room for metadata and scatter/decompression overhead and should be
revised only from measured end-to-end data. The aggregate 15.9--17.1% evidence
and the 6.85%-of-BF16 full-model FP8 result clear an initial byte screen, but do
not establish bucket-level p95; the SFT local-shard FP8 control is 28.54% of its
target, and the prototype must measure real bucket and stage variation.

## Reproduction

```bash
python experimental/lite/examples/verl/scripts/analyze_checkpoint_delta.py \
  --before /path/to/previous/exported-hf \
  --after /path/to/current/exported-hf \
  --output /tmp/weight-delta.json \
  --fp8-block-shape 128,128 \
  --metadata model=my-model \
  --metadata window_steps=1
```

The command is CPU-only. It accepts immutable exported safetensor directories
or trusted PyTorch DCP model-state roots containing `.metadata`; DCP metadata is
pickle and must never come from an untrusted source. A distributed local-state
root measures that shard, not an implied full model. Do not compare optimizer
states, different model revisions, different export dtypes, or independently
quantized artifacts under one result label.

# V4.1 single-rank semantic implementation

This implementation consumes C-phase codecs and pinned-reference tooling.
The 2026-09-11 maintainer resolutions supersede the earlier open training
choices in the planning and training-specification snapshots. The supported
profile is post-training; DSpark and pretraining schedules are excluded.
D8 paired CED boundary work is included in D1/D2.

## Resolved execution contract

* Indexer parameters are frozen on construction and excluded by the trainable
  parameter filter. There is no indexer objective, loss attachment or autoscaler.
  Shared Q/compressor parameters outside the indexer remain trainable.
* MoE follows DS4's current post-training assembly (`compute_aux_loss=False`).
  A configured auxiliary coefficient alone does not attach an objective.
  Text/image selection uses separate persistent FP32 bias buffers. Bias affects
  membership only; weights use unbiased sqrt-softplus scores, temperature,
  selected-score normalization and route scaling.
* Modality counts follow DS4: int64 per-expert counts summed over TP, denominator
  local token count times TP size. Each modality retains its own counts, including
  an empty modality. Bias steps add `0.001 * sign(mean(counts) - counts)`; globally
  empty modality counts leave its bias unchanged. Forward/recompute never updates
  bias. The caller accumulates returned statistics and invokes `update_bias`
  according to the DS4 optimizer's existing accepted/skipped-step policy.
* QAT uses the approved identity STE with detached scales. FP4-off floating
  diagnostics are explicit. Global objectives follow DS4 valid-token weighting.
* `EngramTable(trainable=False)` holds only E4M3 rows and E8M0 scales; lookup
  decodes fetched rows only. The same `trainable` switch enables a persistent
  FP32 master and STE lookup. **FP32 master is a port representation choice,
  not an official recipe.** `refresh_storage` requantizes using the C row/block32
  quantizer after an accepted update. No offload occurs. Parent dtype conversion
  preserves FP8 storage and the FP32 master. Distributed sharding, main-grad and
  optimizer/publication hooks belong to E; this local provider does not claim them.

## Interfaces and evidence

Module paths are relative to `experimental/lite/megatron/lite/`; test paths are
relative to `experimental/lite/`.

| Interface | Behavior | CPU evidence |
|---|---|---|
| `model/deepseek_v41/lite/block.py` | Paired `[B,S,HC,D]` hidden and `[B,S,HC]` pre-mix; shifted attention/FFN contraction and explicit `forward_with_state` triple | `test_hc_boundary.py`, `test_attention.py`: distinct copies, analytical gradients, CED, paired recompute; `test_d_reference.py`: scalar Sinkhorn equations, all coefficient/input VJPs and updates |
| `model/deepseek_v41/lite/attention.py` | KV owners 2/8/14/20, index owners 2/8/14/20/24/28/32/36; RoPE-free latent branches to index K before main quantization; frozen indexer | `test_attention.py`: source identity, empty prefix, shared gradient sums, independent quantizers, optimizer exclusion; `test_d_reference.py`: all 40 floating attention outputs, published KV, indices, input/parameter VJPs and updates |
| `model/deepseek_v41/lite/candidates.py` | Per-block maxima, newest reachable block pinned; native selection within the pool | `test_candidates.py`: outside-pool winner and incomplete/empty prefixes; `test_d_reference.py`: original pinned function with multiple budgets |
| `model/deepseek_v41/lite/engram.py` | Official normalization/hash layout and sequence/image resets; per-copy signed-sqrt gate; local frozen/trainable FP8 provider | `test_engram.py`: integer cards, masked gradients, FP32 master and scale refresh; `test_d_reference.py`: original tokenizer map, prime/multiplier layout, hash, floating forward and every parameter VJP/update |
| `model/deepseek_v41/lite/moe.py` | DS4 shared router extension for explicit per-token selection bias; modality statistics; explicit updates; local expert dispatch with weight applied before down projection | `test_moe.py`: unbiased weights/gradients, two updates, empty modality, TP reduction call contract, dispatch; `test_d_reference.py`: original Gate/Expert output and parameter VJPs/updates |
| `model/deepseek_v41/lite/packing.py` | Unpadded single-rank THD splitting through shared `primitive/utils/packed_seq.py`; pure per-sequence callable | `test_packing.py`: Engram→Full/Reuse/Reindex→MoE; fixed weights/bias/RNG and B-only loss; perturb A and compare B output, input gradients and all parameter contributions |

Initial HC contraction selects copy zero; final contraction consumes the last
returned pre-mix. Layer20 receives
`attn_norm20(contract_hc(h20,p20))`; its compressor returns the normalized latent.
No decoder projection is introduced. Both HC tensors and immutable attention
state must survive recompute/transport. State lifetime is caller-owned: shape
validation does not identify a microbatch. Each packed sequence callable starts
with fresh state, local positions and hash history.

## Reference profile and limits

`tools/deepseek_v41/d_semantics_reference.py` verifies the pinned source digest
before compiling unchanged original methods. CPU attention explicitly disables
quantizers and replaces the GPU sparse kernel with an independent per-query,
per-head gather/softmax equation. Its FP32 diagnostic results and autodiff
updates verify the floating port contract; they **do not certify native quantized
kernels, the official training recipe, or a runnable 40-layer model bundle**.
The scalar mHC card independently transcribes the pinned kernel equations.
Other floating Gate/Expert/Engram methods and integer hash/candidate functions
run directly from the pinned source. Tests fail on missing or changed references;
`DS41_REFERENCE_DIR` can point to another exact copy of that snapshot.

Default attention FP8 Linear invokes the C native `_scaled_mm` path and requires
CUDA. CPU floating tests explicitly disable it. Separate C/D cards exercise the
actual CPU main/index/SWA quantizers and STE. Attention materializes dense scores;
fused sparse performance, TP/EP/CP, decode and optimizer transport are outside
this local implementation's certification.

A deliberately undersized custom candidate pool marks selected `-inf` positions
as padding, preventing excluded positions from returning through Top-K padding.
This differs from the published function's behavior for that custom case; the
40-layer reference uses sufficient pool capacity. Backend-dependent ties still
require the B3/native profile qualification.

Local modules and injected projection/expert providers do not constitute C4
checkpoint-key binding or protocol registration. Runtime optimizer-step hooks,
full bundle integration and native GPU qualification remain separate gates.
No open D5/D6 objective decision is a remaining dependency.

## Reproduction

From the repository root, run the D CPU suite (including independent reference
and packed composition):

```sh
PYTHONPATH=experimental/lite OMP_NUM_THREADS=1 python -m pytest \
  -c /dev/null --confcutdir=experimental/lite --rootdir=experimental/lite -q \
  experimental/lite/tests/unit/deepseek_v41/test_hc_boundary.py \
  experimental/lite/tests/unit/deepseek_v41/test_attention.py \
  experimental/lite/tests/unit/deepseek_v41/test_candidates.py \
  experimental/lite/tests/unit/deepseek_v41/test_engram.py \
  experimental/lite/tests/unit/deepseek_v41/test_moe.py \
  experimental/lite/tests/unit/deepseek_v41/test_packing.py \
  experimental/lite/tests/unit/deepseek_v41/test_d_reference.py
```

Also run the existing Sigmoid router auxiliary/replay regressions after shared
router changes. TE imports are lazy only to expose the existing unfused CPU
arithmetic; fused operations still import and call TE without a fallback.
Cluster regression uses the existing `tests7064.sbatch` carrier and `dev7064`
baseline. Acceptance requires no additional failed test IDs relative to that
baseline, not a claimed zero-failure run.

# V4.1 single-rank semantic implementation: work in progress

This snapshot is not D-phase acceptance or a runnable model bundle. It consumes
C-phase codecs and payload/oracle tooling at `06d4e1c03`. Forward arithmetic is
based on the pinned official source identified by `deepseek_v41_oracle.md`.
Training mathematics is specified at `f725fe983:experimental/lite/docs/specs/
deepseek_v41_training.md`; that specification is not an implemented B2 oracle.
The work-package contract is `766bc22d1:experimental/lite/docs/plans/
deepseek_v41_tasks.json`. Its D8 boundary work is part of D1/D2.

## Implemented interfaces and diagnostic cards

Module paths below are relative to `experimental/lite/megatron/lite/`; test paths
are relative to `experimental/lite/`. These are callable local
building blocks, not protocol registration or checkpoint-key binding.

| Interface | Behavior | Executable cards |
|---|---|---|
| `model/deepseek_v41/lite/block.py` | `[B,S,HC,D]` hidden plus `[B,S,HC]` pre-mix; incoming mix feeds attention, attention mix feeds FFN, FFN mix is returned; source/destination residual orientation; FP32 coefficient arithmetic and official epsilon placement | `tests/unit/deepseek_v41/test_hc_boundary.py`: unequal copies, explicit sublayer inputs, analytical VJPs, zero-projection Sinkhorn card, paired non-reentrant recompute |
| `model/deepseek_v41/lite/attention.py` | Explicit immutable `AttentionState`; KV sources 2/8/14/20, index sources 2/8/14/20/24/28/32/36; unrotated latent feeds index K; ratio 1 uses compressed RoPE/YaRN; grouped output projection; independent QAT switches | `tests/unit/deepseek_v41/test_attention.py`: ratio-1 rotation/inverse, source identity, empty prefix, CED pair gradients, two consumer KV-gradient contributions, actual codec call ordering |
| `model/deepseek_v41/lite/candidates.py` | Per-block maxima, newest reachable block pinned, native second-level selection within the pool; empty inputs accepted | `tests/unit/deepseek_v41/test_candidates.py`: explicit membership, outside-pool winner, unreachable blocks, incomplete tail, pretraining rejection |
| `model/deepseek_v41/lite/engram.py` | Token normalization and odd multipliers, prime layout, per-call integer hash with complete image/packed resets; injected table/projection, per-copy RMS and signed-sqrt sigmoid gate | `tests/unit/deepseek_v41/test_engram.py`: fixed products/XOR/remainders/offsets, image and sample reset, floating gate and masked parameter contributions |

Initial HC contraction is one-hot on copy zero; final contraction must use the
last returned pre-mix. At the CED boundary the caller must supply
`x20 = attn_norm20(contract_hc(h20,p20))` to attention20. Its compressor returns
`latent20 = compressor_norm20(compressor_wkv20(x20))`; no decoder projection is
introduced. A caller must keep both h20 and p20 alive through recompute/transport.

`CSA2Attention.forward` returns `(output, AttentionState)`. The block now offers
`forward_with_state(hidden, pre_mix, state)`, returning
`(hidden, next_pre_mix, state)` through the same shifted-HC arithmetic as its
tensor-only forward. State remains an explicit per-call graph value. Composed
CPU cards cover layer20 CED, layer21 Reuse, layer24 Reindex, non-reentrant
recompute gradients and isolation between equal-shaped independent calls.
The full-layer driver and protocol ownership transport remain unimplemented. Reuse never owns an indexer; Reindex uses its own Q and the
existing index K. State lifetime is caller-owned; shape checks alone do not prove
microbatch identity. Do not reuse state between equal-shaped independent samples.

The attention backend explicitly materializes dense scores. Native fused sparse
ABI/parity, production memory behavior, TP/CP and decode are not certified.
Default linear FP8 calls the C-phase native `_scaled_mm` implementation; explicit
`linear_fp8=False` is a diagnostic profile, not a fallback. Unit tests disable
FP8 Linear. A separate test enables SWA FP8 and both cache QAT paths on CPU and
checks actual post-RoPE values plus independence of the main/index switches.

Candidate selection marks every selected negative-infinity score as padding;
this also prevents an undersized custom candidate pool from reintroducing an
excluded position. Official parity for those custom undersized pools has not
been established. Top-K tie behavior remains backend-dependent, and the present
cards do not establish the complete B3 margin/tie contract.

Engram receives embedding and projection providers; these must eventually be
bound to the approved table/FP8 Linear implementations. It makes no decision
about master weights, trainability or scale regeneration (O08/O09). The token
normalizer and automatic prime/multiplier constructors still require official
fixture parity; the current hash card injects known integer constants.

## Remaining contract and implementation gates

* D1–D4: independent official fixture comparisons, complete mutations, all input
  and owner parameter VJPs/updates, native GPU validation and S-gate review remain.
* D5: no MoE implementation in this snapshot. B2-S explicitly leaves the exact
  per-sequence/per-modality auxiliary objective and post-training activation
  unresolved (training specification lines 142–149 and 208–214). The existing
  DS4 assembly uses `compute_aux_loss=False`. A coefficient of 0.0001 is not a
  complete objective. Bias update/statistics, router extension and runtime step
  integration also remain; none is silently supplied here.
* D6: O12 remains OPEN. D6-S/I require an objective, coefficient, normalization,
  detach/contributor table and independent numerical mutations. No objective or
  dummy successful loss is implemented, and B2-I is not claimed.
* D7: only integer hash reset cards exist. A real composed packed path with fixed
  model/bias/RNG/objective, B-only backward and parameter-contribution comparisons
  is still required. No packed-protocol or CP claim follows from hash isolation.
* C4/full model binding, E-phase state transport, and full 40-layer production
  integration are not delivered by these standalone modules.

## Reproduction and next bounded steps

Run the current CPU cards from the repository root:

```sh
PYTHONPATH=experimental/lite OMP_NUM_THREADS=1 python -m pytest \
  -c /dev/null --confcutdir=experimental/lite --rootdir=experimental/lite -q \
  experimental/lite/tests/unit/deepseek_v41/test_hc_boundary.py \
  experimental/lite/tests/unit/deepseek_v41/test_attention.py \
  experimental/lite/tests/unit/deepseek_v41/test_candidates.py \
  experimental/lite/tests/unit/deepseek_v41/test_engram.py
```

The initial runs collected errors for the missing target modules before their
implementation. One candidate-card expected result was corrected from offsets
`[12,13]` to `[13,19]`: the unmasked second-largest score is at position 8.

After the remaining recipe/scope contract is settled, proceed in small steps:

1. Extend the tensor/state composition cards in `test_attention.py` to the full
   owner/consumer schedule and all parameter VJPs; the initial two-layer cards
   are diagnostic coverage, not D1/D2 acceptance.
2. Add independent pinned-official comparisons to
   `tools/deepseek_v41/validate_d_semantics.py`; run first on the smallest B3
   fixture before broadening coverage or invoking Slurm.
3. Define reviewed D5 objective/statistics vectors before introducing `moe.py`
   and `test_moe.py`; retain D6 as unresolved until O12 is resolved or its full
   acceptance is explicitly deferred.
4. Implement `packing.py` and `test_packing.py` over the composed path; compare
   B-only outputs/VJPs under A perturbation, then validate the approved native
   profile through Slurm with immutable source and explicit non-skip evidence.

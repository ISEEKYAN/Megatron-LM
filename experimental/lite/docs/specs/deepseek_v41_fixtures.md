# DeepSeek-V4.1 fixture specification

This B3-S contract supplies dimensions, deterministic weight rules and independent
expected cards to B3-I. It contains no model implementation or fixture generator.
The companion `deepseek_v41_fixture_vectors.json` is normative input/expected data,
not output captured from MLite. B3-I produces
`tests/fixtures/deepseek_v41/manifest.json` and a generator; its `test_fixtures.py`
must verify shapes, reproducibility, owner identity and the cards below before
B1/B2 consume them. G1 still requires actual dimensions and checkpoint weights.

## Source precedence

Read these repository artifacts at the specified commits, even before their
branches are integrated. Paths below are relative to `experimental/lite/`.

| Prerequisite | Immutable source | Consumed rules |
|---|---|---|
| Corrected A1 | `2b2c7e0c324be7df47f66fe18be393e53bef2fee`, `docs/deepseek_v41_owner_consumer.md`, `tools/validate_deepseek_v41_ced.py` | Paired CED state, 40 owners/modes, eleven behavioral mutations |
| A1/A3 correction | `9d1697a28`, `docs/specs/deepseek_v41_csa2.md`, `tools/validate_deepseek_v41_{owner_consumer,csa2}.py` | Positions, quantization order, ratio-1 compressor |
| A2 | `b3a916425`, `docs/contracts/deepseek_v41/{README.md,weights.json,config.json,validate.py}` | All 96,085 keys, 2,401 MTP keys, header validation and byte carrier |
| A4 | `c4c27b0e6`, `docs/specs/deepseek_v41_optimizer{.md,_vectors.json}` | Semantic groups, numeric constants and Algorithm 1 cards |
| A8 policy | `766bc22d1396c30a6c7d08deabe64f0a56e84d8f`, `docs/plans/deepseek_v41_{decisions.json,a8_evidence.md,phases_bg_v4.md}` | Resolved post-training policies supersede A4 OPEN entries only where explicitly resolved |

Official revision: `df42c109f1defefcbfcedbe7d905718a12266e40`.
The local official config, model, Engram source and index digests are recorded in
the companion JSON. This is a pinned-source contract, not a freshness claim.
No copied fixture may silently use pre-correction raw H20 or reopen A8 policy.

## Profiles and dimensions

`reduced-forward-v1` preserves every global backbone layer and ownership boundary.
The reductions are synthetic test choices, not inferred checkpoint shapes.
`operator-position-v1` uses isolated long-position/candidate cards without allocating
a million-token model. `release-v1` uses the unmodified official config and every
real header/payload: allocation readiness or a reduced run cannot satisfy it.

| Dimension | reduced-forward-v1 | release-v1 |
|---|---:|---:|
| Backbone layers / HC copies | 40 / 4 | 40 / 4 |
| Hidden / vocabulary | 128 / 256 | 5120 / 129280 |
| Query heads / KV heads / head dimension / RoPE tail | 8 / 1 / 64 / 64 | 64 / 1 / 512 / 64 |
| Q rank / output groups / per-group output rank | 64 / 8 / 32 | 1280 / 8 / 1024 |
| Index heads / index head dimension | 4 / 64 | 32 / 128 |
| Routed experts / selected / shared / FFN width | 8 / 6 / 1 / 64 | 384 / 6 / 1 / 2304 |
| SWA / position Top-K / candidate block / candidate Top-K | 128 / 512 / 8 / 2048 | same |
| Engram layers / max ngram / hash heads / head width | [1,14] / 4 / 8 / 32 | [1,14] / 4 / 8 / 256 |
| Engram prime search base / compressed vocabulary | 31 / 256 | 16000000 / 99092 |
| Vision depth / width / heads / FFN width | 2 / 64 / 4 / 128 | 32 / 1024 / 16 / 2816 |
| Patch side / aligner downsample factor | 14 / 3 | 14 / 3 |

Reduced Engram rows are the sums of the 24 primes for each layer obtained by
official `EngramLayout.from_args` rule: start at base-1 for each ngram,
choose strictly increasing primes unused in all preceding layers/ngrams/heads.
Do not set arbitrary row counts. B3-I records the resulting complete prime and
offset arrays and checks them with independent integer primality arithmetic.
Reduced token map is identity for IDs 0..255 (pad=2, image placeholder=255);
it deliberately does not test the real tokenizer normalization. Real tokenizer
hash coverage requires the pinned tokenizer, official normalization and 99092
compressed IDs. The two Engram tables remain separate, nonzero and resident;
no offload policy or table trainability is inferred from this profile.

Norm eps=1e-20, HC eps=1e-6, HC iterations=20; theta=10000 for ratio 0,
160000 with YaRN factor=16, original length=65536, beta_fast=32, beta_slow=1
for ratios 1 and 2. Keep these constants in reduced tests. Scaling uses the
actual profile head dimensions (index uses both head-width and head-count
inverse square roots); do not hardcode release scaling in reduced operators.

For each layer i: ratio is 0 for 0..1, 2 for 2..19, 1 for 20..39.
KV owners are [2,8,14,20]; index owners [2,8,14,20,24,28,32,36].
For i>=2 choose the greatest owner <=i independently in each list.
Full={2,8,14,20}, Reindex={24,28,32,36}, Reuse=the other 30 CSA2 layers.
Layers 0,1 are SWA only; each of all 40 layers has its own local KV.
The manifest MUST contain 40 explicit rows, not just these generating rules.
Candidate owner is 20. Engram remains at 1 and 14; MTP never becomes layers 40+.

## Manifest and tensor interfaces

The generated manifest requires `schema_version=1`, `profile`, exact source
commits/digests, explicit config overrides, `active_decisions`, `inactive_decisions`
with reasons, all owner rows, trainability mask (or `not_applicable` for forward),
weight recipe version, logical-name/shape/dtype/bytes/digest records, input cases,
expected-card IDs and reference provenance. Missing/unknown fields that change
semantics, missing owners and unmatched weight/scale pairs are errors.

Use BSH at the oracle boundary. Packed data has `cu_seqlens` int64 plus explicit
sample IDs and sequence-local positions; never derive positions from physical
packed offsets. `input_ids` int64, `valid_mask`/`text_mask` bool, image spans
half-open `[start,end)` in sequence-local coordinates. Floating tensors have
separate reference and execution dtype metadata; no implicit downcast of expected
values. THD/SBH adapters must round-trip token identity and positions exactly.

| Value | Logical shape |
|---|---|
| CED raw h20 / paired p20 | [B,S,4,H] / [B,S,4] |
| hc_pre then attn_norm input x20 | [B,S,H] |
| pre-main-RoPE compressor latent | [B,floor(S/r),D] at each KV owner |
| index K / index Q | [B,floor(S/r),Di] / [B,S,Hi,Di] |
| main query / local KV | [B,S,Hq,D] / [B,S,D] |
| global integer selections | [B,S,min(512,floor(end_pos/r))], int32 |
| candidate mask | [B,S,floor(end_pos/r)], bool |
| grouped wo_a / wo_b weights | [G,Ro,(Hq/G)*D] / [H,G*Ro] |
| Engram hash IDs / retrieved vectors | [B,S,2,24] / [B,S,24,De] per table |
| Engram projection / normalization gains | [5*H,24*De] / [4,H] |
| aligned image vectors / expanded vectors | [Nimage,H] / [Nimage,4,H] |
| all-token logits | [B,S,V] (padded positions masked separately) |

Packed compressor state restarts at each sample; shapes above describe each
sample before padding. Empty completed prefixes have width zero or sentinel-only
rows as prescribed by the official wrapper, never a fabricated visible token.
The CED card probes the collapse before attention norm, and then separately
records x20, latent20, index K before/after its RoPE/FP4, main KV before/after
its RoPE/FP4 and every decoder output. It must not introduce a decoder projection.
Distinct incoming, attention and FFN pre-mix probes from corrected A1 are required.

## Precision boundaries

CPU algebra cards use float64 with atol=rtol=1e-12; exact integer, boolean,
owner identity and archival byte checks use equality. This is a declared unit-card
threshold, not a hardware acceptance tolerance. RoPE cards use the A3 FP32
atol=rtol=1e-6 at the listed actual positions. Full-model BF16/FP8/FP4 tolerances
require B1/B2/G1 qualification, with reference and target error reported separately.
Never cast a BF16-produced gradient to FP32 and label it native FP32 main_grad.

Keep separate encoded and decoded fixtures for (1) main post-RoPE KV: FP4,
block16/E4M3 scales; (2) index post-RoPE Q/K: FP4, block32/E8M0 scales;
(3) SWA post-RoPE full KV: FP8; and (4) serialized FP8 weights with 32x32
UE8M0 scales and packed FP4 experts. These are distinct codecs. Ratio-2
compressor wkv/wgate projections and pooling operate in FP32 before cast/norm;
ratio-1 projection is BF16 followed by norm, without gate. B3-I records raw codes,
scales and independent decoded values from C1/C2's approved references, including
zero blocks, sign, saturation and rounding ties. This spec does not invent missing
quantizer code/scale expected bytes or use a floating card as codec acceptance.
Kernel tile constraints may require a separate explicitly labelled ABI profile;
padding cannot change logical head counts, owner identities or normalization axes.

## Weights and independent expectations

For reduced dense logical tensors, sort canonical release-style names by Unicode
code point; ordinal a starts at zero. For C-order flattened index j, use the exact
rational `(((17*a + 13*j + 7) mod 101)-50)/256`. Norm gains use `1 + value/16`;
learned scalar multipliers use `1 + value/16`; biases and sinks use `value/16`.
Use separate ordinals for embed and head (untied). All owner parameters are
materialized once; consumers refer to owner IDs, not separately initialized copies.
This recipe is deterministic without framework RNG or platform-specific hashes.
It is a coverage stimulus, not a guarantee of a Top-K margin. Selection tests use
the explicit constructed cards below; reject a random dense run as margin evidence.

Logical reduced families come from the same semantic A2/A4 module roles with
explicit reduced domains (8 experts and 2 vision layers). Their key count is NOT
96,085. Do not invent compressor gates on ratio 1 or index K weights on Reindex.
A separate full-key archival case expands A2's exact 96,085 names, including all
2,401 MTP names, without shrinking their domains. Actual dtype/shape/byte_length
come only from safetensors headers, never the shard index. Full archival weights
are raw bytes; numerical binding and transformed export are separate tests.
MTP is immutable, excluded from autograd/optimizers and preserves stage-specific
key domains. Execution stays disabled without rewriting release dspark_block_size.

Expected JSON cards are hand-specified rational/integer examples. B3-I may encode
them but must not regenerate expected outputs through the tested model, kernels,
cache builder or optimizer. A4's 70-digit S7 vector remains its independent
optimizer reference; do not rebaseline it from DS4. B1 supplies official floating
forward traces; B2 supplies differentiable objectives and update vectors. Preserve
both reference identities; a shared execution path is not independent parity.

## Positions, boundaries, masks and images

At local query p and ratio r, group j has rotary position j*r and becomes visible
only when (j+1)*r <= p+1. SWA positions are max(0,p-127)..p inclusive.
Global cache indices use the official wrapper offset after local selection;
`-1` remains `-1`. Offset is a cache layout coordinate, not a rotary position.
Ratio 1 still uses compressed theta/YaRN. Never deduplicate local/global entries
merely because their token positions coincide; they are different KV sources.

Required local lengths: [1,2,3,7,8,9,127,128,129,511,512,513,16383,16384,16385].
The long lengths are operator cards. Prefill/decode splits include [1,2], [7,2],
[127,2], [511,2], [16383,2]; compare with unsplit execution and interleave two
requests to detect stale source state. RoPE-only queries use positions
[0,1,127,128,65535,65536,131071,1048575] with a genuine 64-feature rotary tail.
Packed case lengths [3,9,129], cu_seqlens=[0,3,12,141], plus padding to 144;
each sample restarts causal, compression, Engram and RoPE history independently.

At candidate level retain block size 8, block score=max reachable position score,
pin newest reachable block with +infinity before selection, drop -infinity blocks.
An empty prefix retains nothing. Add an isolated 2049-block card (16392 positions)
with block b score 4096-b, Kblocks=2048: newest block 2048 is pinned, so keep
blocks 0..2046 and 2048, rejecting 2047. This crosses the REAL block Top-K limit.
Position card scores[j]=1024-j for 513 positions with K=512 keeps 0..511;
replace score[512] by 2048 to keep 0..510 and 512. A mere small-prefix card
cannot demonstrate either production selection boundary.

Two reduced images use 9x9 patch grids (126x126 pixels each), aligned to 3x3
features. Include start, three rows each followed by newline, and end: 14 tokens
per image. In a length-37 sample use spans [3,17), [19,33); both start and end
are nonaligned to ratio 2 and block 8. This synthetic image is below release
minimum pixels, so enter through the operator patch-grid interface with that
explicit override; do not claim release preprocessor acceptance. Each vector
copies identically into all four HC slots. Use upstream gradients [1,2,4,8]
per copy: the input vector gradient is 15 per feature. Shifted spans, a dropped
copy or detached encoder must fail; equivalent copy implementations may pass.
Engram text_mask=False for the entire image span, including delimiters/newlines;
lookback pads at every dead token and sample start. Test repeated IDs across an
image and across packed samples: neither can import pre-boundary history.

## Margins and ties

Measure scores at the actual selection boundary AFTER quantization, TP reduction,
causal/candidate masks and modality correction bias. Let e bound the absolute
per-score discrepancy of both implementations against the independent expected
score. Require finite cutoff gap delta=s[K-1]-s[K] > 2*e for membership parity.
Use e=1/8 and delta>=1 for the integer-score cards. This bound is a card contract,
not an asserted BF16/model error bound; if observed errors exceed e, the test
fails qualification. Record dtype, pre/post-quant scores, gap and measured error.
Pinned infinities and invalid sentinels are tested separately from finite margins.

Official topk does not specify a stable tie winner. Positional sorting AFTER
selection does not choose tied membership. Tie cards therefore require all
strictly better elements, no strictly worse elements, correct cardinality, unique
valid indices and positional output order, permitting any subset at the cutoff.
For tied membership with different values, compare downstream output against the
reference with the SAME injected selected map; do not assert raw output equality
across two lawful different maps. A future deterministic tie-break is a new
explicit policy, not an assumed official lower-index rule. MoE expert ties use
the same admissibility check, with scoring and selection bias kept distinct.

Ranking-reversal card in JSON has non-collinear queries and a selected/unselected
crossing. Source replay is a positive equivalence card; positive scalar query
rescaling is not a negative control. Reindex must use its new query with source
index K; Reuse must preserve source selections while retaining local SWA.

## A4 policies and implementation-stage tests

Keep optimizer K=11 distinct from HC K=20; use beta=0.95, tau=0.001,
epsilon=1e-20, gamma=0.18 for Algorithm 1 expected math. A4 S1..S8 and R1..R7
remain mandatory downstream cards. Include zero/no-decay, threshold equality,
nonuniform heads, norm gains versus learned scales, independent text/image
correction loads, aliases, MTP exclusions and changed second-step gradients.
The companion ownership gradient card uses distinct consumer vectors and two
owners to expose omitted, duplicated or misdelivered contributions. It is a
floating-KV linear diagnostic, not an indexer auxiliary objective.

This specification's active policy IDs are O01,O02,O03,O04,O07,O10,O11,O13,O14,
O16,O17. They prescribe downstream conditional groups, FP32 main_grad and momentum,
STE with detached scales, modality reduction and valid-token weighting. All are
RESOLVED in pinned A8; apply Engram projection/norm 5x only when trainable.
O05/O06 are inactive: no pretraining unfreeze schedule is tested. O15 is inactive:
no FP4-off model parity is claimed by mathematical cards. O08/O09 are inactive
for reduced-forward-v1: no quantized table update, master storage or regeneration
is chosen. O12 is inactive: no indexer training objective is chosen. These last
three remain OPEN. Forward lookup does not assert that production tables are
frozen. Any profile enabling quantized table updates activates O08/O09 and must
block until resolved; indexer training similarly activates O12. Full training
acceptance is not authorized by this forward/fixture profile.

Run the pinned A8 `validate_deepseek_v41_plan.py` once per active ID using
`--active-decision ID`. Also run O08/O09/O12 as negative controls and require
nonzero exits naming the unresolved ID. Record source commit and exit codes.
The plan validator checks decision status and graph structure, not this prose,
model precision or whether a profile truthfully declares its active decisions.

B3-I must validate positive cards AND mutations: wrong owner, raw-H20 bypass,
pre-mix shift, index K from post-RoPE main KV, ratio-1 wrong theta, partial group
visibility, packed offset treated as position, missing newest block, stale Reindex
map, head/group permutation, image span shift, cross-boundary hash, margin collapse,
and loss of weight/scale or MTP bytes. Report each observable discrepancy. CPU
mathematical checks of these expectations do not certify GPU or full-model parity.

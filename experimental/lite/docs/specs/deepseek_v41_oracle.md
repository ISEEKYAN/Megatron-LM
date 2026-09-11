# DeepSeek-V4.1 official forward oracle contract

Status: specification for review. This defines B1-S; it does not implement
`tools/deepseek_v41/oracle.py`, certify B1-I, or establish training parity.
Paths are relative to `experimental/lite/`. B1-I consumes B3-I's generated
dimensions, weights and token fixtures before its execution gate can close.

## Reference and prerequisite boundary

Use the audited DeepSeek-V4.1-Flash revision
`df42c109f1defefcbfcedbe7d905718a12266e40`, not an unpinned remote main.
The inspected `inference/model.py` SHA-256 is
`4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
Its `config.json` SHA-256 is
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.
This is a local pinned-source audit, not a latest-release assertion.

Consume these existing contracts without recreating their manifests:

| Prerequisite | Required content and use |
|---|---|
| A1 `docs/deepseek_v41_owner_consumer.md` | Corrected paired CED state, 40-layer ownership, floating KV versus discrete selection dependencies |
| A2 `docs/contracts/deepseek_v41/{README.md,weights.json,config.json}` | Exact 96,085-key and 87-leaf coverage, header/payload binding, 2,401 inactive MTP keys |
| A3 `docs/specs/deepseek_v41_csa2.md` | Operator order, three modes, RoPE, distinct quantizers, candidate and sparse ABI |
| A4 `docs/specs/deepseek_v41_optimizer.md` | Parameter roles, modality biases and separation of inference from training recipes |
| A8 `docs/plans/deepseek_v41_phases_bg_v4.md`, `deepseek_v41_decisions.json` | Post-training scope, current decision resolutions and S/I boundaries |

A1 must include the replacement at `2b2c7e0c3` and the executable CED entry
point from `9d1697a28`; A2 was inspected at `b3a916425`, A3 at `9d1697a28`,
A4 at `c4c27b0e6`. The current base does not contain the A7 commit as an
ancestor. This document consumes those reviewed source artifacts externally;
it does not assert they are merged. Before B1-I, verify landed ancestry or
reviewed replacement digests and resolve the prerequisite files in its tree.
The corrected A7 CED probe has eleven mutations; the older eight-mutation
probe and old integer-gradient arrows are insufficient.

## Request interface

The planned API is `run_forward(request) -> OracleResult`. Requests and results
use schema version `deepseek-v41-oracle-v1`. An immutable manifest identifies
the request independently of tensor filenames. B1-I must reject unknown fields,
missing required inputs and mismatched manifest/payload digests before execution.

| Field | Contract |
|---|---|
| `reference` | Revision and SHA-256 of model, kernel, Engram, vision, conversion, processor and tokenizer files actually used; record dependencies, Torch/CUDA and kernel revisions |
| `fixture` | B3 fixture ID, manifest digest, seed, full/reduced flag, complete explicit dimensions and expected tensor identities; no implicit `ModelArgs` shape defaults |
| `config` | Original release JSON plus a separate explicit runtime override map and complete release-leaf → ModelArgs/store/scope-waiver mapping; validate against A2 |
| `weights` | Converted official shards with a release-key → converted-key/rank/slice/dtype transform manifest; header and payload digests, scale pairing and alias ownership |
| `sequences` | Ordered independent sequence IDs and int64 token vectors; every ID in `[0,vocab_size)`, nonempty, no padding passed to official forward |
| `chunks` | Per-sequence contiguous schedule: first chunk starts at 0; later chunks have length 1 and `start_pos` equals the consumed prefix length; total length ≤ runtime cache limit |
| `images`, `token_types` | Optional official processor output per sequence; type tensor has token shape, TEXT=-1, image-span types ≥0; all spans contained in initial chunk |
| `execution` | One model per process, rank/world size and shard mapping, device, deterministic seed, inference mode, temperature=0, published quantization profile |
| `capture` | Named stage set below; full CED/consumer coverage mandatory for B1-I, explicit token/layer coverage manifest; capture limits cannot silently truncate |
| `enable_dspark_execution` | Must be false; true raises `NotImplementedError` before DSpark allocation or execution |

Convert the nested release config through an explicit field mapping into official
`ModelArgs`; do not pass it through a permissive dictionary filter. Preserve the
original `dspark_block_size=5` in archival metadata. The oracle may use an
explicitly recorded constructor-only override `dspark_block_size=0` to avoid
official `Transformer.__init__` allocating DSpark blocks. This is an inference
harness adaptation, never a rewritten export config or a production execution
switch. Preserve MTP payloads in the independent A2 carrier and do not call
`forward_spec`. Retain vision, aligner and image vectors even in text-only cases.

Each active converted tensor must be loaded exactly once into its canonical
official object/slice, with separate records for remote shards and aliases.
Missing/extra/duplicate/unbound keys fail; no `strict=False` acceptance without
an exact accounted key-set difference. Inactive MTP keys are covered as stored
bytes, not as executed parameters. Fixture weights must be nontrivial and
distinguishable, including scales, biases, embedding rows and head rows.
An output perturbation cannot prove use of an unrouted expert; coverage also
requires binding records and targeted expert/submodule executions in B1-I.

Official forward accepts rectangular `[B,S]` real-token batches with a common
`start_pos`; it does not accept a THD packed attention mask. The canonical oracle
runs each logical sequence independently with fresh state and concatenates
outputs in declared sequence order. Equal-length batching is an additional
equivalence test. Ragged/padded/packed native input is compared through explicit
`cu_seqlens`/valid-position mapping, never sent as one official sequence.
Reset position, Engram history, partial compressor groups and caches between
independent sequences. Chunk continuation retains them. Use fresh processes or
fresh complete model/runtime/hash state; do not reset only `shared_attn` while
retaining per-layer cache history. Concurrent models in one official module
namespace are invalid because rank, dtype and shared state are module globals.

For images, record source and processor digests, patch dtype/shape, `n_vit_h/w`,
span start and types. Require nonoverlapping in-bounds spans, matching IMAGE-slot
and aligner-row counts, and image placeholder IDs. Include start/end/newline
slots in image masks. Use the official preparation/patch ordering; do not invent
resize or grid-padding semantics. Missing processor/tokenizer dependencies fail
the multimodal/Engram case explicitly, without replacing it with text-only.

## Result and all-token head

`OracleResult` contains `manifest`, `sequences`, `captures`, `load_coverage`
and `comparison`. Each sequence returns FP32 `logits [S,V]`, sequence-local
positions, chunk boundaries, and the official last-token logits `[V]` for each
chunk. Do not return loss, gradients, optimizer updates or generated rollouts.
The manifest includes input/weight/config digests, effective ModelArgs,
constructor override, rank layout, RNG state/seed, dtype policy, source/adaptation
digests and executed/nonexecuted coverage. Failures carry a stage and reason;
missing dependencies, skipped stages and nonfinite numerical outputs are failures,
not passing comparisons. Masked scores may contain the prescribed `-inf`.

Run original `Transformer.forward` and original submodule methods. Install a
scoped pre-hook on the *backbone* head to capture a clone of its normalized
`[B,S,D]` input. Let the original forward/head call complete unchanged, then
call the original `ParallelHead.forward(captured_input, full_logits=True)`.
Remove hooks in `finally`. This preserves official final shifted HC contraction
and norm, avoids rewriting the model loop, and uses the actual FP32 head weight
and vocabulary all-gather. Every rank must participate in this extra call.
Compare its final slice with the original logits and independently reconstruct
small head fixtures as below. Do not attach the hook recursively to the second
head call. Sampling at temperature=0 is only the official forward's unavoidable
argmax side effect; its IDs are not a parity substitute.

## Instrumentation contract

Every event has `(sequence_id, chunk_id, layer_id, stage, occurrence)`, global
owner IDs, sequence-local query/key positions, dtype/shape, rank and tensor
digest. Capture by immediate detached clone on the producing device, with
synchronization before serialization. A detached view is insufficient because
official RoPE and quantizers mutate storage in place. Capture must not alter
arguments, returns, RNG, kernels, dtype, ordering or cache publication. Test
instrumented and uninstrumented runs from identical fresh states. Retain logical
owner/publication identity separately from cloned tensor storage identity.

Let B/S denote batch/chunk length, H the HC multiplicity, D hidden width, K main
head width, I index head width, C the completed compressed prefix length. All
shapes use the fixture dimensions; released H=4, D=5120, K=512, I=128.

| Stage | Capture point and shape / meaning |
|---|---|
| `embedding`, `image_merged` | Before/after merge, `[B,S,D]`; `hc_expanded [B,S,H,D]` verifies every copy |
| `engram.hash`, `engram.output` | Integer hash IDs and masked Engram output at layers 1,14; preserve token map and modality mask |
| `block.input`, `block.pre_mix` | Block inputs `[B,S,H,D]`, `[B,S,H]`; no averaging replacement |
| `attn.input`, `attn.output` | After shifted `hc_pre` and attn norm / Attention return, each `[B,S,D]`, for all 40 layers |
| `ffn.input`, `ffn.output` | After collapse using `attn_pre` and FFN norm / MoE return; record modality expert IDs and unbiased weights |
| `block.output`, `block.next_mix` | Block return residual and `ffn_pre`, preserving the pair; Block(19) gives `h20,p20` |
| `ced.x20` | Layer20 `attn.input`, not raw h20; `[B,S,D]` |
| `compressor.latent_pre_rope` | Compressor return cloned before Indexer/main mutation, `[B,new_C,K]`; owner20 alias `latent20` |
| `index.k_pre_rope` | `k_norm(wk(latent))` return before index RoPE/FP4, `[B,new_C,I]`; owner20 alias `index_k20` |
| `index.k_published` | Published shared index K active prefix `[B,C,I]`, after its own RoPE and group32/E8M0 FP4 |
| `main.kv_published` | `_compress_kv` active-prefix return `[B,C,K]` after main RoPE and group16/E4M3 FP4; owner20 alias `main_kv20` |
| `index.q`, `index.scores` | Post-index-quantized Q and reduced scores before/after causal and candidate masking; record scales, local heads and TP sum |
| `candidates`, `topk` | Source20 Boolean pool `[B,S,C]`; every index publisher's int32 selection, offsets and -1 slots; Reuse read events |
| `swa.kv`, `sparse.args`, `sparse.output` | Layer-local post-RoPE FP8 KV, actual q/KV/sink/indices/scale passed to sparse kernel, output before inverse RoPE |
| `head.input`, `head.all_logits` | Final contracted and normalized head input and all-token FP32 output |

Module hooks can capture norms, compressor outputs, block pairs, heads and
Engram outputs. Internal scores, shared publications and kernel arguments need
scoped call wrappers or a minimal instrumented source copy. Record an auditable
AST diff containing capture-only additions, and execute unmodified original
methods as the baseline. A capture-only source copy must not replace arithmetic
with native MLite operators. AST-extracted methods with doubles are separately
labeled `dataflow_probe`; they cannot supply quantized model parity.

Layer20 publishes KV once per completed chunk group; layers21–39 consume that
owner's values while producing distinct local SWA values and attention/block
outputs. Full sources are 2,8,14,20; Reindex sources24,28,32,36 read shared index
K and use their own Q/weights; other compressed layers reuse selection. Validate
all consumers, not just layer21. Empty/incomplete prefixes have explicit empty
capture payloads and counts; absence is not an acceptable substitute.

## Precision and independent expected fixtures

Published forward preserves original operator dtypes: BF16 residuals, FP32
mHC coefficients/reductions and head, ratio2 FP32 projections/pooling before
cast/norm, ratio1 BF16 projection/norm, dynamic Linear FP8, local post-RoPE FP8,
main FP4 group16/E4M3, and index FP4 group32/E8M0. Do not disable quantization
to fit CPU execution or conflate model weight dtype with cache dtype. Record
actual dtype at each stage. Raw payload/code/scale values, shape, token positions,
mask membership and alias ownership require exact agreement. Floating results
are not universally bitwise across kernels/devices.

The following hand-derived vectors define wrapper-level expectations independent
of MLite. B3 supplies complete model dimensions/weights and ranking fixtures;
these vectors do not replace its 40-layer model fixture.
Machine-readable values are in `deepseek_v41_oracle_vectors.json`; they are
expected data, not an oracle implementation or a replacement for B3 fixtures.

| Fixture | Input and independent expectation | Acceptance / negative control |
|---|---|---|
| All-token head | Normalized input rows `(1,2),(3,5),(7,11)`; head rows `(1,0),(0,1),(2,-1)`; logits `[(1,2,0),(3,5,1),(7,11,3)]` | FP32 exact (small integer products); default head returns only `(7,11,3)`. Drop middle token, swap head rows, omit weight load or transpose token axis must fail |
| Shifted HC/CED probe | HC copies `(1,2),(5,10)`; incoming p=(1/4,3/4), attention p=(3/4,1/4), FFN p=(1/2,1/2). Probe attn norm adds13, compressor multiplies3 then adds5, index projection multiplies5 then adds11, probe RoPE adds7, quantizer multiplies2 | Collapsed=(4,8), x20=(17,21), latent20=(56,68), index_k20=(291,351), published index=(596,716), main=(126,150); FFN collapse=(2,4) if probe hc_post returns residual; returned p=(1/2,1/2). All values exact in FP32. These are labeled doubles, not RMS/RoPE/FP4 mathematics |
| Publication lifetime | Capture latent `(56,68)`, then mutate source storage through probe RoPE/quantizer | Saved latent remains `(56,68)`; published main=(126,150). A view-only recorder fails |
| Ratio2 visibility | Query positions0,1,2,3,4 | Completed counts0,1,1,2,2; key positions0,2. Incomplete group at4 invisible; use actual offset in returned int32 indices |
| Initial HC | h=(2,7), H=4 | Four identical copies, initial p=(1,0,0,0), collapse=(2,7); initialization must not sum all four |
| Packed mapping | Two independent sequences of lengths3,2 | Flattened offsets `[0,3,5]`, positions `[0,1,2,0,1]`; perturb sequence A and require unchanged B logits/hash/masks under fresh state |

For exact sentinel fixtures use atol=rtol=0. Existing A3 FP32 rotary probes use
atol=rtol=1e-6 at actual boundary positions; retain that local test's scope.
For instrumented versus uninstrumented original execution on the same pinned
device/backend, require exact equality when deterministic; if a kernel is
nondeterministic, measure repeated uninstrumented noise and obtain a reviewed
per-stage tolerance before accepting that case. Do not silently loosen checks.
Record max absolute error, relative error with stated denominator floor, finite
count and first mismatch coordinate per stage. Final-slice versus full-head GEMM
may choose different kernels, so its GPU tolerance also requires this qualification.
These deferred GPU thresholds belong to B1-I/G1-S qualification; no cross-kernel
BF16/FP4 full-model numeric acceptance is claimed by this S specification.

Ranking fixtures require selected/unselected margins greater than twice the
measured score error bound. Test ties separately: official `torch.topk` does
not establish a portable stable tie order. Record the pinned backend's selected
set and sorted positional output; portable tie tests check valid tied choices,
cardinality, masks and -1 semantics rather than inventing lowest-index priority.
Compare candidate membership independently before comparing consumer attention.
Use a later-query score reversal whose winner is outside the published pool;
positive score rescaling is an equivalence control, not a negative mutation.

## Decision activation and execution gates

Profile `published-forward-only`: active decision IDs = `[]`.
No optimizer, objective, quantization derivative or diagnostic branch executes.
O01–O07 concern optimizer groups/schedules; O08–O11 table training state;
O12 indexer training; O13 modality updates; O14 backward; O15 FP4-off diagnostics;
O16 objectives; O17 optimizer skip. All are inactive for this profile, not
implicitly resolved. Preserve current resolved policy; in particular O08/O09/O12
remain OPEN. B1 does not decide them. Expanding the request to training or FP4-off
must be rejected by this interface and handled by its named downstream contract.
Run the plan validator with no `--active-decision` for this empty active set;
any later profile must enumerate every active ID and pass each explicitly.

B1-I must execute `tests/unit/deepseek_v41/test_oracle.py` with B3-I fixtures:

1. Compare original execution and captures, all-token outputs and binding coverage;
   replay seeds from fresh state and compare prefill plus single-token continuation.
2. Run corrected A7 CED probes and verify all eleven distinct mutations are present
   and rejected. Add head/token omission, view-aliasing and stale-sequence-state
   mutations, and full Reindex K/Q/candidate execution (not only its K prefix).
3. Cover 40 layers, three modes, both Engrams, multiple images, span/copy mapping,
   nonaligned compression/SWA boundaries and sequence isolation. Reduced weights
   cannot be reported as all-release-payload load or full-size parity.
4. Run actual official quantized kernels through Slurm for GPU cases, with job ID,
   exit status and non-skip stage counts in internal evidence. Missing dependencies
   remain a failed gate. Require reviewed numerical thresholds for those cases.

G1 retains published dimensions, actual complete shards, long-context boundaries
and full-size forward comparison. B2 owns independent differentiable equations;
official inference-mode capture proves neither gradients nor optimizer updates.

## Specification verification record

The following checks were executed against the pinned local bytes on 2026-09-11:

| Check | Observed result |
|---|---|
| Corrected A7 `validate_deepseek_v41_ced.py --official-model <model.py>` | Exit0, eleven named dataflow mutations rejected without a hash gate |
| A8 `validate_deepseek_v41_plan.py` with empty active set | Exit0, `PLAN_STRUCTURE_OK`, 32 S/I pairs, 64-node graph, 17 decisions and nine rejected mutations |
| AST-extracted official `ParallelHead.forward` and `Block.hc_pre` on the vectors above | Exit0, exact FP32 head and collapsed HC values; token omission, head-row swap and zero weight rejected; clone lifetime and arithmetic CED values checked |
| `git diff --check` | Exit0 |

To reproduce the vector check without importing GPU dependencies, AST-extract
only the named two official methods with `torch`, `torch.nn.functional as F`
and `world_size=1` in their namespace. Call the head with a weight-bearing object
and `full_logits=True` using the JSON head input/weights; compare with
`torch.equal`. Repeat without the flag against `last_logits`. Call `hc_pre`
with the JSON residual and incoming/attention mixes, then apply only the stated
probe arithmetic to the collapsed value. Check each JSON expected intermediate
before and after in-place mutation, preserving a clone of the latent. The
hand-derived expectations must remain fixed when mutating calls or weights.

No official full Transformer, full quantized kernel execution, real checkpoint
payload load, processor execution, GPU job or backward was run for this S gate.
The table records scoped specification evidence; review and B1-I execution
remain separate acceptance steps.

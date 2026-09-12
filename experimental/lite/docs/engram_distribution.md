# Resident Engram row distribution

These components implement the Engram table/provider boundary. They do not
include attention, hashing or backbone assembly. Sinkhorn state integration is
provided separately from the full model/protocol integration.
Megatron Core remains an environment dependency.

- `EngramLayout` maps existing global ranks as `[replica][row shard]`; it does not
  add a parallel axis. Each replica covers all rows exactly once. Corresponding
  columns form replica groups; optimizer intervals partition each shard among
  replicas. Alias chains resolve to one owner. Empty intervals are supported.
- `RowLookup` routes IDs by all-to-all, transfers FP8 values and E8M0 scales as
  bytes, and restores original order/duplicates. Backward coalesces FP32 gradients
  at each owner. All group members participate, including empty requesters.
- `ShardedEngramTable` exposes `forward(ids)` and `lookup_fp8(ids)` for model
  composition. `trainable=False` retains only published bytes; `True` adds a
  persistent FP32 master (a port choice, not an official training recipe).
  No table is offloaded to CPU. Distributed lookup requires CUDA storage.
- `load_engram_table` in the model checkpoint module consumes a validated manifest with `entries[name]` containing
  release key, dtype, shape, byte length, source shard, offset and SHA256 digest.
  Only local rows are allocated; host staging is bounded by `chunk_rows`.
  Whole-tensor digests require a full streaming scan, not full materialization.
  Gaps, overlaps, truncated payloads and digest mismatches are rejected.
- `EngramFP8Projection.forward_lookup(provider, ids)` passes published values and
  scales to native FP8 multiplication without activation requantization. The
  24 rows × 256 values flatten to 6144 projection inputs. This blockwise numerical
  implementation is not a throughput-qualified fused kernel.

`EngramPrefetch(state).start(step, {microbatch_id: ids})` gathers the entire local
batch once. Pass `batch.view(id)` as the embedding provider for that microbatch;
keep the registered source table as the parameter owner. After backbone backward,
call `batch.flush()` once. Detached FP32 leaves buffer gradients tagged by step,
publication version and microbatch. Missing, duplicate and stale returns fail;
flush sends the combined gradient through the original lookup. CUDA ready/return
events order streams; CPU follows the same lifecycle synchronously.

`EngramTableState.step(update_rule, lr=..., beta=.95)` prepares full local-shard
M=.95M+.05G and N=.95M+.05G, including unvisited rows with historical momentum.
The supplied rule transforms current N without repeating momentum. This explicit
rule interface remains available for component diagnostics. Training uses
`EngramSinkhornState` below.
Tests use an identity rule solely to isolate state semantics. Publication changes
only after all candidate tensors exist; skips preserve master/momentum/bytes and
version. Failed quantization can retry. Quiescent snapshots preserve the next
step trajectory. Frozen snapshots have no master or momentum.

Model/protocol assembly, mixed-algorithm clip/skip coordination, CUDA overlap
performance and full-size memory/performance remain integration work. While managed prefetch is active, callers must not separately mutate or
refresh its source table. Budget master, gradient, momentum, cached leaves/returns
and candidate workspace in addition to FP8 storage.

## Sinkhorn logical statistics and table publication

The mathematical contract is A4 `c4c27b0e6` and B2-S `f725fe983`, Algorithm 1:
M=.95M+.05G; N=.95M+.05G; mask rho <= .001 mean(rho); start U from current N;
11 alternating L2 divisions (row first/last, epsilon 1e-20 outside sqrt);
direction=sqrt(logical columns)*U; W-=.18*base_lr*multiplier*direction. No decay,
normalization cache or warm-start is persisted. Engram's multiplier is 5.
O08/O09's approved port policy permits frozen publication-only or trainable FP32
master plus scale regeneration. The FP32 master is a port representation, not an
official storage recipe. Indexers remain frozen and must not enter this optimizer.

`primitive.optimizers.sinkhorn.Sinkhorn` operates on resident FP32 matrix masters.
A native FP32 `main_grad` takes precedence over `.grad`; BF16 inputs are rejected.
Replica contributions are SUM-reduced; the caller owns token/loss normalization.
Replica reduce-scatter pads unequal row intervals transiently, strips padding
before statistics, and stores momentum only for the local optimizer-owned rows.

The caller creates a rectangular grid of process groups over **disjoint optimizer
state pieces**, using the existing ranks:

- `row_group`: different logical row intervals with identical feature columns.
  Include the row intervals split between replicas, each exactly once.
- `column_group`: different feature intervals with identical row intervals.
- `replica_group`: copies of the same physical parameter shard. Its ascending
  process-group rank order determines the balanced contiguous row-state slices.
- `None` always means local, never implicit WORLD. Zero-sized local intervals
  participate in the collective sequence. The logical matrix must be nonempty.

For two row shards replicated twice on ranks `[[0,1],[2,3]]`, lookup groups are
`[0,1]`/`[2,3]`, replica groups `[0,2]`/`[1,3]`, and Sinkhorn's row group is all four
ranks. The four momentum pieces cover the logical matrix once; statistics never
count duplicate parameter replicas or padding. Only norm vectors cross row/column
groups. Candidate weights gather only within the replica group, not across the
full table. Parameter masters remain resident for lookup/STE.

`EngramSinkhornState(table, row_group=..., column_group=..., replica_group=...)`
uses the existing `EngramPrefetch` lifecycle. After `flush()`, `state.step(lr=...)`
stages W/M, regenerates row/block32 FP8/E8M0 publication, agrees on failure, then
commits W/M and the value/scale pair. Nonfinite gradients or explicit skip preserve
all weights, momentum, publication and version. Quantization exceptions discard
candidates and retain the active gradient for retry. Frozen tables allocate no
optimizer/master/momentum and retain their published bytes.

Optimizer checkpoints retain FP32 momentum and the exact layout/rank map; table
checkpoints additionally retain master, bytes, publication version and last step.
Restoring under a different layout is rejected until an explicit resharder is
provided. No prepared update or active prefetch may be checkpointed. This does
not certify E5's full text/vision optimizer, RNG, scheduler or phase restoration.

Tests: `tests/unit/primitive/optimizers/test_sinkhorn.py` uses independent Python
float64 scalar arithmetic with fixed Algorithm 1 constants. Local W/direction
bounds are atol=rtol=2e-6. `test_sinkhorn_shards.py` uses four-rank NCCL, actual
backward, distinct replica contributions and multi-step W/M plus save/load;
W bounds are atol=rtol=3e-6, momentum atol=2e-5/rtol=3e-6. These are explicit test
thresholds for the reduced fixtures, not a full-scale or BF16 tolerance approval.

## Paired pipeline components and model integration boundary

Consume corrected A1 `2b2c7e0c3` and B2-S `f725fe983`; O12 supersedes the older
A1 auxiliary-objective OPEN rows: indexer parameters are frozen, with no indexer
loss or optimizer state. Integer Top-K dependencies do not create gradients.

The boundary after Block(19) is the pair `(h20,p20)`, not only h20. Source 20
must run `attn_norm_20(hc_pre(h20,p20))`, then its compressor. Index-K branches
from the pre-main-RoPE latent. Decoder layers 21–39 consume the source-20 KV;
reindex layers 24/28/32/36 replace selection, not the floating KV owner.

A C4 pipeline adapter needs a per-microbatch record with these distinct fields:

| Field | Ownership / lifetime | Backward |
| --- | --- | --- |
| Current HC residual and shifted mixing coefficients | Paired output of the preceding block; preserve shape/dtype separately | Return both cotangents |
| CED h20/p20 | Same generation and microbatch; retain through all source-20 consumers/recompute | Sum all floating consumer paths into the pair |
| Published main KV, quantization bytes/scales and FP32 STE carrier | One canonical owner per A1 source; no implicit re-encoding at a PP boundary | Return consumer VJPs to the floating owner once |
| Index-K/Top-K and shadow indexer identity | Frozen dependency, matching publication generation and CP global positions | No parameter/indexer-loss gradient |
| Step, microbatch, virtual chunk, generation, source layer, CP token range | Exact integer metadata; validated before a read | A return must match its outstanding forward record |

`model.deepseek_v41.lite.pipeline.PairedPayload` carries these tensor fields,
while `PipelineTag` identifies step, microbatch, virtual chunk, generation and
source owners. `PipelineLedger` retains each graph until its exact consumer set
returns all cotangents, then invokes backward once. `finish_step` retires stale
generations and bounds bookkeeping after the scheduler drains a step.

`primitive.parallel.tensor_payload` sends integer shape/dtype/generation headers
and exact tensor bytes over P2P. The receiver acknowledges generation/schema
acceptance before data transfer. FP8/E8M0 publication and int64 positions therefore
retain their representation; floating tensors become independent autograd leaves
whose cotangents must be returned explicitly. Both peers must follow a matching
schedule. This component does not register parameters or schedule microbatches.

`test_deepseek_v41_pipeline.py` exercises this transport on four NCCL ranks with
two interleaved microbatches, reverse backward order, nonreentrant recomputation,
three distinct consumer vectors and an independently derived owner update.
It also checks FP8/E8M0 values, integers above float32's exact range, rejected
wire generations and ledger lifetime errors. This is a transport/graph component
fixture, not the actual source-20 attention/compressor or a full PP model run.

The common pipeline scheduler currently accepts one rank-3 tensor and casts it
to its configured pipeline dtype. It cannot silently carry this mixed-dtype
record: tensor fields require explicit dtype/shape-preserving transport, and
integer generations must not be packed into floating hidden activations. C4's
`set_input_tensor`/forward result contract must expose the paired state before
extending the common scheduler. Do not substitute a standalone cache test for
that production model path.

Lifecycle: allocate a record for each scheduled microbatch; publish each source
once; read only the matching generation; retain through recompute; track the
expected consumer VJP set; add each vector exactly once; release only after all
returns complete. Interleaved microbatches cannot overwrite a stage-global
cache. Skipped/retried steps invalidate pending publications before rebuilding
prefetch. Optimizer registration resolves aliases to the canonical owner only.

Discriminating fixtures for the eventual C4-backed CPU/Slurm tests:

- B2 T2: h=[2,5], p=[1/4,3/4], loss=2*(p dot h) yields input contraction 17/4,
  grad(h)=[1/2,3/2], grad(p)=[4,10]. Dropping p20 or using current coefficients
  changes the result. Extend through actual source-20 normalization/compression.
- Three remote consumers contribute [1,2], [3,-5], and [-2,7] to one floating
  owner: total [2,4]. Identity-scaled consumer vectors would conceal some
  omission/duplication errors and are not a sufficient fixture.
- Use distinct residuals, coefficients and generation IDs for two interleaved
  microbatches; repeat with nonreentrant recompute and reverse backward order.
  Wrong generation, duplicate return, missing return and post-release read fail.
- Assert that source 20 updates once after all consumers, while every frozen
  shadow/indexer stays byte-identical and absent from optimizer state.

Full PP/TP/EP/CP clipping, mixed-optimizer atomic publication and E5 text/vision
restart require the actual C4 model plus F2 routing/F3 visual training contracts.
The standalone Engram/Sinkhorn tests above do not certify those integrations.

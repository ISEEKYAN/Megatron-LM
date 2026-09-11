# Engram row ownership and lookup

This implements the row lookup/provider boundary of the post-training design.
Source contracts: `../deepseek_v41_owner_consumer.md`,
`../contracts/deepseek_v41/weights.json`, and `deepseek_v41_fixtures.md`.
The hash provider supplies global row IDs in ngram-major/head-minor order.
Three ngrams and eight heads yield 24 rows, each 256 values: the flattened
provider output has 6144 features for the existing Engram `wkv` projection.

## Placement contract

`EngramLayout(rows, rank_groups, world_size=...)` accepts a rectangular map of
**existing global ranks**: `[replica][row shard]`. This is an explicit placement
choice over the dense TP/CP/PP/DP and expert ETP/EP/PP/DP decompositions in
`primitive/parallel/state.py`, not a new factor of world size. A stage/subset may
participate; nonparticipants still join process-group creation in the same order.
Actual stage placement and the rank map must be supplied by model assembly.
No hidden inference from TP or EP rank numbers is made.

Example, 7 rows on existing ranks `((0,2,4),(1,3,5))`:

| Row interval | Lookup replica 0 | Lookup replica 1 | Optimizer subintervals |
|---|---|---|---|
| [0,3) | 0 | 1 | rank 0: [0,2), rank 1: [2,3) |
| [3,5) | 2 | 3 | rank 2: [3,4), rank 3: [4,5) |
| [5,7) | 4 | 5 | rank 4: [5,6), rank 5: [6,7) |

Each lookup replica covers every logical row exactly once. Corresponding columns
are replica groups, not duplicate rows in logical Sinkhorn statistics. The first
replica identifies canonical ownership; optimizer subintervals partition the
owned interval across replicas, including empty intervals. This describes state
placement; actual replica gradient reduction and optimizer sharding are separate
step operations. PP aliases/shadows resolve transitively to one canonical name;
cycles are rejected. Indexers are frozen: no indexer objective, loss, or optimizer
contribution is introduced.

## Collective ABI and lifetime

`RowLookup(boundaries, group)` requires identical nondecreasing boundaries in
process-group rank order, starting at zero. Empty row shards are legal.
`group=None` explicitly selects a one-rank local reference path. Distributed
execution requires CUDA-resident IDs, table values, scales and optional master.
`EngramLayout.create_groups` requires ascending global ranks within lookup groups
to match PyTorch group ordering. All WORLD ranks call it in identical order.

`raw_rows(values, scales, ids)` returns tensors shaped `[*ids.shape, width]`.
Values and scales travel as uint8 without decode/requantization. Requests are
stably grouped by row owner, exchanged by all-to-all, indexed at the owner, and
returned by reverse all-to-all. The original request order and duplicates are
restored. Only per-rank split counts are copied to host; no embedding payload or
table is offloaded. Invalid IDs are collectively rejected before routing.

`ShardedEngramTable` plugs directly into `Engram(..., embedding=provider, ...)`.
The same `trainable` switch selects frozen FP8/E8M0 storage or persistent FP32
master plus straight-through gathered rows. FP32 master representation is a port
choice, not an official training recipe. Existing `refresh_storage` regenerates
scales after an accepted update. The custom backward returns each requester’s
FP32 gradient to its row owner and coalesces duplicate rows. The route is owned by
the autograd context, so multiple forward calls do not overwrite its IDs/order.
All group ranks execute forward/backward in matching order, even empty requesters;
conditionally omitting a rank's backward is unsupported. Prefetch scheduling,
replica reductions, accepted-step publication and checkpoint integration remain
separate integration work, not claims of this lookup implementation.

## Discriminating verification

`test_ownership.py` uses hand-enumerated rank/interval expectations, including
empty shards and aliases. `test_lookup_local.py` compares byte indexing against
an unsharded published table and checks weighted duplicate-row gradients.
Slurm `tests/distributed/deepseek_v41/test_lookup.py` uses four NCCL ranks in two
noncontiguous lookup groups, two replicas, uneven/empty shards and empty
requesters. It compares raw values/scales bitwise and FP32 gradients against
independently accumulated request vectors. Missing Slurm/four GPUs fails rather
than skips. These reduced fixtures do not establish full-size memory, performance,
or combined model TP/EP/CP/PP acceptance.

## Streaming load and native projection

`load_engram_rows` validates a complete disjoint interval partition, then allocates
only the local value/scale tensors on the requested device. `iter_rows` limits
host staging to `chunk_rows` rows. The immutable release entry identifies the
key, dtype, shape, file offset and whole-tensor digest. Since that manifest has
no per-row hashes, integrity verification scans the entire source tensor using
bounded memory; this is not a claim of partition-proportional disk traffic.
Both iterators must finish successfully before the factory publishes the table.
`ShardedEngramTable.from_checkpoint` binds the loaded interval to the same lookup
boundaries; a distributed provider rejects CPU destination storage.

`EngramFP8Projection` obtains `lookup_fp8` directly from the table provider and
flattens 24 rows and their 24×8 E8M0 scales in the same order. The native
`published_fp8_linear` passes these bytes to the existing `_fp8_gemm`, quantizing
only the floating projection weight. There is no activation re-quantization or
BF16 decode before GEMM. Backward decodes published operands for FP32 derivative
arithmetic and routes the identity-STE contribution into the optional master.
This blockwise correctness kernel is not a performance-qualified fused kernel.
`test_fp8_lookup.py` observes the actual kernel operands, forbids activation
requantization, and checks native output/dW/table gradients against independent
exactly representable inputs with atol=1e-3, rtol=1e-5. Performance remains a
separate representative-size gate.

## Batch prefetch and delayed gradient return

`EngramPrefetch(state).start(step, {microbatch_id: ids})` acquires one table
publication and performs one concatenated row lookup before stage microbatches.
Empty microbatches remain explicit entries. Each `batch.view(id)` is a provider
for `Engram.forward(..., embedding=view)` or the FP8 projection. The override does
not replace the registered embedding parameter, so ownership/enumeration remains
unchanged. IDs must exactly match the prefetched microbatch. The provider can be
reused in nonreentrant activation recomputation without another lookup.

Trainable cached rows are detached FP32 leaves. Their hooks retain gradient
returns tagged by `(step, publication_version, microbatch_id)`, rather than
immediately traversing the original lookup autograd graph. `batch.flush()` runs
only after backbone backward, rejects missing/duplicate/stale returns, and calls
backward through that original lookup once with the combined FP32 vector.
Repeated global IDs consequently coalesce at the row owner using the existing
lookup primitive. The result lands in the state-owned native FP32 `main_grad`;
it is not a BF16 parameter gradient widened after accumulation. Each scheduled
microbatch contributes once per backward, including explicit empty microbatches.
Multiple separate backward calls on the same microbatch are rejected as duplicate
returns; combine its consumers into one backward graph.

An optional prefetch CUDA stream waits for the producer stream. Ready events are
waited on before consuming cached IDs/rows; return events are waited on before
flush. Buffers record consumer-stream use. CPU runs the same state machine
synchronously. CPU tests verify ordering through an injected event recorder;
actual CUDA stream overlap and multi-rank scheduling are not certified here.
Frozen views retain only published values/scales and require consumption but no
gradient return. Closing a batch invalidates its views and releases cached rows,
IDs, return vectors, and the original lookup graph.

## Full-row state and publication transaction

`EngramTableState` owns the full local-shard FP32 gradient and momentum matrices.
It prepares `M = beta*M_previous + (1-beta)*G` and
`N = beta*M + (1-beta)*G` over **all** local rows, including unvisited rows.
`step(update_rule, lr=..., beta=.95)` requires an explicit full-matrix direction
function of current N. Tests use the identity function solely as a Nesterov/state
reference. This is not a substitute for Sinkhorn: its logical-matrix collectives,
normalization and LR factors are supplied by the subsequent optimizer work.
There is no persistent normalization cache or warm-start state in this layer.

Only after candidate master, momentum, values and scales have been prepared does
publication advance its version. Skips/nonfinite updates leave all four unchanged
and consume only the attempt's step tag. A failed quantization keeps the ready
gradient available for retry. A step with a live batch cannot update or checkpoint.
Checkpoint/restore is allowed only between attempts and preserves master,
momentum, raw FP8/E8M0 bytes, publication version and the last attempt tag.
Frozen checkpoints contain no master/momentum and never regenerate release
quantization. Parameter identity stays stable across update and restore.

These APIs are the sole publication path while managed prefetch is in use;
callers must not separately mutate/refresh the table during a live attempt.
Replica gradient reduction, optimizer-state sharding and globally agreed atomic
skip remain integration work. Full-size memory budgeting must include the local
master/gradient/momentum, cached row leaves/returns and candidate/publication
workspace, not only FP8 storage. This implementation does not claim a 196B-table
memory or throughput result.

CPU tests exercise eager-vs-prefetched Engram recompute gradients, delayed return,
exact FP32 accumulation of `1 + 2^-10`, wrong/missing/duplicate generation tags,
sub-FP8 repeated updates, unvisited historical momentum, failed-publication retry,
frozen state and restored next-step trajectories. Hand-computed unvisited-row
example: after G=4 at beta=.95 and lr=.1, M=.2 and W=.961; next step G=0 still
produces M=.19, N=.1805 and W=.94295. A visited-row-only update is incorrect.

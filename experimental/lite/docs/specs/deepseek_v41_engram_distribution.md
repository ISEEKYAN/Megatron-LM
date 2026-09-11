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

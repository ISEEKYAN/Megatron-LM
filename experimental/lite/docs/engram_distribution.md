# Resident Engram row distribution

These components implement the Engram table/provider boundary. They do not
include attention, hashing, backbone assembly or a Sinkhorn optimizer.
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
The supplied rule transforms current N; it must eventually perform the global
logical-matrix Sinkhorn statistics and normalization, not repeat momentum.
Tests use an identity rule solely to isolate state semantics. Publication changes
only after all candidate tensors exist; skips preserve master/momentum/bytes and
version. Failed quantization can retry. Quiescent snapshots preserve the next
step trajectory. Frozen snapshots have no master or momentum.

Model/protocol assembly, replica reductions, globally coordinated skip, optimizer
state sharding, CUDA overlap and full-size memory/performance remain integration
work. While managed prefetch is active, callers must not separately mutate or
refresh its source table. Budget master, gradient, momentum, cached leaves/returns
and candidate workspace in addition to FP8 storage.

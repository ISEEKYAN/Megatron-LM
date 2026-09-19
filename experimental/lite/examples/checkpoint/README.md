# Bounded row export

Run `python examples/checkpoint/row_stream.py --output /tmp/row-checkpoint` from
`experimental/lite`. This example streams a trainable FP32 table to FP8 storage,
loads a receiver's local row span, and writes ordinary HF safetensors.

`stream_rows` yields `RowChunk(name, offset, total_rows, weight, scale)` records.
The offset and total are global **rows**, not elements or bytes. Weight and
scale refer to the same rows. The optional scale key is the weight key's
`.weight` suffix replaced with `.scale`. Scale columns are per-row block scales.
Chunks are ordered by owner rank, then increasing global offset. Empty owners
issue no payload collective. Every group member must supply identical ownership
boundaries, plane shapes/dtypes, quantization mode and buffer limit, and drain
the iterator in full. Group ranks are translated to global broadcast sources.

Both planes occupy one byte buffer and use one broadcast per block. FP8/E8M0
are transported as raw bytes; no numeric conversion of scale bytes occurs.
Chunk tensors are borrowed and valid only until advancing the iterator. Copy
into resident destination storage immediately; do not `list()` the stream or
retain chunk views. An ordinary `(name, full_tensor)` HF loader cannot consume
these records. It needs an explicit adapter such as `RowReceiver`, which copies
the intersection of each chunk with its local global-row span and validates
complete, ordered coverage. A hash-head owner uses the cumulative bucket sizes
as its span's start and length. Receiver scale storage may use raw `uint8`.

`buffer_max_size_bytes` limits temporary tensor storage in the row producer,
excluding the already resident source and receiver's final destination. The
byte scratch buffer is rounded down to 512-byte allocation alignment; a limit
that cannot hold one row is rejected. `quantize=True` emits E4M3 plus E8M0 using
independent (1,32) blocks and reserves sixteen FP32 planes per row plus 8 KiB
for allocation rounding. It never quantizes the complete table first. Actual
live-allocation and CUDA peak tests enforce the limit, including quantization.
This guarantee concerns allocated tensor memory, not allocator-reserved memory,
NCCL's external workspace, file cache, or arbitrary consumer-retained copies.

`stream_export_to_shards` accepts these records alongside normal full-tensor
pairs. For each row table it writes a safetensors header with full shapes, then
seeks directly to each plane's byte offsets. It never constructs a full table
on the GPU or in host tensors. A large table is a single file even when larger
than the ordinary shard target: that target bounds buffering, not a single
HF tensor's on-disk size. GPU chunks incur at most one plane-sized CPU transfer
at a time while writing. This host staging is separate from GPU scratch; pass
half the total staging budget to the producer if budgeting their sum.

The wire format changes the old two whole-table gathers into bounded paired
broadcasts. Tensor bytes and global row order are preserved; collective counts
necessarily change. Tests require every rank to execute the same block sequence.

The two-rank Gloo writer test feeds multiple row tables and ordinary tensors
through the shared writer, checks the actual broadcast trace on both ranks,
and reads every W/S tensor using the independent safetensors reader. Rank zero
advances inside the table writer; peers advance in the outer drain loop. Each
iteration on either rank requests exactly the next producer block.

Bound checkpoint integration opts into this protocol explicitly:

```python
# The model's checkpoint policy supplies binding, row ownership and codec metadata.
save_bound_model(model, output, spec, buffer_max_size_bytes=256 * 1024)
# Direct transport consumers must understand RowChunk and drain every rank.
for record in export_checkpoint(model, spec, row_chunks=True,
                                buffer_max_size_bytes=256 * 1024):
    consume(record)
```

The bound saver streams both frozen storage and trainable row quantization;
its separate `mlite_masters` files stream exact FP32 resume rows. It reserves
one quarter of the buffer for row production and the remainder for host staging
and the ordinary shard. The budget bounds row staging, not the resident model,
ordinary non-row codecs, archival tensors, or arbitrary consumer copies. Large
or distributed row tables reject the legacy full-tensor iterator with
`ROW_STREAM_REQUIRED` before emitting tensors; small unsharded calls remain
compatible. A rollout loader still needs the explicit row-offset adapter.

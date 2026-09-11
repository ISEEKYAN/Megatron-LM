# V4.1 payload implementation contract

This implementation consumes the unchanged A2 contract at `b3a916425` in
`docs/contracts/deepseek_v41/`. The original release config remains archival
metadata. Storage does not allocate a model or select a training precision.
Engram scale regeneration and master weights remain unresolved and unimplemented.

`CheckpointTensorStore.load(paths, expected_keys)` validates exact key coverage,
header dtype, shape and byte ranges before recording immutable file-backed entries.
Each entry records source shard, byte length and SHA-256. Reads and writes recheck
the digest; a changed backing file is an error. `save(path)` streams unchanged
payloads into a new safetensors file. `shard(rank, world_size)` assigns sorted
global keys by ordinal modulo world size; `merge(stores, expected_keys)` rejects
duplicate, missing and extra keys, including empty-rank cases.

`validate_execution(enable_dspark_execution=False)` accepts storage-only use;
true raises `NotImplementedError`. This guard is not yet a backbone model factory.
No storage object is an `nn.Module` or `nn.Parameter`. The caller retains the
original `dspark_block_size=5` and passes the separate runtime flag.

Independent byte cards: I8 payload `00 7f 80 ff`, shape `[2,2]`; BF16 payload
`80 3f 00 c0`, shape `[2]`; E8M0 payload `7e 7f 80 ff`, shape `[4]`.
Roundtrip preserves all bytes, including nonfinite scale codes, without interpreting
them. Numerical decode must validate its own scale domain separately.

Execution increments (each a focused 2–5 minute step):

1. Add `tests/unit/deepseek_v41/test_mtp_store.py`, run
   `PYTHONPATH=experimental/lite python -m pytest -c /dev/null --confcutdir=experimental/lite --rootdir=experimental/lite -q experimental/lite/tests/unit/deepseek_v41/test_mtp_store.py`;
   require failure for the missing implementation.
2. Implement `megatron/lite/model/deepseek_v41/lite/checkpoint_store.py`; rerun
   the same test for raw cards, malformed headers, modified backing files,
   duplicate/missing keys, and storage repartition.
3. Expand the A2 MTP names with synthetic payloads and verify exact 2,401-key
   coverage through repartition. This is a synthetic namespace test only.
4. Run the same streaming path on pinned real MTP shards, comparing
   dtype/shape/bytes/digests. The release run preserved all 2,401 keys and
   7,932,874,632 payload bytes through seven storage ranks; source digests are
   recorded in `deepseek_v41_mtp_provenance.json`.
   Entry point: `PYTHONPATH=experimental/lite python experimental/lite/tools/deepseek_v41/validate_mtp_store.py --checkpoint <release-directory> --output <new-directory> --storage-ranks 7`.

The later C1 numerical decoder and B1 official GPU oracle remain separate gates.
CPU archival tests cannot certify official quantized execution or full-model parity.

## Numerical decoding interface

`load_weight(store, name, output_dtype=torch.bfloat16)` reads a digest-checked
weight and its exact sibling `scale`. I8 weights use the existing MXFP4 decoder
(low nibble first, E2M1, 32-element groups, E8M0 scales). E4M3 matrices use the
existing block-FP8 decoder with 32x32 scales; Engram `embed.weight` uses explicit
row-by-32 scale layout. BF16/F16/F32 exports pass through without a scale;
an unexpected sibling scale fails. The result is a detached numerical binding,
not an executable Linear or a training master-weight policy.

Independent code card: bytes `10 32 54 76 98 ba dc fe`, repeated twice, decode
to `[0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6]` repeated twice at scale 1.
At scale 2 every value doubles; two rows dotted with all-ones produce zero.
For FP8 input ones `[64,64]` with scales `[[.5,1],[2,4]]`, an all-ones vector
produces first 32 outputs 48 and last 32 outputs 192. These CPU cards are exact;
official kernel comparison and loaded GPU GEMM remain required separately.

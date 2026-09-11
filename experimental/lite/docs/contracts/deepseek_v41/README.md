# V4.1 checkpoint mapping contract

This is a Phase A specification, not an implemented model loader. `weights.json`
expands to the complete published key set; `config.json` enumerates every config
leaf and its planned production consumer. No keys are excluded. A `planned`
consumer does not claim that the current DS4 model supports V4.1.

Source: DeepSeek-V4.1-Flash revision
`df42c109f1defefcbfcedbe7d905718a12266e40`, `config.json`,
`model.safetensors.index.json`, and `inference/model.py` (classes Block,
DSparkBlock, Transformer). The key-set digest is independent of shard filenames.
The source index has 96,085 keys, including 2,401 under `mtp.`.

## Canonical storage and binding

Each family lists exact index domains, count and an injective destination.
Expand the Cartesian product of `indices` into both `pattern` and `target`.
`checkpoint_store.<release key>` is a logical tensor-store address, not a Python
parameter name. Retain the release namespace, global layer IDs and expert IDs;
never fold `mtp.N` into `layers.40+N`. No prefix fallback or ignored-key bucket
is permitted. Unknown, missing, duplicate source or duplicate destination keys
are errors. Shared KV/indexer consumer layers do not create additional copies
in the canonical checkpoint. PP-local indices must resolve to global owners.

The loader must read each safetensors header before allocation and retain
`{release_key, dtype, shape, byte_length, source_shard, payload_digest}`. The
index contains no shapes/dtypes: this specification does **not** infer them
from names or claim header validation. A header mismatch or absent tensor is
fatal. Bind backbone, vision and aligner entries to their Phase D/F modules;
`checkpoint_store` alone is not evidence of executable model support.

Quantized `weight` and `scale` are separate covered entries and a coupled binding
unit. C1 must reuse the verified dequantization reference, supporting FP4 packed
experts and FP8 32x32 blocks with UE8M0 scales. Raw-byte archival must never cast,
dequantize, fuse w1/w3, transpose, pad or rewrite scale names. Numerical bindings
may transform them only with a documented inverse/export policy. Raw release
round-trip, dequantization correctness and training export are separate gates.
Training export must serialize updated backbone tensors rather than stale store
bytes; inactive MTP tensors remain original bytes.

## MTP carrier and sharding

C3 shall implement a standalone `CheckpointTensorStore`, outside `nn.Module`
and outside the model parameter tree. Its entries are immutable CPU/file-backed
byte ranges with header metadata, not `nn.Parameter` objects. Load/save/shard
must work without importing or constructing DSparkBlock or a full model.

For `P` archival ranks, sort all 2,401 MTP keys lexicographically. Rank `r` owns
keys whose ordinal modulo `P` equals `r`. Each tensor (including packed bytes)
is indivisible; this is storage sharding, independent of execution TP/EP/PP.
Each key has exactly one owner. Distributed save gathers manifests, checks the
exact union and uniqueness, and streams payloads from owners. Repartitioning
recomputes ownership from global names; source safetensors file boundaries do
not constrain placement. Empty ranks are valid. Weight/scale pairs can have
different storage owners; numerical consumers fetch both before decoding.

MTP is excluded from optimizer groups, gradient allocation/reduction and
trainable-parameter coverage. It remains mandatory in whole-checkpoint coverage.
Changing backbone weights must not mutate its bytes. Restore validates every
MTP dtype, shape and payload digest, then exports under the original names.

Official DSparkBlock initializes main_proj/main_norm only for stage 0 and
norm/markov_head/confidence_head only for the final stage (2). The manifest
preserves these domains; a uniform three-block expansion is incorrect.

## Execution switch

Keep the original release config, including `text_config.dspark_block_size=5`,
unchanged for export. Add a separate runtime `enable_dspark_execution=False`.
The model factory must reject True with `NotImplementedError` before allocating
DSpark modules. A direct DSparkBlock construction must also raise. The normal
backbone factory never constructs DSparkBlock. Do not test the release block
size as an execution switch or rewrite it to zero. Metadata preservation alone
does not implement DSpark computation; its config consumers have a scope waiver.

## Acceptance boundaries

Run the CPU contract validator against the pinned local source files:

```sh
python experimental/lite/docs/contracts/deepseek_v41/validate.py \
  --index /tmp/v41_index.json --config /tmp/ds41-review/config.json
```

It proves exact key/config coverage, bijection and archival ownership, including
negative controls. It does not prove real tensor load/save, payload bit equality,
GPU execution or training precision. C3 must additionally stream all 2,401 real
MTP payloads through load/save/reload (including repartition), compare dtype,
shape and bytes, and test execution rejection and optimizer exclusion. C1 must
independently validate dequantization; downstream runtime tests must demonstrate
that every planned config consumer is connected, not merely JSON-preserved.

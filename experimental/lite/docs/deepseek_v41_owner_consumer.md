# DeepSeek-V4.1-Flash: CSA2 owner/consumer contract and CED gate

This specifies the released boundary, not a completed MLite implementation.  The official
`DeepSeek-V4.1-Flash/main` snapshot downloaded 2026-09-10 has SHA-256
`model.py=4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`,
`config.json=8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`,
and `model.safetensors.index.json=74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.

## CED boundary in the released execution path

The previous claim that official inference does not implement CED was incorrect.
Absence of the strings `CED` or `H_{L/2}` says nothing about value provenance.
The ordered loop (model.py:1261–1267) already produces the boundary state after
Block(19). Naming that pair `(h20, p20)` makes existing behavior explicit; it
is not a new training definition and must not introduce a decoder projection.

The directly comparable forward intermediates are:

```text
h20, p20 = Block(19) return values
x20 = attn_norm_20(hc_pre(h20, p20))
latent20 = compressor_norm_20(compressor_wkv_20(x20))  # ratio=1
index_k20 = k_norm_20(wk_20(latent20))                 # before main RoPE
main_kv20 = official main RoPE then FP4 on latent20
# index_k20 separately undergoes index RoPE and FP4 before publication
# decoder 21..39 reads source 20 main_kv20; SWA projects each layer's own x
```

Raw `h20` has shape `[batch, sequence, hc, hidden]`; `p20` has shape
`[batch, sequence, hc]`. Block attention first collapses HC with `p20`, then
normalizes the resulting three-dimensional input (985–987). Neither operation
may be bypassed. Compressor allocation is restricted to KV sources (654–661),
and source 20 publishes the cache reused by decoder consumers (741–763).
Indexer K consumes the pre-main-RoPE latent (527–555).

The full loop does not perform deployment scheduling that skips most decoder
prefill or uses bounded SWA replay. That does not remove its existing shared
encoder-to-decoder KV architecture. Compare forward against the official
intermediates and consumer outputs; backward needs a separate training oracle.

## 40-layer static assignment

From official config: `kv_source_layer_ids=[2,8,14,20]`,
`index_source_layer_ids=[2,8,14,20,24,28,32,36]`.  Source code selects the KV
owner at lines 653--661, readers at 739--765, and Top-K publishers/readers at
721--737.  `K` and `T` are unique KV and Top-K sources. `—` is local SWA.

| layer | side | ratio | mode | K | T | floating KV autograd | integer selection dependency | auxiliary objective contribution |
| ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| 0 | encoder | 0 | SWA | — | — | local SWA | none | no CSA2 indexer |
| 1 | encoder | 0 | SWA | — | — | local SWA | none | no CSA2 indexer |
| 2 | encoder | 2 | Full | 2 | 2 | 2→K2→θKV2; floating sum at owner | read T2; integer, no autograd | OPEN D6-S: contributor 2, shared indexer owner 2 |
| 3 | encoder | 2 | Reuse | 2 | 2 | 3→K2→θKV2; floating sum at owner | read T2; integer, no autograd | OPEN D6-S: contributor 3, shared indexer owner 2 |
| 4 | encoder | 2 | Reuse | 2 | 2 | 4→K2→θKV2; floating sum at owner | read T2; integer, no autograd | OPEN D6-S: contributor 4, shared indexer owner 2 |
| 5 | encoder | 2 | Reuse | 2 | 2 | 5→K2→θKV2; floating sum at owner | read T2; integer, no autograd | OPEN D6-S: contributor 5, shared indexer owner 2 |
| 6 | encoder | 2 | Reuse | 2 | 2 | 6→K2→θKV2; floating sum at owner | read T2; integer, no autograd | OPEN D6-S: contributor 6, shared indexer owner 2 |
| 7 | encoder | 2 | Reuse | 2 | 2 | 7→K2→θKV2; floating sum at owner | read T2; integer, no autograd | OPEN D6-S: contributor 7, shared indexer owner 2 |
| 8 | encoder | 2 | Full | 8 | 8 | 8→K8→θKV8; floating sum at owner | read T8; integer, no autograd | OPEN D6-S: contributor 8, shared indexer owner 8 |
| 9 | encoder | 2 | Reuse | 8 | 8 | 9→K8→θKV8; floating sum at owner | read T8; integer, no autograd | OPEN D6-S: contributor 9, shared indexer owner 8 |
| 10 | encoder | 2 | Reuse | 8 | 8 | 10→K8→θKV8; floating sum at owner | read T8; integer, no autograd | OPEN D6-S: contributor 10, shared indexer owner 8 |
| 11 | encoder | 2 | Reuse | 8 | 8 | 11→K8→θKV8; floating sum at owner | read T8; integer, no autograd | OPEN D6-S: contributor 11, shared indexer owner 8 |
| 12 | encoder | 2 | Reuse | 8 | 8 | 12→K8→θKV8; floating sum at owner | read T8; integer, no autograd | OPEN D6-S: contributor 12, shared indexer owner 8 |
| 13 | encoder | 2 | Reuse | 8 | 8 | 13→K8→θKV8; floating sum at owner | read T8; integer, no autograd | OPEN D6-S: contributor 13, shared indexer owner 8 |
| 14 | encoder | 2 | Full | 14 | 14 | 14→K14→θKV14; floating sum at owner | read T14; integer, no autograd | OPEN D6-S: contributor 14, shared indexer owner 14 |
| 15 | encoder | 2 | Reuse | 14 | 14 | 15→K14→θKV14; floating sum at owner | read T14; integer, no autograd | OPEN D6-S: contributor 15, shared indexer owner 14 |
| 16 | encoder | 2 | Reuse | 14 | 14 | 16→K14→θKV14; floating sum at owner | read T14; integer, no autograd | OPEN D6-S: contributor 16, shared indexer owner 14 |
| 17 | encoder | 2 | Reuse | 14 | 14 | 17→K14→θKV14; floating sum at owner | read T14; integer, no autograd | OPEN D6-S: contributor 17, shared indexer owner 14 |
| 18 | encoder | 2 | Reuse | 14 | 14 | 18→K14→θKV14; floating sum at owner | read T14; integer, no autograd | OPEN D6-S: contributor 18, shared indexer owner 14 |
| 19 | encoder | 2 | Reuse | 14 | 14 | 19→K14→θKV14; floating sum at owner | read T14; integer, no autograd | OPEN D6-S: contributor 19, shared indexer owner 14 |
| 20 | decoder | 1 | Full | 20 | 20 | 20→K20→θKV20; floating sum at owner | read T20; integer, no autograd | OPEN D6-S: contributor 20, shared indexer owner 20 |
| 21 | decoder | 1 | Reuse | 20 | 20 | 21→K20→θKV20; floating sum at owner | read T20; integer, no autograd | OPEN D6-S: contributor 21, shared indexer owner 20 |
| 22 | decoder | 1 | Reuse | 20 | 20 | 22→K20→θKV20; floating sum at owner | read T20; integer, no autograd | OPEN D6-S: contributor 22, shared indexer owner 20 |
| 23 | decoder | 1 | Reuse | 20 | 20 | 23→K20→θKV20; floating sum at owner | read T20; integer, no autograd | OPEN D6-S: contributor 23, shared indexer owner 20 |
| 24 | decoder | 1 | Reindex | 20 | 24 | 24→K20→θKV20; floating sum at owner | read T24; integer, no autograd | OPEN D6-S: contributor 24, shared indexer owner 24 |
| 25 | decoder | 1 | Reuse | 20 | 24 | 25→K20→θKV20; floating sum at owner | read T24; integer, no autograd | OPEN D6-S: contributor 25, shared indexer owner 24 |
| 26 | decoder | 1 | Reuse | 20 | 24 | 26→K20→θKV20; floating sum at owner | read T24; integer, no autograd | OPEN D6-S: contributor 26, shared indexer owner 24 |
| 27 | decoder | 1 | Reuse | 20 | 24 | 27→K20→θKV20; floating sum at owner | read T24; integer, no autograd | OPEN D6-S: contributor 27, shared indexer owner 24 |
| 28 | decoder | 1 | Reindex | 20 | 28 | 28→K20→θKV20; floating sum at owner | read T28; integer, no autograd | OPEN D6-S: contributor 28, shared indexer owner 28 |
| 29 | decoder | 1 | Reuse | 20 | 28 | 29→K20→θKV20; floating sum at owner | read T28; integer, no autograd | OPEN D6-S: contributor 29, shared indexer owner 28 |
| 30 | decoder | 1 | Reuse | 20 | 28 | 30→K20→θKV20; floating sum at owner | read T28; integer, no autograd | OPEN D6-S: contributor 30, shared indexer owner 28 |
| 31 | decoder | 1 | Reuse | 20 | 28 | 31→K20→θKV20; floating sum at owner | read T28; integer, no autograd | OPEN D6-S: contributor 31, shared indexer owner 28 |
| 32 | decoder | 1 | Reindex | 20 | 32 | 32→K20→θKV20; floating sum at owner | read T32; integer, no autograd | OPEN D6-S: contributor 32, shared indexer owner 32 |
| 33 | decoder | 1 | Reuse | 20 | 32 | 33→K20→θKV20; floating sum at owner | read T32; integer, no autograd | OPEN D6-S: contributor 33, shared indexer owner 32 |
| 34 | decoder | 1 | Reuse | 20 | 32 | 34→K20→θKV20; floating sum at owner | read T32; integer, no autograd | OPEN D6-S: contributor 34, shared indexer owner 32 |
| 35 | decoder | 1 | Reuse | 20 | 32 | 35→K20→θKV20; floating sum at owner | read T32; integer, no autograd | OPEN D6-S: contributor 35, shared indexer owner 32 |
| 36 | decoder | 1 | Reindex | 20 | 36 | 36→K20→θKV20; floating sum at owner | read T36; integer, no autograd | OPEN D6-S: contributor 36, shared indexer owner 36 |
| 37 | decoder | 1 | Reuse | 20 | 36 | 37→K20→θKV20; floating sum at owner | read T36; integer, no autograd | OPEN D6-S: contributor 37, shared indexer owner 36 |
| 38 | decoder | 1 | Reuse | 20 | 36 | 38→K20→θKV20; floating sum at owner | read T36; integer, no autograd | OPEN D6-S: contributor 38, shared indexer owner 36 |
| 39 | decoder | 1 | Reuse | 20 | 36 | 39→K20→θKV20; floating sum at owner | read T36; integer, no autograd | OPEN D6-S: contributor 39, shared indexer owner 36 |

Every consumer has one reachable K and one reachable T.  A Reindex owns only
its T; it never creates K. Keep floating KV tensors differentiable (not
cache-detached) and sum their consumer gradients at the KV owner. Integer Top-K
indices and masks carry dependency only; ordinary autograd does not train an
indexer through them. D6-S must resolve objective, detach boundaries, coefficient
and per-consumer contributions to shared indexer owners before training acceptance.
The auxiliary column names candidate ownership, not an approved contribution rule;
Reuse has no private indexer, which does not imply zero shared auxiliary gradient.

## Exact checkpoint ownership assertion

For KV owner `i`, the complete owner key set is every index key with prefix
`layers.{i}.attn.compressor.`; for index owner `j`, every key with prefix
`layers.{j}.attn.indexer.`.  Required suffixes are `compressor.{norm.weight,
wkv.weight[,wgate.weight]}` and `indexer.{wq_b.weight,weights_proj.weight,
`wk.weight,k_norm.weight` only when that indexer is also a KV owner; plus any
index-present scales).  The validator
compares the union to **all** compressor/indexer keys in the 96,085-key
`weight_map`, rejects modules on non-owner layers, and checks every layer's
unique provenance.  It is not a sampled check.

## Executable validation and limits

Run the CPU dataflow probes independently (each rejected mutation is reported):

```sh
python experimental/lite/tools/validate_deepseek_v41_ced.py \
  --official-model /path/to/model.py
```

The prerequisite A1 snapshot (9d1697a28) supplies the complete owner/key validator,
which also calls these probes; that additional script is not shipped in this patch:

```sh
python experimental/lite/tools/validate_deepseek_v41_owner_consumer.py \
  --model /path/to/model.py --config /path/to/config.json \
  --index /path/to/model.safetensors.index.json
```

The validator executes AST-extracted official Transformer/Block control flow,
ratio-1 Compressor, index-key publication, main cache publication/read, and SWA
projection on CPU fixtures. Lightweight projections and in-place RoPE/quantizer
probes make ordering and provenance observable without GPU kernels. Eleven
mutations bypass the hash gate and must fail on behavior, including lost paired
state, bypassed pre-mix/norm, wrong index input, consumer recompression, and wrong
cache reads. Hashes independently pin the original source and metadata.
This is a dataflow check, not full model numerical, quantized-kernel, GPU or
backward parity. Source/consumer allocation is also checked against all keys.

The CED probe uses distinct incoming, attention and FFN coefficients. It checks
attention/FFN inputs and returned residual/mix, with eleven rejected mutations.
Layer-loop and hc_post doubles do not validate full mHC mathematics or gradients.
The indexer prefix tests K generation/publication only; full Reindex shared-K
consumption and native consumer behavior remain B1/D2 obligations.

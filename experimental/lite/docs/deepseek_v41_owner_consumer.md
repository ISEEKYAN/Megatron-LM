# DeepSeek-V4.1-Flash: CSA2 owner/consumer contract and CED gate

This is an auditable boundary, not an implementation claim.  The official
`DeepSeek-V4.1-Flash/main` snapshot downloaded 2026-09-10 has SHA-256
`model.py=4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`,
`config.json=8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`,
and `model.safetensors.index.json=74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.

## The CED fact, and the definition a port must add

Official `inference/model.py` does **not** implement CED and does **not**
define `H_{L/2}`.  It builds all `range(args.n_layers)` blocks at lines
1201--1205, is inference-only (`@torch.inference_mode()`, lines 1241--1242),
and executes them in the single ordered loop at lines 1261--1267.  Therefore
the exact official definition is: **`H_{L/2}` is undefined.**  The prior
claim that it feeds layer-20 KV was false.

A training CED port must introduce, and test, this exact definition instead:

```text
L = config.text_config.num_hidden_layers = 40
H_{L/2} = H_20 = h after Block(19) returns at official model.py:1267
P_20 = the paired pre_mix returned at that same line
```

Layer 20 continuation consumes `(H_20, P_20)`; a CED global-KV projection may
consume only `H_20`.  Keeping `P_20` is mandatory because the official loop
updates both state values.  This is a port contract, not released behavior.

## 40-layer static assignment

From official config: `kv_source_layer_ids=[2,8,14,20]`,
`index_source_layer_ids=[2,8,14,20,24,28,32,36]`.  Source code selects the KV
owner at lines 653--661, readers at 739--765, and Top-K publishers/readers at
721--737.  `K` and `T` are unique KV and Top-K sources. `—` is local SWA.

| layer | side | ratio | mode | K | T | consumer gradient route |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 0 | encoder | 0 | SWA | — | — | local |
| 1 | encoder | 0 | SWA | — | — | local |
| 2 | encoder | 2 | Full | 2 | 2 | owner; sum consumers 3--7 |
| 3 | encoder | 2 | Reuse | 2 | 2 | 3→K2,T2→θKV2,θI2 |
| 4 | encoder | 2 | Reuse | 2 | 2 | 4→K2,T2→θKV2,θI2 |
| 5 | encoder | 2 | Reuse | 2 | 2 | 5→K2,T2→θKV2,θI2 |
| 6 | encoder | 2 | Reuse | 2 | 2 | 6→K2,T2→θKV2,θI2 |
| 7 | encoder | 2 | Reuse | 2 | 2 | 7→K2,T2→θKV2,θI2 |
| 8 | encoder | 2 | Full | 8 | 8 | owner; sum consumers 9--13 |
| 9 | encoder | 2 | Reuse | 8 | 8 | 9→K8,T8→θKV8,θI8 |
| 10 | encoder | 2 | Reuse | 8 | 8 | 10→K8,T8→θKV8,θI8 |
| 11 | encoder | 2 | Reuse | 8 | 8 | 11→K8,T8→θKV8,θI8 |
| 12 | encoder | 2 | Reuse | 8 | 8 | 12→K8,T8→θKV8,θI8 |
| 13 | encoder | 2 | Reuse | 8 | 8 | 13→K8,T8→θKV8,θI8 |
| 14 | encoder | 2 | Full | 14 | 14 | owner; sum consumers 15--19 |
| 15 | encoder | 2 | Reuse | 14 | 14 | 15→K14,T14→θKV14,θI14 |
| 16 | encoder | 2 | Reuse | 14 | 14 | 16→K14,T14→θKV14,θI14 |
| 17 | encoder | 2 | Reuse | 14 | 14 | 17→K14,T14→θKV14,θI14 |
| 18 | encoder | 2 | Reuse | 14 | 14 | 18→K14,T14→θKV14,θI14 |
| 19 | encoder | 2 | Reuse | 14 | 14 | 19→K14,T14→θKV14,θI14 |
| 20 | decoder | 1 | Full | 20 | 20 | owner; sum KV consumers 21--39 |
| 21 | decoder | 1 | Reuse | 20 | 20 | 21→K20,T20→θKV20,θI20 |
| 22 | decoder | 1 | Reuse | 20 | 20 | 22→K20,T20→θKV20,θI20 |
| 23 | decoder | 1 | Reuse | 20 | 20 | 23→K20,T20→θKV20,θI20 |
| 24 | decoder | 1 | Reindex | 20 | 24 | 24→K20→θKV20; T24→θI24 |
| 25 | decoder | 1 | Reuse | 20 | 24 | 25→K20,T24→θKV20,θI24 |
| 26 | decoder | 1 | Reuse | 20 | 24 | 26→K20,T24→θKV20,θI24 |
| 27 | decoder | 1 | Reuse | 20 | 24 | 27→K20,T24→θKV20,θI24 |
| 28 | decoder | 1 | Reindex | 20 | 28 | 28→K20→θKV20; T28→θI28 |
| 29 | decoder | 1 | Reuse | 20 | 28 | 29→K20,T28→θKV20,θI28 |
| 30 | decoder | 1 | Reuse | 20 | 28 | 30→K20,T28→θKV20,θI28 |
| 31 | decoder | 1 | Reuse | 20 | 28 | 31→K20,T28→θKV20,θI28 |
| 32 | decoder | 1 | Reindex | 20 | 32 | 32→K20→θKV20; T32→θI32 |
| 33 | decoder | 1 | Reuse | 20 | 32 | 33→K20,T32→θKV20,θI32 |
| 34 | decoder | 1 | Reuse | 20 | 32 | 34→K20,T32→θKV20,θI32 |
| 35 | decoder | 1 | Reuse | 20 | 32 | 35→K20,T32→θKV20,θI32 |
| 36 | decoder | 1 | Reindex | 20 | 36 | 36→K20→θKV20; T36→θI36 |
| 37 | decoder | 1 | Reuse | 20 | 36 | 37→K20,T36→θKV20,θI36 |
| 38 | decoder | 1 | Reuse | 20 | 36 | 38→K20,T36→θKV20,θI36 |
| 39 | decoder | 1 | Reuse | 20 | 36 | 39→K20,T36→θKV20,θI36 |

Every consumer has one reachable K and one reachable T.  A Reindex owns only
its T; it never creates K.  For a training port, keep these values
differentiable (not cache-detached) and assert `dL/dθo=dLowner/dθo+Σc dLc/dθo`.

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

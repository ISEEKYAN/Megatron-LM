# DeepSeek-V4.1-Flash: CED and CSA2 ownership contract

This is the implementation boundary for the DeepSeek-V4.1-Flash port.  It is
derived from the official `config.json` and the *DeepSeek-V4.1 Technical
Report*, §2.2--2.3.  It deliberately describes the 40 backbone layers only:
DSpark forward/rollout is not part of this port.

## Ownership graph

```text
vision encoder -> pixel-unshuffle -> vision projector --\
text embedding ----------------------------------------+-> language input
                                                           |
                                             CED encoder (layers 0..19)
                                                           |
                                      H[19] --global-KV projection--> decoder CSA2 Full
                                                           |                        |
                                      encoder local SWA KV             shared global KV/indexer K
                                                           |                        |
                                  bounded replay consumer <--- decoder local SWA KV
                                                           |
                                                 CED decoder (layers 20..39)
                                                           |
                                                    norm -> LM head
```

The owners and consumers are intentionally separate:

| State / decision | Single owner | Consumers | Contract |
| --- | --- | --- | --- |
| multimodal token embeddings | vision encoder + projector, then language embedding | layer 0 | The projector inserts visual embeddings at image-token positions before the language stack; no CSA2 module accepts raw images. |
| CED split and encoder terminal hidden state | V4.1 language model | CSA2 Full layer 20 | Layers `0..19` are the causal encoder; `H[19]` is the sole source of decoder **global** KV projections. |
| local SWA KV | the current language layer | that same layer's local-attention branch | Every layer owns its own SWA KV.  Decoder SWA is populated by bounded replay; it must not be substituted with encoder SWA KV. |
| global main KV and indexer K | most recent CSA2 Full layer | later Reindex/Reuse layers | A Full layer produces main KV; indexer K is projected from that KV.  Under CED, decoder Full layer 20 projects its global KV from `H[19]`, not its own hidden state. |
| indexer Q and fresh Top-K | current Full or Reindex layer | its attention and later Reuse layers | Reindex has its own Q, scores the shared indexer K, and publishes a new Top-K.  In the decoder it searches the Full-20 candidate pool. |
| reused Top-K | latest Full/Reindex layer for the active KV source | Reuse layers | Reuse has no indexer-Q or score computation; it reads the latest compatible Top-K. |
| static schedule validation | V4.1 config builder | model constructor and CSA2 primitive | `compress_ratios`, `kv_source_layer_ids`, and `index_source_layer_ids` are immutable model configuration, not runtime routing decisions. |

`protocol.py` owns batch normalization, packed-sequence/CP adaptation, and the
public forward call.  It does **not** choose a CSA2 mode or slice CED states.
The V4.1 model owns the encoder/decoder transition and passes an explicit,
validated per-layer assignment into the shared CSA2 primitive.  This preserves
the existing model/protocol boundary and prevents a second CP implementation
from appearing inside attention.

## CED boundary

For `L = 40`, CED partitions the backbone at `L / 2 = 20`:

| Phase | Layers executed over the full prompt | Global KV source | Local/SWA work |
| --- | --- | --- | --- |
| prefill | encoder `0..19` | encoder Full layers make their normal global KV; decoder layer 20 projects global KV from `H[19]` | decoder local state is reconstructed only through bounded replay of the trailing window |
| decode | encoder state plus decoder `20..39` | decoder Full-20 global KV remains the source; later decoder CSA2 layers reuse it | every decoder layer uses its own, current-layer SWA KV |

The key prohibition is that CED shares **global** KV only.  It neither shares
decoder hidden states nor turns the decoder into an ordinary cross-attention
stack.  Position IDs, causal masking, packed-sequence metadata, and CP layout
therefore continue through the language-model forward boundary unchanged.

## Static CSA2 assignment

The table is mechanically derived from the official arrays:

```text
Full    := layer in kv_source_layer_ids
Reindex := layer in index_source_layer_ids but not kv_source_layer_ids
Reuse   := compress_ratio > 0 and neither source array contains the layer
SWA     := compress_ratio == 0
```

| Layer(s) | CED side | attention / ratio | mode | global-KV source | Top-K producer / consumer |
| --- | --- | --- | --- | --- | --- |
| 0--1 | encoder | SWA only | — | — | — |
| 2 | encoder | CSA2 / 2 | Full | 2 | produces Top-K for 2--7 |
| 3--7 | encoder | CSA2 / 2 | Reuse | 2 | consume Top-K from 2 |
| 8 | encoder | CSA2 / 2 | Full | 8 | produces Top-K for 8--13 |
| 9--13 | encoder | CSA2 / 2 | Reuse | 8 | consume Top-K from 8 |
| 14 | encoder | CSA2 / 2 | Full | 14 | produces Top-K for 14--19 |
| 15--19 | encoder | CSA2 / 2 | Reuse | 14 | consume Top-K from 14 |
| 20 | decoder | CSA2 / 1 | Full | 20, projected from `H[19]` | full-range index; creates the hierarchical candidate pool and Top-K |
| 21--23 | decoder | CSA2 / 1 | Reuse | 20 | consume Top-K from 20 |
| 24 | decoder | CSA2 / 1 | Reindex | 20 | fresh Top-K within layer-20 candidate pool |
| 25--27 | decoder | CSA2 / 1 | Reuse | 20 | consume Top-K from 24 |
| 28 | decoder | CSA2 / 1 | Reindex | 20 | fresh Top-K within layer-20 candidate pool |
| 29--31 | decoder | CSA2 / 1 | Reuse | 20 | consume Top-K from 28 |
| 32 | decoder | CSA2 / 1 | Reindex | 20 | fresh Top-K within layer-20 candidate pool |
| 33--35 | decoder | CSA2 / 1 | Reuse | 20 | consume Top-K from 32 |
| 36 | decoder | CSA2 / 1 | Reindex | 20 | fresh Top-K within layer-20 candidate pool |
| 37--39 | decoder | CSA2 / 1 | Reuse | 20 | consume Top-K from 36 |

Thus the only global-KV producers are `2`, `8`, `14`, and `20`; the
index-producing layers are `2`, `8`, `14`, `20`, `24`, `28`, `32`, and `36`.
All other CSA2 layers are consumers.  `ratio=1` is an uncompressed main-KV
CSA2 setting, not a request to fall back to DS4 CSA or HCA.

## CSA2 mode interface

Every CSA2 invocation receives its per-layer static assignment plus a
read-only shared-state handle.  The primitive has the following minimal
responsibilities:

| Mode | Current-layer computation | Reads | Publishes |
| --- | --- | --- | --- |
| Full | main Q, main KV, indexer K from main KV, indexer Q, scores, Top-K, SWA KV | none (except CED's `H[19]` input for decoder Full-20 KV projection) | KV-source record and Top-K record; Full-20 also publishes candidate pool |
| Reindex | main Q, indexer Q, scores, fresh Top-K, SWA KV | most recent Full main KV + indexer K; decoder candidate pool | Top-K record, associated with the reused KV source |
| Reuse | main Q and SWA KV | most recent Full main KV + indexer K and latest compatible Top-K | attention output only |

An implementation must reject an assignment that has no preceding Full source,
or a Reuse row whose Top-K is associated with another KV source.  It must also
reject a decoder Reindex without Full-20's candidate pool.  Those checks make
the owner/consumer graph executable rather than a comment-only convention.

## Scope hand-off

This document decides ownership only.  The checkpoint-key contract (including
the three MTP layers), individual CSA-to-CSA2 operator deltas, and optimizer
parameter groups belong to their dedicated follow-up work.  DSpark metadata
(`dspark_target_layer_ids = [37, 38, 39]`) may be loaded/mapped by the
checkpoint path, but DSpark forward and rollout are expressly out of scope.

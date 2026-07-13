# EP / expert weight-export residency: upstream comparison

This note records how Megatron Lite's Megatron→HF weight export handles the MoE
expert dimension, compared against the two upstream Megatron→HF bridges, and why
Lite's current expert path is already at least as memory-frugal as either
upstream. It backs the residency unit test
`tests/unit/primitive/test_hf_weights_streaming.py::test_expert_export_never_materializes_the_whole_expert_set`.

## Reference sources

| Project | Ref | Fetched |
|---|---|---|
| ISEEKYAN/mbridge | `a61943d7fcb34a190471cfeb0a0eb8bbda621ddf` | 2026-07-13 |
| NVIDIA-NeMo/Megatron-Bridge | `833b9965a9a4dfe8872509b0432bcff7886a3b8a` | 2026-07-13 |

(NVIDIA/Megatron-Bridge redirects 404; the NVIDIA project now lives under
NVIDIA-NeMo/Megatron-Bridge.)

## The three in-memory export paths

All three projects stream Megatron→HF `(name, tensor)` pairs and, for MoE
models, must reconstruct the global expert ids by all-gathering each local
expert tensor across the expert-parallel (EP) group. The relevant difference is
**how much of the expert set is resident at once** on each rank during that
gather.

### 1. ISEEKYAN/mbridge — bucketed EP gather (bounded)

`mbridge/core/bridge.py`:

- `export_weights` (line 956) → `_iter_bucketed_export_outputs` (line 802).
- EP params are accumulated into `ep_bucket`, capped by
  `_get_collective_bucket_size_bytes(ep_size)` = `buffer // ep_size`
  (lines 659–662, cap applied at line 816).
- `_flush_ep_bucket` (line 883) gathers one capped `ep_bucket`, then expands it
  into an `etp_bucket` holding `ep_size ×` the gathered bucket (lines 895–906)
  before `_iter_etp_bucket_outputs` streams it out.
- Flat gather buffer: `bucketed_all_gather_into_tensor`
  (`mbridge/core/util.py` line 785).
- Default buffer: `export_weights_buffer_max_size_bytes = 2 GiB`
  (`bridge.py` line 41).

Peak expert residency ≈ one bounded `ep_bucket` fully expanded across EP ranks
(≈ `buffer_max_size_bytes`), **not** the whole expert set. This is a genuinely
bounded, streaming path.

### 2. NVIDIA-NeMo/Megatron-Bridge — per-param EP all-gather (uncapped)

`src/megatron/bridge/models/conversion/param_mapping.py`:

- `gather_from_ep_ranks` (line 733): for each expert param it allocates
  `[empty_like(w) for _ in range(ep_size)]` and does a single
  `torch.distributed.all_gather(gathered_weights, megatron_weights, group=ep_group)`
  (lines 805–806), then returns a dict mapping every global expert id to its
  gathered tensor (lines 809–819).
- There is **no per-rank byte cap / bucket** on the EP dimension — the full
  gathered set of that param (one tensor per EP rank) is materialised per param.

Per-param (so not the *whole model* at once), but uncapped along EP: every rank
holds the full EP-gathered copy of each expert param with no `buffer // ep_size`
cap. For a wide EP group this is strictly less frugal than a `buffer // ep_size`
bucket.

### 3. ISEEKYAN/mbridge — `export_weights_without_gather` is a *disk* path

`export_weights_without_gather` (`bridge.py` line 1007) is the only "no gather"
option and it is **not** an in-memory colocated-vLLM resync: it yields each
rank's local shard plus `(tp_rank, ep_rank, partition_dim, …)` metadata for a
**distributed-filesystem** checkpoint, where a later reader reassembles
`weights[rank::world_size]`. It has no counterpart in the RL weight-sync path
(colocated rollout needs the full HF tensor on the receiving rank), so it is not
a copy source for reducing the in-memory expert peak.

### 4. Megatron Lite — bucketed EP gather with a double cap (bounded)

`experimental/lite/megatron/lite/primitive/ckpt/hf_weights.py`:

- `export_hf_weights` (line 791), `pp=1` branch.
- Expert params accumulate in `expert_bucket`, capped by
  `expert_bucket_limit_bytes = buffer // ep_size` (lines 825–830).
- `_flush_expert_bucket` (line 863) gathers one bucket via
  `bucketed_all_gather_into_tensor` (line 304) and yields each global-id shard
  immediately (`packed_name is None`, DS4 path, line 888), releasing shards as
  it goes.
- Two flush triggers (lines 946–949): `expert_bucket_bytes >= buffer // ep_size`
  **and** `len(expert_bucket) >= 4`. The extra `len >= 4` trigger caps residency
  even for tiny experts where the byte cap alone would let many params pile up.
- Default buffer: `DEFAULT_EXPORT_BUFFER_MAX_SIZE_BYTES = 2 GiB` (line 70).
- The `pp>1` path streams the PP dimension over the NCCL `pp_group` one bounded
  bucket + one in-flight param at a time (never the whole stage), see
  `export_hf_weights` PP branch and `ds4-resync-memory-protocol.md`.

## Verdict: Lite is already ≥ both upstreams

| Path | EP gather | Per-rank cap | Peak expert residency |
|---|---|---|---|
| NVIDIA-NeMo Megatron-Bridge | per param, `all_gather` | **none** | full EP-gathered copy per param, uncapped |
| ISEEKYAN mbridge | bucketed | `buffer // ep_size` | one bounded bucket, expanded across EP |
| **Megatron Lite** | bucketed | `buffer // ep_size` **and** `len >= 4` | one bounded bucket, ≤4 local params |

- vs NVIDIA-NeMo Megatron-Bridge: Lite is **strictly more frugal** — it applies
  a `buffer // ep_size` byte cap that Megatron-Bridge's `gather_from_ep_ranks`
  lacks entirely.
- vs ISEEKYAN mbridge: Lite matches mbridge's `buffer // ep_size` cap and adds a
  `len >= 4` residency ceiling, so Lite is **at least as** frugal.

Therefore "align to upstream to eliminate whole-set expert materialisation" has
**no in-memory copy source**: neither upstream in-memory path is more frugal
than Lite's existing one. The only structurally different option — feeding each
rank only its local experts and skipping the EP gather ("owner-only") — is a new
architecture, not a copy of either upstream, and would require an EP-aware
receiver on the rollout side; it is deliberately out of scope here and deferred.

## Evidence

- CPU unit test: the residency guard
  `test_expert_export_never_materializes_the_whole_expert_set` asserts the
  expert path streams bounded buckets and goes RED if reverted to whole-set
  materialisation (verified by removing the flush triggers: the count of EP
  collectives before the first yield jumps from 1 to the whole local set).
- 8-GPU measured peak: a DS4-proxy `pp2 ep2` export on 8×H100 measured
  `peak = 2.39 MiB/rank` (`n_tensors = 142`, `missing_layers = []`,
  `layers_ok = true`), i.e. bounded streaming, not whole-dict materialisation.
  The absolute per-rank export peak stays orders of magnitude below the whole
  expert set, consistent with the bucketed path above.

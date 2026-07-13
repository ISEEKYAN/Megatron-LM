# DeepSeek-V4 CSA — CP activation-memory review

Scope: the reported DS4 Compressed Sparse Attention (CSA) out-of-memory under
context parallelism (CP), and the upstream community proposal that adds a
streaming (online) softmax to work around it.

The finding below is that **our port dropped the memory-bounding property of the
Megatron reference**: the reference CSA is *sparse by construction* (it only ever
materializes scores for the top-k gathered keys), whereas our port materializes a
*dense* `[B, H, Sq, Skv]` fp32 score matrix and then masks it. Fixing the port to
match the reference is preferable to layering a streaming-softmax band-aid on top
of the dense algorithm.

File references below use two sources on branch `dev-mlite-7-deepseek-v4`:

- **PORT** — `experimental/lite/megatron/lite/primitive/modules/attention/csa.py`
  (this is byte-identical to the file the community PR targets).
- **REF** — `megatron/core/transformer/experimental_attention_variant/csa.py`
  (the Megatron-core reference the port was derived from).

---

## 1. The mis-copy (root cause)

### Megatron reference is sparse: `O(Sq · topk)`

`unfused_compressed_sparse_attn` (REF:188–250) is the differentiable pure-torch
attention. It receives precomputed `topk_indices : [b, sq, topk]` (REF:201) — the
union of the sliding window and the compressed/indexer top-k — and **gathers only
those keys**:

- `kv_gathered = torch.gather(...) -> [b, sq, topk, hn]` (REF:216–218)
- `scores = einsum("bnsh,bskh->bnsk") -> [b, np, sq, topk]` (REF:226)

`topk = window_size + index_topk` is a **fixed constant**, independent of the
total key length. So the score/probability activations are `O(sq · topk)` and are
independent of sequence length. Under CP, `sq` is the *local* (sharded) query
count, so per-rank activation memory drops ~linearly with CP.

The index selection itself (`fused_qk_topk_naive`, REF:548) runs on the small
compressed sequence (`sq // ratio`), never on the full `Sq × Skv` grid.

### Our port is dense: `O(Sq · Skv)`

`CompressedSparseAttention.forward` (PORT:284–474) has three paths:

- fused DSA (PORT:315–331) — requires `cp_size == 1` (PORT:586–587)
- fused sparse no-indexer (PORT:332–340) — requires `cp_size == 1`
- **the pure-torch fallback (PORT:342–474)** — the only path valid for `cp_size > 1`

The fallback builds the score matrix over the **entire** key sequence and then
applies sparsity as `-inf` *masks*:

- window/dense scores: `torch.matmul(q, source_headsᵀ) -> [B, H, Sq, Skv_source]`
  per CP source, then `torch.cat(..., dim=-1)` over all all-gathered CP sources →
  `[B, H, Sq, Skv_total]` fp32 (PORT:344–367).
- compressed scores: `[B, H, Sq, n_compressed]` (PORT:387–395).
- the indexer top-k **is computed** (PORT:445–448) but is only used to build
  `topk_mask` and `masked_fill(~topk_mask, -inf)` on the already-materialized
  dense `compressed_scores` (PORT:447–452). It never reduces the tensor size.
- `scores = torch.cat(score_parts, -1)`, `+ sink`, full `softmax` in fp32
  (PORT:456–464), then `probs @ values` (PORT:466–472).

Peak activation is therefore `O(Sq · Skv)` in fp32 (plus the fp32 softmax
buffers). Under CP, `Sq` is local (`S/cp`) but `Skv_total` is the full
all-gathered sequence, so per-rank score memory is `O(S² / cp)` — dense, and only
linearly relieved by CP. For long context this OOMs; the fused kernels that would
avoid it are `cp_size == 1`-only, so **CP training is forced onto the dense path.**

### Why this is a mis-copy and not a CP necessity

After `iter_cp_sources` all-gathers every rank's KV
(`attention/cp.py::iter_cp_sources`), each rank holds the *full* key sequence with
*local* queries. At that point the reference's sparse gather is directly
applicable — gather window ∪ compressed-top-k keys from the assembled `kv_full`
and score only those. The port instead re-derived a dense masked softmax and lost
the reference's defining memory property. MLite ported the *fused* GPU sparse path
(`dsa_kernels.SparseAttnFunc`, FlashMLA + cuDNN, CP=1-only) but **never ported the
pure-torch differentiable sparse gather** (`unfused_compressed_sparse_attn`),
which is exactly the piece the CP path needs.

Note: because the default `attention_backend = "torch"` (PORT:248) makes
`use_sparse_backend` False (PORT:314), the dense path is also the default even at
CP=1 — CP simply makes it unavoidable.

---

## 2. The faithful fix (preferred over adopting the PR)

Port Megatron's `unfused_compressed_sparse_attn` (REF:188–250) — a ~60-line
pure-torch, differentiable, sparse-gather kernel — into MLite, and route the
`cp_size > 1` fallback (and the CP=1 torch-backend default) through it:

1. assemble `kv_full` from the all-gathered CP sources + compressed KV (the port
   already all-gathers these);
2. build `topk_indices = window_idxs ∪ compressed_topk_idxs` (the port already
   computes the indexer top-k at PORT:445–448; the window indices exist as
   `_window_topk_indices`);
3. gather + score only those keys → `[B, H, Sq, topk]`, softmax with sink, weighted
   sum — identical math to REF:207–250.

Result: activation memory becomes `O(Sq · topk)` (CP-scaling, sequence-length
independent) with **no** hand-rolled streaming-softmax autograd. This is the
"copy it correctly" fix and matches the reference numerics by construction.

Target branch note: `csa.py` lives on `dev-mlite-7-deepseek-v4` (the DS4 line);
it is not present on `main`. The fix + its GPU verification must land there.

---

## 3. GPU verification plan (pending budget / pre-GPU gate)

Not yet run. Required before claiming the fix (per project GPU discipline):

- 8-card proxy, same recipe scaled; compare **CP1 vs CP4** per-rank activation
  memory for the CSA block (a pre-burn memory budget table + measured curve).
- Expectation: with the faithful fix, the sequence-scaling score activation drops
  ≈ CP factor and is bounded by `topk`, not `Skv`; dense baseline should show the
  `O(S²/cp)` blowup for contrast.
- DS4 env per the established recipe (GB200/SM100 overlay). Needs GPU budget
  allocation + the pre-GPU review gate.

---

## 4. Evaluation of the community streaming-softmax proposal

What it does: keeps the **dense** algorithm but replaces the monolithic
score-matrix + softmax with a chunked **online softmax** over the query
dimension, plus three custom `torch.autograd.Function`s (dense / compressed /
compressed-indexed) that **recompute scores in backward** to avoid caching the
`[B,H,Sq,Skv]` tensors. It is gated on `attention_mask is None` (the causal/CP
training path); the `attention_mask is not None` path keeps the old dense code.
~1200 LOC added, with unit tests asserting numerical equivalence to the dense
path. A reviewer bot flagged one real perf nit: a `torch.equal` on GPU tensors in
the forward introduces a device-host sync (`csa.py:1089` in the PR).

Assessment:

- **Pros**: bounds peak fp32 score memory to ~256 MB/chunk and cuts backward
  activation caching; stays within the current dense masked design; ships parity
  unit tests.
- **Cons**: it treats the symptom, not the cause. It is still the dense algorithm
  — every query still touches every key, so **compute stays `O(S²)`** even though
  peak memory is bounded. It adds ~1200 lines of hand-rolled forward/backward
  autograd (a large correctness/maintenance surface) to reproduce what the
  reference gets for free from a ~60-line sparse gather. It only engages for
  `attention_mask is None`.

**Verdict: do not adopt as the primary fix.** Implement the faithful sparse port
(§2) first; that removes the OOM at the root and makes the streaming path
unnecessary for the general CSA case. Residual value is narrow: the
streaming-softmax + backward-recompute technique is still useful specifically
where a *genuinely dense* score matrix is unavoidable — e.g. a `csa_dense_mode`
layer, an explicit-`attention_mask` path, or a 128×-compress "attend-to-all" layer
where `topk ≈ full`. In those niches the technique could be kept as an opt-in; it
is not the correct general-purpose remedy for the CP OOM.

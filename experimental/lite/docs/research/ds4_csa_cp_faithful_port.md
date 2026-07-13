# DeepSeek-V4 CSA Context-Parallel: Faithful-Port Research (v2)

Status: research deliverable (回炉调研, second pass). **This version supersedes
v1.** v1 read a *stale* vendored Megatron copy and concluded the cross-CP
semantics were "invisible" (an unresolved A/B reading). That conclusion was
wrong: the merged upstream now contains an explicit, fully-materialized,
memory-bounded DS4 CSA CP path. bayan's directive ("你读的参考是过期的——最新
NVIDIA Megatron dev 已 merge 了 DS4 CP") is confirmed by primary source; the A/B
tension is **resolved**.

Scope: (1) characterize the current MLite DS4 CSA CP delivery ("the hack"),
(2) reconstruct the **merged** upstream fused DS4 CSA CP from primary source with
fresh line numbers, (3) specify the faithful-port plan and its real scope.
**Does not** include GPU implementation/verification — that is the downstream
implementation task on `dev-mlite-7-deepseek-v4`.

## References

- **Upstream faithful reference**: `NVIDIA/Megatron-LM` branch `dev`, tip
  `fd1121b8f` (fetched 2026-07-13; commit date 2026-07-09). Files under
  `megatron/core/transformer/experimental_attention_variant/`:
  `csa.py`, **`csa_cp_utils.py` (NEW)**, **`csa_cp_layout_kernels.py` (NEW)**,
  `deepseek_v4_hybrid_attention.py`, `dsa.py`, `dsa_kernels.py`.
  The two NEW modules are exactly the merged CP path that v1's stale reference
  lacked; upstream also ships unit tests `test_csa_cp_utils.py`,
  `test_csa_cp_layout_kernels.py`.
- **MLite port (the hack)**: branch `dev-mlite-7-deepseek-v4`,
  `experimental/lite/megatron/lite/primitive/modules/attention/csa.py` (CSA),
  `.../attention/dsa.py` (sibling DSA — *not* on the DS4 path, see §6),
  `.../attention/cp.py` (CP helpers).

---

## 1. What the current MLite CSA delivers (the "hack")

`experimental/lite/.../attention/csa.py`, `forward` (L284):

- `cp_size == 1` → real fused sparse path (`_forward_fused_dsa_cp1` L574 →
  `dsa_kernels.fused_indexer_sparse_attn` / `indexer_topk` + `dsa_sparse_attn`).
  **Correct and kept.**
- `cp_size > 1` → **dense masked-softmax fallback** (L342–464):
  - `iter_cp_sources` (`cp.py` L13) → `_all_gather_cp` (`cp.py` L9–10) does a
    `torch.distributed.nn.functional.all_gather` of the **full dense KV** across
    every CP rank,
  - builds `dense_scores` via `torch.einsum` + `masked_fill(-inf)` + `torch.softmax`
    over the **entire gathered sequence** (L342–464).
  - `_forward_fused_dsa_cp1` additionally **hard-raises**
    `NotImplementedError("DeepSeek V4 fused DSA path currently supports CP=1 only.")`
    (L586–587).

**Consequence.** Under CP>1 the sparse structure is discarded: the path
materializes an `(L_local × S_total)` = `O(S²/cp)` dense score matrix plus a
full-sequence KV all-gather (`O(S)` rows/rank). This defeats the memory saving CP
exists to buy and matches the 128-card resync OOM (memory
`ds4-csa-cp-dense-miscopy`, `ds4-128card-resync-oom`).

---

## 2. What the **merged** upstream fused DS4 CSA CP does (primary source)

The merged path is a THD-packed (variable-length, `bsz=1`) context-parallel
branch that is **memory-bounded** — it never all-gathers full dense KV and never
builds a dense score matrix. Call chain:

### 2a. Wrapper: `deepseek_v4_hybrid_attention.py::forward` (L222)

- Preconditions for CP>1 (L267–271): `qkv_format == 'thd'` **and** a
  **contiguous** CP partition (`cp_partition_mode == "contiguous"`); otherwise
  raises. (No zigzag layout on this path — contiguous row blocks per rank.)
- Computes a **bounded left-boundary window** of hidden rows via
  `cp_utils.exchange_cp_boundary_hidden(hidden_states, compress_ratio,
  csa_window_size, cp_group)` (L276–281).
- `get_query_key_value_tensors(..., boundary_hidden=...)` (L288) additionally
  returns a projected **`boundary_kv`** (the boundary rows run through the KV
  projection; L296–300, projection at L668–754).
- Passes `boundary_hidden` + `boundary_kv` into `core_attention(...)`
  (L314–325).

### 2b. Boundary exchange: `csa_cp_utils.py`

- `_LeftBoundaryExchange(torch.autograd.Function)` (L123–187): a **P2P**
  (`batch_isend_irecv`) exchange of only the fixed left-boundary window from the
  previous rank, with a matching gradient scatter in `backward`.
- `exchange_cp_boundary_hidden` (L190–201): window size
  `d_window = max(csa_window_size, d_comp)` where `d_comp = 8 if ratio==4 else
  ratio` (L197–198). **Bounded** — independent of sequence length.

### 2c. Core CP branch: `csa.py::_forward_thd_cp` (L2279–2557)

Dispatched from `forward` (L1836–1844: `thd` + `cp.size() > 1` → `_forward_thd_cp`,
else `_forward_thd`).

1. `global_start = cp_rank * l_local`; `kv_local` is this rank's local KV
   (`L/cp` rows) (L2310–2311); `boundary_kv` squeezed (L2317); `d_window` from
   boundary (L2318).
2. Build fixed-capacity compressor input from local + boundary hidden via
   `cp_utils.prepare_cp_compressor_input(x, boundary_hidden, ...)` (L2349–2359),
   which calls the Triton `CompressorInputCompact` kernel
   (`csa_cp_layout_kernels.py` L648) and returns `seq_to_rank_row` (the
   sequence-major↔rank-major compressed-row map).
3. Indexer path (L2361–2433): compress the indexer K locally, **all-gather the
   *compressed* indexer K rank-major** (`gather_from_sequence_parallel_region`,
   L2410–2412) — gather is in **compressed space** (`S/ratio`), reindex to
   sequence-major via `seq_to_rank_row`, then `cp_utils.compute_cp_indexer_topk`
   (L2421–2433) produces local-Q/full-(compressed-)K top-k.
4. Attention compressed KV (L2436–2445): compress KV locally, **all-gather
   compressed KV rank-major** — again compressed space (`S/ratio`).
5. **`kv_full_thd = torch.cat((boundary_kv, kv_local, compressed_kv_rank_major),
   dim=0)`** (L2450). The three sources are: boundary window (`d_window`, bounded)
   + local KV (`S/cp`) + full **compressed** KV (`S/ratio`). **No full dense KV.**
6. `csa_cp_layout_kernels.build_attention_indices` (L751; called L2462–2476)
   lowers logical top-k ids into physical rows of `kv_full_thd`.
7. Sparse attention on the top-k indices: `dsa_sparse_attn` (fused, L2544) or
   `unfused_compressed_sparse_attn` (L2554). Indexer-loss training variant at
   L2477–2540.

### 2d. Memory bound (the whole point)

Per-rank KV working set on the merged path:
`d_window + S/cp + S/ratio` rows, and attention is sparse over top-k
(`O(L_local · topk)`), **not** `O(S²/cp)`. With `ratio=4`, `d_window≈window`,
this is dramatically below the hack's dense matrix. This is the memory-factor
saving the AC demands, and it is achievable by construction — unlike the hack.

---

## 3. A/B tension: RESOLVED

v1 could not decide between (A) "cross-CP semantics hidden below the visible
layer / in compiled kernels" and (B) "fused CP is RoPE-only, CP>1 sparse
correctness unproven." Both are now settled by the merged source:

- **Reading A is essentially correct, but the mechanism is explicit, not hidden.**
  Cross-CP KV movement is real and lives in plain Python: a bounded P2P
  boundary exchange (`_LeftBoundaryExchange`) + **compressed-space** rank-major
  all-gathers (`gather_from_sequence_parallel_region` on `S/ratio` rows) +
  fixed-capacity layout kernels (`csa_cp_layout_kernels.py`). It is *not* buried
  in `dsa_kernels`, and it is *not* a full dense all-gather.
- **Reading B is refuted.** CP>1 sparse correctness is not "unproven RoPE
  threading" — it is a first-class, unit-tested path (`test_csa_cp_utils.py`,
  `test_csa_cp_layout_kernels.py`). The load-bearing unknown v1 flagged ("do the
  MLite `dsa_kernels` carry cross-CP capability?") **dissolves**: upstream does
  the cross-CP work *outside* the sparse kernel, in the new utils/layout modules;
  the sparse kernel itself stays local and unchanged. So the port does not depend
  on any hidden kernel capability — it depends on porting the utils/layout code.

---

## 4. Faithful-port plan (and its **real** scope)

The faithful port is **not** the "~30-line dispatch unification" v1 floated. It
is a port of the three merged upstream pieces plus wrapper wiring, and a deletion
of the dense fallback:

1. **Port `csa_cp_utils.py`** (≈399 lines): `_thd_cp_position_ids` + fused/unfused
   THD-CP RoPE wrappers (L25–115), `_LeftBoundaryExchange` +
   `exchange_cp_boundary_hidden` (L123–201), `prepare_cp_compressor_input`
   (L209–273), `_build_cp_indexer_layout` + `compute_cp_indexer_topk` (L276–399).
2. **Port `csa_cp_layout_kernels.py`** (≈840 lines, Triton + autograd):
   `CompressorInputCompact` (L648) and `build_attention_indices` (L751) with their
   fwd/bwd Triton launchers. Must reconcile with MLite's own `dsa_kernels`/kernel
   conventions.
3. **Port `_forward_thd_cp`** into MLite `csa.py` (≈280 lines, L2279–2557) and
   wire `forward` to dispatch `thd + cp_size>1 → _forward_thd_cp` (mirror upstream
   L1836–1844).
4. **Wire the wrapper**: MLite's DS4 attention wrapper must compute
   `boundary_hidden` (call `exchange_cp_boundary_hidden`), produce `boundary_kv`
   in the KV projection, and thread both into `core_attention` — mirror
   `deepseek_v4_hybrid_attention.py` L267–325 (incl. the `qkv_format=='thd'` +
   contiguous-partition preconditions).
5. **Delete the hack**: remove MLite `csa.py` dense fallback L342–464 and the
   `iter_cp_sources` full-KV gather usage; remove the `NotImplementedError`
   CP-1-only guard (L586–587). Decide whether `cp.py::iter_cp_sources` /
   `_all_gather_cp` become dead code (they may still serve the sibling DSA — §6).
6. **primitive.contract compliance** (MLite skills): document
   `process_groups_or_device_placement` (boundary P2P + compressed-space
   rank-major gather; what stays sharded), `what_must_match_reference`
   (near-bitwise vs upstream fused `_forward_thd_cp`), `forward_backward_update_details`
   (indexer loss + `_LeftBoundaryExchange.backward` grad scatter under CP), and
   `failure_modes` (non-contiguous partition, `bsz>1`, `local_rows < d_window`).

**Scope callout for bayan**: this is ~1200+ lines of ported kernel/layout/CP
code + wrapper rewrite, not a dispatch tweak. It is a genuine port, and the
Triton layout kernels (`csa_cp_layout_kernels.py`) are the hardest part to get
bit-faithful. Sizing the implementation task accordingly is recommended.

---

## 5. THD-CP preconditions the port must carry (parity contract)

Upstream restricts the CP path; a faithful port must enforce the same, else it
silently diverges:

- `qkv_format == 'thd'` required for CP>1 (else raise) — wrapper L267–268.
- **contiguous** CP partition (not zigzag) — wrapper L270–271.
- `bsz == 1` on the indexer path — `_forward_thd_cp` L2364–2367.
- `local_rows >= d_window` — `_LeftBoundaryExchange.forward` L134–138.

---

## 6. Sibling DSA: upstream unifies by CSA-ownership, so DSA is **not** in scope

bayan asked whether the sibling DSA path changes in lockstep, "上游怎么统一我们
怎么统一." Primary source answer:

- **Upstream `dsa.py` has no boundary-exchange THD-CP path.** It carries no
  `_forward_thd_cp`, no `exchange_cp_boundary`, no rank-major KV all-gather; its
  only CP touch is RoPE `cp_group` + the indexer-loss `pg_collection`. All DS4 CP
  machinery lives in **`csa.py`** (CSA embeds the indexer). Upstream `dsa.py` is a
  separate GPT-DSA variant (cf. `test_dsa_gpt_mamba_equivalence.py`).
- **MLite DS4 uses CSA, not DSA.** `deepseek_v4/lite/model.py` L42 imports
  `CompressedSparseAttention` (csa.py); the sibling `dsa.py` (`_all_gather_cp`
  L201, "Correctness-first" L320, `zigzag_slice_for_cp`) is **not on the DS4 hot
  path**.

**Conclusion**: the faithful DS4 CP fix is a CSA-only change. The sibling
`dsa.py` dense reconstruct has no upstream faithful-port counterpart to copy and
is not exercised by DS4 — it should be left as the correctness-first interim
(addressed separately if/when a DSA-based model needs bounded CP), **not**
force-changed by this task. This narrows v1's "by symmetry the sibling faces the
same question" — symmetry does not apply, because upstream did not unify them.

---

## 7. Structural constraints (why this is a report, not a verified fix)

1. **Code not in this worktree.** Branch
   `feature/megatron-fused-ds4-cp-mlite-hack` holds only the old "Keep only
   experimental lite" snapshot: no `deepseek_v4/`, no CSA port. All DS4 code +
   the port target live on `dev-mlite-7-deepseek-v4`. A faithful *code* port must
   be authored/committed there.
2. **Verification needs GPU + Slurm.** The AC (CP1 vs CP4 memory curve真降 +
   upstream parity) requires multi-GPU Slurm on `dev-mlite-7-deepseek-v4` (task
   profile GPU rule). Not reproducible here.

---

## 8. Verification plan (for the implementation task on `dev-mlite-7-deepseek-v4`)

1. **Memory-factor AC**: CP1 vs CP4 activation-memory curve on ≥8 GPU (or
   physically-justified proxy). The merged path predicts per-rank KV
   `≈ d_window + S/cp + S/ratio`; the hack cannot pass this by construction. Show
   the CP-factor reduction.
2. **Parity AC**: MLite `_forward_thd_cp` (CP4) vs upstream Megatron fused
   `_forward_thd_cp` (CP4) on the same config — this is now a **fused-vs-fused**
   comparison (v1's "dense-vs-dense only" caveat no longer binds), targeting
   near-bitwise. Also fused-vs-unfused within MLite (`dsa_sparse_attn` vs
   `unfused_compressed_sparse_attn`).
3. **Boundary/backward correctness**: verify `_LeftBoundaryExchange.backward`
   grad scatter and indexer-loss reduction (`reduce_group=cp_group`) under CP.
4. **Gate discipline**: CONFIG_ONLY init-chain gate + 8-card proxy before any
   large job; py-spy within 5 min of RUNNING (GPU铁律); two review.moe gates
   (pre-GPU + pre-Done).

---

## 9. Recommendation

The A/B blocker is gone. Open an implementation task on
`dev-mlite-7-deepseek-v4` to **port** upstream `csa_cp_utils.py` +
`csa_cp_layout_kernels.py` + `_forward_thd_cp` + wrapper boundary wiring, and
**delete** the MLite dense fallback. Size it as a ~1200-line port (Triton layout
kernels are the critical-path risk), CSA-only (sibling DSA out of scope), verified
by the §8 plan under Slurm. This is a faithful port of a real, merged, bounded CP
implementation — no streaming-softmax/unfused-gather substitute, exactly as the
guide requires.

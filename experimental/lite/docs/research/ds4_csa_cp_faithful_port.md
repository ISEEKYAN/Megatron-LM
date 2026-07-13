# DeepSeek-V4 CSA Context-Parallel: Faithful-Port Research (v3)

Status: research deliverable (回炉调研). **v3 closes the five acceptance criteria
explicitly** — adds the authoritative merge commit + author feedback (AC1, §Refs
+ §2e), an exhaustive hack-audit checklist (AC2, §1a), a per-function ownership
table (AC1, §2e), a phased verification *contract* (AC4, §8), and the finalized
English PR#6 non-adoption comment draft (AC5, §10). See the **AC coverage map**
below.

**v2 superseded v1:** v1 read a *stale* vendored Megatron copy and concluded the
cross-CP semantics were "invisible" (an unresolved A/B reading). That conclusion
was wrong: the merged upstream now contains an explicit, fully-materialized,
memory-bounded DS4 CSA CP path. bayan's directive ("你读的参考是过期的——最新
NVIDIA Megatron dev 已 merge 了 DS4 CP") is confirmed by primary source; the A/B
tension is **resolved**.

Scope: (1) characterize the current MLite DS4 CSA CP delivery ("the hack"),
(2) reconstruct the **merged** upstream fused DS4 CSA CP from primary source with
fresh line numbers, (3) specify the faithful-port plan and its real scope.
**Does not** include GPU implementation/verification — that is the downstream
implementation task on `dev-mlite-7-deepseek-v4`.

## AC coverage map (trace each AC → section)

| AC | Requirement (abridged) | Where satisfied |
|----|------------------------|-----------------|
| #1 | Authoritative commit/branch/production call chain; per-function Q/KV ownership, CP collective, indexer top-k, sparse-attn kernel; file:line **and author feedback** | §2 (call chain + line numbers), §2e (per-function ownership table), §References (authoritative commit + author) |
| #2 | reference-vs-port diff audit as a **checklist** of every hack/mis-copy/dead-code/layering violation | §1a (audit checklist table) + §1 (narrative) |
| #3 | Minimal faithful rework spec: copy/adapt/delete functions & deps; preserve CP=1 alignment, position/mask, differentiable collective, fused sparse invariants; no streaming-softmax/60-line-unfused-as-production | §4 (port plan) + §5 (parity preconditions) |
| #4 | Phased verification **contract** (no GPU burn): CPU/static routing, zero-GPU full init, independent Megatron matched-pair, 8-card CP1-vs-CP4 numeric + per-card peak + real fused-kernel hit; GPU budget + pre-GPU moe gate left to bayan | §8 (phased contract with per-phase evidence + pass criteria) |
| #5 | Finalize the **English** PR#6 non-adoption comment draft; separate its dense-fallback mitigation value from the correct fused DS4 CP architecture; no internal task id | §10 (PR#6 comment draft, ready to paste) |

## References

- **Upstream faithful reference**: `NVIDIA/Megatron-LM` branch `dev`, tip
  `fd1121b8f` (fetched 2026-07-13; commit date 2026-07-09). Files under
  `megatron/core/transformer/experimental_attention_variant/`:
  `csa.py`, **`csa_cp_utils.py` (NEW)**, **`csa_cp_layout_kernels.py` (NEW)**,
  `deepseek_v4_hybrid_attention.py`, `dsa.py`, `dsa_kernels.py`.
  The two NEW modules are exactly the merged CP path that v1's stale reference
  lacked; upstream also ships unit tests `test_csa_cp_utils.py`,
  `test_csa_cp_layout_kernels.py`.
- **Authoritative merge commit (AC1 anchor)**: the DS4 CSA CP path was landed by
  **`bfa33263c`** — *"[dev] [DeepSeek-v4] Context Parallel support (#5087)"*,
  authored by **Kunlun Li `<kunlunl@nvidia.com>`, 2026-07-03**, now contained in
  `dev` tip `fd1121b8f`. This single commit adds `csa_cp_utils.py`,
  `csa_cp_layout_kernels.py`, `_forward_thd_cp` in `csa.py`, the wrapper boundary
  wiring in `deepseek_v4_hybrid_attention.py`, and the two unit tests — i.e. the
  entire faithful-port surface is one authored, reviewed, upstream-merged PR, not
  scattered speculative code.
- **Author's stated design ownership (author feedback, AC1)**: the
  `csa_cp_utils.py` module docstring states verbatim — *"MCore-facing utilities
  for the DSv4 THD context-parallel path. This module owns CP row mapping,
  boundary exchange, compressor-input layout, and indexer top-k metadata. It
  reuses MCore's fused MLA RoPE and calls the retained compaction kernel;
  `csa.py` calls final-index lowering directly."* This is the author's own
  statement that cross-CP work lives in the utils/layout layer (not the sparse
  kernel), corroborating §3's resolution of the A/B tension. Inline author
  comments confirm the memory-bounding intent, e.g. `csa_cp_utils.py` L244–246:
  *"A compressed group belongs to the rank containing its last token … its
  fixed-capacity rank-major slot follows directly; no (seq, comp, valid) tensors
  or repack …"* — a deliberate fixed-capacity (bounded) layout, not a dense
  gather.
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

### 1a. Hack audit checklist (AC2 — reference-vs-port diff, exhaustive)

Every defect on the MLite CSA CP path, classified. "Ref" = the merged upstream
faithful counterpart the port must restore.

| # | Class | MLite site (file:line) | Defect | Upstream faithful counterpart (ref) | Port action |
|---|-------|------------------------|--------|-------------------------------------|-------------|
| H1 | **Mis-copy (algorithmic)** | `csa.py::forward` CP>1 branch, L342–464 | Builds an `(L_local × S_total)` dense score matrix via `einsum` + `masked_fill(-inf)` + `torch.softmax` — a dense masked-softmax that discards the sparse top-k structure | `csa.py::_forward_thd_cp` L2436–2554: sparse attention over top-k rows of a *bounded* `kv_full_thd`, no dense score matrix | **Delete** dense branch; port `_forward_thd_cp` |
| H2 | **Mis-copy (memory)** | `cp.py::iter_cp_sources` L13 → `_all_gather_cp` L9–10, used by `csa.py` CP>1 | `all_gather` of the **full dense KV** (`O(S)` rows/rank) across all CP ranks | Bounded sources: `boundary_kv` (`d_window`) + `kv_local` (`S/cp`) + **compressed-space** rank-major all-gather (`S/ratio`) — `csa.py` L2436–2450, `csa_cp_utils.py` gathers | **Delete** full-KV gather usage from CSA path; port bounded exchange (`csa_cp_utils.py`) |
| H3 | **Missing capability (漏抄) / dead gate** | `csa.py::_forward_fused_dsa_cp1` L586–587 | Hard `raise NotImplementedError("… supports CP=1 only.")` — the fused path is fenced off from CP>1 entirely | Upstream has a first-class CP>1 fused path (`_forward_thd_cp`), unit-tested | **Remove** the `NotImplementedError` guard once `_forward_thd_cp` is ported |
| H4 | **漏抄 (whole modules absent)** | MLite has **no** `csa_cp_utils.py`, **no** `csa_cp_layout_kernels.py`, **no** `_forward_thd_cp`, **no** boundary wiring in the DS4 attention wrapper | The entire bounded-CP mechanism (boundary P2P exchange, compressor-input layout, indexer-topk metadata, final-index lowering Triton kernels) is simply not ported | `csa_cp_utils.py` (≈399L), `csa_cp_layout_kernels.py` (≈840L), `_forward_thd_cp` (≈280L), wrapper L267–325 | **Port** all four |
| H5 | **漏抄 (preconditions)** | MLite CP>1 path silently accepts any layout | No enforcement of `qkv_format=='thd'`, contiguous CP partition, `bsz==1`, `local_rows≥d_window` | Wrapper L267–271; `_forward_thd_cp` L2364–2367; `_LeftBoundaryExchange.forward` L134–138 | **Port** the precondition guards (§5) |
| H6 | **Dead-code risk (post-fix)** | `cp.py::iter_cp_sources` / `_all_gather_cp` after H1/H2 removal | Once CSA stops using them, they may be unreachable on the production DS4 path (test-only supply ≠ live) | — | **Audit reachability**; if the sibling `dsa.py` still uses them keep, else remove (see §6) |
| L1 | **Layering violation (candidate, verify at impl)** | `csa.py` CP>1 branch reaching into `cp.py` generic all-gather to reconstruct dense KV | A *primitive/attention* module encoding a full-sequence gather policy inline instead of delegating bounded CP layout to a dedicated utils/kernel layer | Upstream cleanly separates: `csa.py` owns dispatch + sparse call; `csa_cp_utils.py`/`csa_cp_layout_kernels.py` own CP layout/collectives | Port restores the separation (utils/layout modules own CP); **confirm no primitive→app leakage** during impl review |

Notes on H3/H4: these are the "漏抄" (failed-to-copy) core — the port is defined
by restoring H4's four absent pieces and lifting H3's gate. H1/H2 are the active
mis-copies to **delete**. L1 is flagged as a *candidate* layering finding to be
confirmed against the MLite skills `primitive`/`model-compose` invariants at
implementation-review time (not asserted here as a proven violation).

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

### 2e. Per-function ownership (AC1 — Q/KV ownership · CP collective · indexer top-k · sparse kernel)

| Function (file:line) | Q ownership | KV ownership | CP collective | Indexer top-k | Sparse-attn kernel |
|----------------------|-------------|--------------|---------------|---------------|--------------------|
| `deepseek_v4_hybrid_attention.py::forward` L222 | local Q rows (`S/cp`) | local KV + requests `boundary_kv` via KV proj (L296–300) | calls `exchange_cp_boundary_hidden` (P2P) L276–281 | — | — (delegates to `core_attention`) |
| `csa_cp_utils.py::_LeftBoundaryExchange` L123–187 | — | moves only left-boundary window rows | **P2P `batch_isend_irecv`** (prev rank), grad scatter in backward | — | — |
| `csa_cp_utils.py::exchange_cp_boundary_hidden` L190–201 | — | `d_window=max(csa_window_size,d_comp)` bounded window | wraps the P2P exchange | — | — |
| `csa_cp_utils.py::prepare_cp_compressor_input` L209–273 | — | builds fixed-capacity compressor input (local+boundary), returns `seq_to_rank_row` | Triton `CompressorInputCompact` (layout kernel) | — | — |
| `csa_cp_utils.py::compute_cp_indexer_topk` L276–399 | local Q | full **compressed** K (`S/ratio`) | consumes rank-major gathered compressed indexer-K | **produces local-Q / full-compressed-K top-k ids** | — |
| `csa.py::_forward_thd_cp` L2279–2557 | local Q (`global_start=cp_rank*l_local`) | `kv_full_thd = cat(boundary_kv, kv_local, compressed_kv_rank_major)` L2450 | **compressed-space rank-major all-gather** (`gather_from_sequence_parallel_region`) L2410, L2436 | calls `compute_cp_indexer_topk` L2421–2433 | `dsa_sparse_attn` (fused) L2544 / `unfused_compressed_sparse_attn` L2554 |
| `csa_cp_layout_kernels.py::build_attention_indices` L751 | — | lowers logical top-k ids → physical rows of `kv_full_thd` | — (Triton, on-device) | consumes top-k ids | feeds the sparse kernel |

Key invariant across the table: **Q stays local on every rank; the only cross-CP
data movement is the bounded boundary window (P2P) and the compressed-space
(`S/ratio`) rank-major all-gathers.** No function all-gathers full dense KV; no
function forms a dense score matrix. That is the exact property the MLite hack
violates (H1/H2 above).

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

## 8. Phased verification contract (AC4 — no GPU burn in this task)

This is a **contract**, not a run log: it defines, per phase, the exact evidence
artifact and the pass criterion the implementation task must produce **before**
the next phase is unlocked. Phases P0–P2 are zero/low-cost and gate the GPU
phases P3–P4. **GPU budget and the pre-GPU review.moe gate are explicitly left to
bayan's signoff (see P-gate below); nothing in this task consumes GPU.**

| Phase | Cost | What it exercises | Evidence artifact (required) | Pass criterion |
|-------|------|-------------------|------------------------------|----------------|
| **P0 — CPU / static routing** | CPU, minutes | The dispatcher actually routes `thd + cp_size>1 → _forward_thd_cp` and *not* the deleted dense branch | Unit/CPU test that constructs a CP>1 THD config and asserts (a) `_forward_thd_cp` is entered (monkeypatch/spy on the entry), (b) `iter_cp_sources`/dense-softmax is **never** called, (c) the `NotImplementedError` guard is gone. Capture rc + assertion output. | Spy shows `_forward_thd_cp` hit; dense-branch spy count == 0; import of the ported `csa_cp_utils`/`csa_cp_layout_kernels` succeeds |
| **P1 — Zero-GPU full init** | CPU (no CUDA kernels), minutes | The full model/attention init chain constructs with CP>1 config without touching CUDA — config parse → module build → process-group plumbing | CONFIG_ONLY run that walks the *complete* init chain (not just the changed function): build the DS4 attention module with `cp_size=4`, assert boundary-exchange/process-group wiring is constructed, no dense fallback objects allocated. rc + log. | Full init chain returns rc=0; CP wiring present; no `NotImplementedError`; no dense-KV buffer allocation on the path |
| **P2 — Independent Megatron matched-pair (kernel/util level)** | CPU or 1-GPU, low | The ported `csa_cp_utils`/`csa_cp_layout_kernels` match upstream **bit-for-bit** at the op level, isolated from the full model | Port-vs-upstream matched-pair unit tests: reuse upstream `test_csa_cp_utils.py` / `test_csa_cp_layout_kernels.py` against the MLite port; same inputs → compare `seq_to_rank_row`, boundary rows, top-k ids, `build_attention_indices` output. rc + max-abs-diff. | Matched-pair diffs within tolerance (integer maps **exact**; float within upstream test tol); upstream tests pass unmodified against the port |
| **P3 — 8-card CP1-vs-CP4 numeric + per-card peak** | GPU (Slurm) — **bayan-gated** | End-to-end numeric equivalence and the memory-factor claim | Slurm job (real job id + `sacct` rc=0, non-skip): (a) CP1 vs CP4 forward output max-abs/rel diff; (b) **per-card** `torch.cuda.max_memory_allocated` for CP1 and CP4; (c) plot/table of the CP-factor memory reduction | Numeric diff within tol; per-card CP4 peak KV working set scales ≈ `d_window + S/cp + S/ratio` (bounded), i.e. materially below CP1 and below the hack — the CP-factor curve **truly drops** |
| **P4 — real fused-kernel hit** | GPU (Slurm) — **bayan-gated** | The fused sparse kernel actually ran (not the unfused fallback, not a skip) | Slurm job evidence: (a) fused path counter/marker (`dsa_sparse_attn` invoked, e.g. env/log marker analogous to `DS4_MLITE_SM100_FORWARD_OK`), (b) fused-vs-unfused within MLite (`dsa_sparse_attn` vs `unfused_compressed_sparse_attn`) diff, (c) `_LeftBoundaryExchange.backward` grad-scatter + indexer-loss (`reduce_group=cp_group`) finite & correct | Fused marker present (skip ≠ pass); fused-vs-unfused within tol; backward grads finite and match reference |

**P-gate (bayan signoff, not executed here):**
- The **GPU budget** for P3+P4 (node count, wall-clock, GB200 vs H100) is a bayan
  decision; this report does not pre-commit it. Sizing note: DS4 forward smoke
  precedent is 4×GB200 (memory `K-0161`, jobs 4352216/4486073); an 8-card CP4
  proxy is the smallest config exercising a real `cp_size=4` group.
- The **pre-GPU review.moe gate** must pass **after P0–P2 evidence exists and
  before any P3 job is submitted** (per bayan 2026-07-11 v2: two moe gates,
  pre-GPU + pre-Done; two-round cap). This research task marks "待 moe 门"; it
  does **not** self-authorize GPU.

**Reproduce-first discipline (memory `prefer-light-verification`)**: P0–P2 must be
exhausted and green before any GPU minute is spent; a failure in P0–P2 is cheaper
to fix than a burned Slurm job. Every GPU job follows the 5-min-py-spy GPU铁律 and
records a job-id/RUNNING/first-diagnosis ledger.

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

---

## 10. PR#6 non-adoption comment draft (AC5 — English, ready to paste)

Finalized under the current ruling: the correct fix is a faithful port of the
**merged upstream fused/bounded DS4 CSA CP** (§2/§4), *not* PR#6's streaming
softmax and *not* the earlier "port the 60-line unfused gather" idea. The draft
below separates (a) PR#6's genuine mitigation value for the wrong dense fallback
from (b) the correct fused DS4 CP architecture, and carries **no internal task
identifiers**. This **supersedes** the earlier draft
`experimental/lite/docs/reviews/ds4-csa-cp-memory-pr-comment-draft.md`, which was
written before the merged upstream CP path was located.

> Draft only — for maintainer review before posting. Post as a maintainer
> comment; do not merge.

---

Thanks for tackling the long-context OOM here, and for the parity tests — the
online-softmax forward with backward recompute is correct in isolation and the
effort is appreciated.

We're going to hold off on adopting this, because it mitigates a symptom of a
mis-ported attention path rather than the root cause. The CP branch this PR
optimizes builds a full dense `[B, H, Sq, Skv]` score matrix and applies the
sliding-window / indexer top-k only as `-inf` masks — so the indexer top-k never
actually shrinks the working set. That dense materialization is the real OOM
source; chunking the query dimension bounds *peak* memory but leaves compute at
`O(S²)` and adds a large hand-rolled autograd surface. CP then only relieves this
linearly (`O(S²/cp)`), which is why long context still blows up.

The upstream DeepSeek-V4 CSA context-parallel path (merged into `dev`) is
memory-bounded by construction and does not need a dense score matrix at all. It:

- exchanges only a fixed **left-boundary window** of KV rows between adjacent CP
  ranks via a bounded P2P send/recv (window size independent of sequence length);
- all-gathers KV **in compressed space** (`S / compress_ratio` rows), rank-major,
  never the full dense KV;
- runs the sparse attention kernel over the per-query top-k rows of the resulting
  `cat(boundary_kv, local_kv, compressed_kv)` working set.

Per-rank memory is therefore `≈ window + S/cp + S/ratio`, and attention compute is
`O(Sq · topk)` — both sequence-length-bounded, so CP scales memory down by the CP
factor instead of merely flattening a peak. Our plan is to align our port to that
upstream path, which removes the OOM at the root.

That said, the streaming-softmax technique here retains value for the narrow cases
where a genuinely dense score matrix is unavoidable — e.g. an explicit dense-mode
/ full-mask path, or high-compression "attend-to-all" layers where `topk ≈ full`.
We'd be glad to keep it as an opt-in for those, decoupled from the default sparse
CP path.

Minor, as the review bot noted: the `torch.equal` on GPU tensors in the forward
forces a device→host sync; worth dropping or guarding behind a debug flag.

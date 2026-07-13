# DeepSeek-V4 CSA Context-Parallel: Faithful-Port Research

Status: research deliverable (回炉调研). Scope: characterize the current MLite
DS4 CSA CP delivery, reconstruct Megatron's fused DS4 CP mechanism from primary
source, and specify the faithful-port plan. **Does not** include GPU
implementation/verification — see "Structural constraints" and "Open reconciliation".

References are line numbers on branch `dev-mlite-7-deepseek-v4`:
- Megatron reference: `megatron/core/transformer/experimental_attention_variant/csa.py`
  and `.../deepseek_v4_hybrid_attention.py`, `.../dsa_kernels.py`.
- MLite port: `experimental/lite/megatron/lite/primitive/modules/attention/csa.py`
  (CSA), `.../attention/dsa.py` (sibling DSA), `.../attention/cp.py` (CP helpers).

---

## 1. What the current MLite CSA delivers (the "hack")

`experimental/lite/.../attention/csa.py`, class forward dispatch (L284–341):

- `cp_size == 1` → fused sparse kernels:
  - `_forward_fused_dsa_cp1` (L574) calls `dsa_kernels.fused_indexer_sparse_attn`
    (training) / `indexer_topk` + `dsa_sparse_attn` (eval). This is a real fused
    DS4 sparse path and is **correct and kept**.
  - `_forward_fused_sparse_no_indexer_cp1` (L502) for the no-indexer case.
- `cp_size > 1` → **dense masked-softmax fallback** (L342–484):
  - `iter_cp_sources` (from `cp.py`) does `all_gather` of the **full dense KV**
    across every CP rank (`_all_gather_cp`, `cp.py` L9),
  - then builds `dense_scores` via `torch.einsum` + `masked_fill(-inf)` +
    `torch.softmax` over the **entire gathered sequence** (L342–464).
  - `_forward_fused_dsa_cp1` additionally **hard-raises**
    `NotImplementedError("DeepSeek V4 fused DSA path currently supports CP=1 only.")`
    (L586–587).

Consequence (matches memory `ds4-csa-cp-dense-miscopy` and the 128-card resync
OOM): under CP>1 the sparse structure is discarded and the path materializes an
`O(S²/cp)` dense score matrix + a full-sequence KV all-gather, defeating the
memory saving CP is supposed to buy. bayan's characterization ("丢了 fused DS4
CP，换成 all-gather 全量 KV + dense fallback") is confirmed by primary source.

MLite skills read: this violates **perf.fusion** ("fusion must not replace
precision validation" — here fusion is entirely *dropped* for CP>1) and the
**primitive.contract** `what_must_match_reference` invariant (the primitive no
longer matches the Megatron sparse semantics it claims to port).

---

## 2. What Megatron's fused DS4 CP actually does (primary source)

Reconstructed call chain for `DSv4HybridAttention.forward`
(`deepseek_v4_hybrid_attention.py` L222):

1. `get_query_key_value_tensors` (L511) → `qkv_up_proj_and_rope_apply` (L608):
   produces `query`, `key`, `value` that are the **local CP-shard** tensors.
   CP enters **only** through RoPE: `apply_rotary_pos_emb(..., cp_group=self.pg_collection.cp)`
   (L631/652 fused, L685/703 unfused) — i.e. RoPE position indices are corrected
   for the CP-sharded (zigzag) layout, nothing else.
2. `core_attention(...)` = `CompressedSparseAttention.forward` (`csa.py` L905):
   - `_build_kv_full` (L654) = `torch.cat([kv, compressed_kv])` on the **local**
     `kv` — **no cross-CP gather** (L942, L667–677).
   - dispatch (L964–975) to fused paths: `_forward_fused_indexer_training`
     (`fused_indexer_sparse_attn`), `_forward_fused_indexer_inference`
     (`indexer_topk` + `dsa_sparse_attn`), `_forward_fused_no_indexer`, or the
     unfused `unfused_compressed_sparse_attn` (L779). **None** pass a `cp_group`
     to the sparse kernels.
3. Inverse RoPE after attention, again CP only via `cp_group` (L365/380).

**Verified by exhaustive grep**: there is **no `all_gather` / cross-CP KV gather
anywhere** in `csa.py` or `deepseek_v4_hybrid_attention.py`, and **no `cp` /
`all_gather` / `ring`** references in `dsa_kernels.py`. Megatron's "fused DS4 CP"
= *the fused sparse kernels operating on the local shard, with CP applied purely
at the RoPE-position level.* It never builds a dense score matrix and never
gathers full dense KV.

---

## 3. The primary-source tension the guide must reconcile

The guide's premise is "faithfully copy Megatron's already-implemented fused DS4
CP." But the primary source shows Megatron's fused path, under CP>1:

- applies CP **only** to RoPE positions, and
- runs the sparse kernel on **local** `kv_full` with **no cross-CP KV exchange**.

For standard causal attention with sequence-sharded CP, a query on rank *r* must
attend to compressed/window KV that lives on *other* ranks. Megatron's visible
code performs no such cross-rank KV movement in this attention module and no CP
comm inside `dsa_kernels`. Two readings are possible and the code alone cannot
decide between them:

- **(A) Cross-CP KV movement lives below the visible layer** — e.g. inside the
  compiled `dsa_kernels` (ring/all-gather not expressed in Python), or Megatron's
  CP framework feeds this attention a non-sequence-sharded input. If so, a
  faithful MLite port needs that same kernel-level capability, which the MLite
  `dsa_kernels` (`experimental/lite/.../primitive/kernels/dsa_kernels.py`) must be
  confirmed to have. This is the load-bearing unknown.
- **(B) Megatron's fused DS4 path is not yet a correct cross-CP sparse attention**
  (RoPE-CP threading is defensive; true CP>1 sparse correctness is unproven in
  this file). If so, "faithfully copying" it yields a path that is *fused* but not
  *correct* for CP>1 — and MLite's dense-reconstruct fallback is in fact the
  house **correctness-first** convention (see §4), not a botched copy.

This tension is exactly why the task is 回炉调研: it must not be forced through
by writing kernel CP code that cannot be verified here (§5).

---

## 4. MLite house convention: the sibling DSA path does the *same* dense fallback

Important context for the "hack" label. The sibling `DynamicSparseAttention`
(`dsa.py` L323, docstring "Correctness-first DSA attention path") handles CP>1 by
the **identical** strategy the CSA path is being faulted for:

- `_gather_cp_inputs` → `_forward_dense_full` → `zigzag_slice_for_cp` (L432–470),
- i.e. all-gather the full inputs across CP, compute the **dense** full attention,
  then slice the local shard back out.

So MLite's established, reviewed convention for CP>1 sparse attention is
gather-full + dense-reconstruct. This aligns with knowledge `K-0141`
("若对齐路径实际走 dense 重建而非 fused 稀疏路径，结论只能声明 dense-vs-dense")
and `K-0002` (route DSA through the shared primitive + core fused path). The CSA
CP>1 fallback is therefore not an isolated mistake by one author — it is the
same correctness-first pattern the whole MLite sparse-attention family uses.
bayan's ruling changes that policy **for the DS4/CSA path specifically**; the
faithful port must replace the convention, and by symmetry the sibling DSA path
faces the same question.

---

## 5. Structural constraints (why this is a report, not a verified fix)

1. **Code not in this worktree.** This task's branch
   `feature/megatron-fused-ds4-cp-mlite-hack` contains only the old
   "Keep only experimental lite" snapshot: **no** `deepseek_v4/`, **no** CSA
   port, **no** Megatron reference. All DS4 code lives on `dev-mlite-7-deepseek-v4`
   (the branch the cancelled sibling TASK-1.2.12.1 named as the landing target).
   A faithful *code* port cannot be authored/committed meaningfully here.
2. **Verification needs GPU + Slurm on the real branch.** The AC ("CP1 vs CP4
   memory curve真降 + Megatron parity") requires multi-GPU Slurm runs
   (task profile GPU rule) on `dev-mlite-7-deepseek-v4`. Not reproducible in this
   worktree.
3. **Load-bearing unknown (§3).** The faithful-port shape depends on whether MLite
   `dsa_kernels` can do cross-CP sparse attention the way Megatron relies on — an
   unresolved fact that determines feasibility.

---

## 6. Faithful-port plan (conditional on §3 reconciliation)

Assuming reading (A) — Megatron's kernels/layout carry cross-CP semantics and the
MLite kernels can match:

1. **Delete the dense fallback**: remove `csa.py` L342–484 (dense masked-softmax)
   and the `iter_cp_sources` full-KV gather usage; drop `_gather_cp_sources`.
2. **Make the fused path CP-aware exactly like Megatron**: unify the `cp_size==1`
   and `cp_size>1` cases into one fused dispatch. Thread `cp_group`/`cp_rank`/
   `cp_size` through RoPE only (mirror `_apply_rope(..., cp_group=...)` and the
   `apply_rotary_pos_emb(cp_group=...)` calls in the Megatron reference); keep
   `kv_full` local; call `fused_indexer_sparse_attn` / `indexer_topk` +
   `dsa_sparse_attn` unchanged, **provided** those MLite kernels supply the same
   cross-CP behavior Megatron's do.
3. **Remove the `NotImplementedError` CP-1-only guard** (L586–587) once (2) holds.
4. **primitive.contract compliance**: document `process_groups_or_device_placement`
   (which tensors stay sharded, where cp_group enters), `what_must_match_reference`
   (bitwise/near-bitwise vs Megatron fused), `forward_backward_update_details`
   (indexer loss + backward under CP), and `failure_modes`.

If reading (B) holds, the faithful port is **not** yet definable — escalate to
re-scope (either fix Megatron's fused CP first, or keep correctness-first dense
reconstruct as the sanctioned interim and only tighten the numerical claim).

---

## 7. Verification plan (for the implementation task on `dev-mlite-7-deepseek-v4`)

Per `K-0141`, `K-0053`, and the GPU双闸 rules:

1. **Memory-factor AC**: CP1 vs CP4 activation-memory curve on ≥8 GPU (or
   physically-justified proxy), showing real per-rank reduction by ~CP factor —
   the dense fallback cannot pass this by construction.
2. **Parity AC**: fused CP4 vs Megatron fused CP4 on the same config; if the
   comparison path is dense reconstruct, the result may only be claimed as
   *dense-vs-dense* bitwise, **not** as fused-precision proof (`K-0141`).
3. **Non-determinism**: run-to-run accept-with-proof + fused-vs-unfused on the
   deterministic path for the certifiable portion (`K-0053`).
4. **Gate discipline**: CONFIG_ONLY init-chain gate + 8-card proxy before any
   large job; py-spy within 5 min of RUNNING (GPU铁律).

---

## 8. Recommendation

Deliver this research + reconcile §3 with bayan before opening an implementation
task on `dev-mlite-7-deepseek-v4`. The single blocking question: **does Megatron's
fused DS4 CP achieve cross-CP sparse correctness inside `dsa_kernels`/CP-layout
(reading A), and do the MLite `dsa_kernels` provide the same — or is the fused CP
RoPE-only and cross-CP correctness still open (reading B)?** The answer sets
whether the faithful port is a ~30-line dispatch unification (A) or a re-scope (B).

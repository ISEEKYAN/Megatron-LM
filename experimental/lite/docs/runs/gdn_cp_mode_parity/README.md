# Qwen3.5 GatedDeltaNet `gdn_cp_mode` parity proxy

Answers TASK "qwen3.5 gdn_cp_mode=shared 精度问题": is the `sharded` CP execution
strategy numerically consistent with the default `replicated` strategy and with
CP-off, for the native Qwen3.5 GatedDeltaNet primitive?

## Mode mapping (aligned with `dead_ends/cp.md` history)
| current `gdn_cp_mode` | strategy | history |
| --- | --- | --- |
| `replicated` (default) | all-gather full seq → compute full → zigzag-slice | correctness-first, ex-`legacy_full_gather` |
| `sharded` | zigzag→contiguous swap → FLA `cp_context` ring → swap back | ex-`fla_allgather`, C2 state-passing family |

Task phrase "默认 cp vs shared cp" = `replicated` vs `sharded`.

## What the harness measures
`tests/smoke/primitive/gdn_cp_mode_parity.py` builds a tiny **proxy** GDN
(REAL head dims `dk=dv=128`, `conv_kernel=4`; only head COUNT + seq truncated,
because dim=4 probes are non-representative per K-0125). Input is **SBHD**
`x = [seq, batch, hidden]` (the primitive's native layout — `in_proj(x)` then
`transpose(0,1)`), zigzag-sharded along the seq dim. Per CP size in `{2, 4}`:

1. CP-off reference: identical-weight `cp_size==1` module on the FULL sequence
   (rank 0) — the baseline. Weights are randomized (incl. `A_log`/`dt_bias`, so
   gating is non-trivial) and broadcast so every rank/mode shares them bitwise.
2. `replicated` and `sharded` at that CP size, each rank fed its zigzag shard.
3. Per-tensor diff (`max_abs` / `max_rel`) of:
   - forward **output** vs the zigzag slice of the reference output,
   - **input-activation grad** vs the zigzag slice of the reference input grad,
   - **weight grads** — CP-all-reduced (sum over the CP group) then compared to
     the reference weight grads (worst tensor reported).
4. `localize_sharded`: taps the sharded module's per-stage tensors
   (`swap_in` / `conv` / `rule` / `swap_out`) and diffs the sharded vs replicated
   final output on identical weights+input.

## Results (job `13950929`, 8×H100 single node, COMPLETED rc=0, ~1.4 GPU-h)

`fwd`/`in_grad`/`w_grad` are `max_rel` (bf16). Reference = CP-off full sequence.

| CP | mode | fwd max_rel | in_grad max_rel | w_grad max_rel (worst) |
| --- | --- | --- | --- | --- |
| 2 | `replicated` | **0.000e+00** (bitwise) | 3.7e-3 | 5.7e-3 (`in_proj.linear.weight`) |
| 2 | `sharded` | 7.96e-3 | 7.5e-3 | 1.4e-2 (`in_proj.linear.weight`) |
| 4 | `replicated` | **0.000e+00** (bitwise) | 7.5e-3 | 5.7e-3 (`in_proj.linear.weight`) |
| 4 | `sharded` | 7.96e-3 | 7.6e-3 | 1.98e-2 (`in_proj.linear.weight`) |

`localize_sharded`: `sharded_vs_replicated_out` max_rel 6.9e-3 (CP2) / 8.9e-3 (CP4).

Raw log: `evidence/slurm-13950929.completed.out`.

## Verdict (AC #2 — where does `sharded` deviate, and is it a bug?)

**`replicated` (default) is bitwise-exact in forward** (`max_abs = 0.000e+00`) at
both CP2 and CP4 — its all-gather → full-seq compute → zigzag-slice path adds
**zero** forward error; its ~4–6e-3 grad diffs are ordinary bf16 backward
reassociation noise.

**`sharded`'s deviation is NOT in any gather/split step:**
- The zigzag↔contiguous swaps (`_cp_swap_qkvzba` → `_zigzag_contiguous_chunk_swap`)
  are a **pure chunk-level all-to-all** — slice / cat / `all_to_all`, **zero
  arithmetic** — so they are lossless by construction and cannot introduce numeric
  error.
- The `replicated` all-gather is empirically bitwise-exact (above).
- Therefore the ~8e-3 forward deviation can only originate in the **FLA
  `cp_context` chunked ring recurrence** (`_gated_delta_rule` / `_causal_conv1d`
  with `cp_context`), which is a genuinely different arithmetic order than the
  full-sequence kernel.

**It is the bf16 rounding floor, not a correctness defect:**
- 7.96e-3 max_rel ≈ 2 ULP of bf16 (bf16 eps ≈ 3.9e-3), the same order as
  `replicated`'s own bf16 backward noise.
- The forward deviation is **flat** across CP2 and CP4 (both 7.96e-3). A real
  gather/split indexing bug would give O(1) relative error (wrong tokens) and
  would **grow** with more ring hops; a flat ULP-floor deviation is reassociation
  noise inherent to chunked cross-rank state passing.

**Fix direction:** no code fix is required for correctness — `sharded` is
numerically sound within bf16. If exact `replicated`-parity is ever needed
(e.g. bitwise reproducibility), keep the ring but (a) confirm the FLA
`cp_context` chunk-state accumulator runs in fp32 (state carry is the only place
reassociation compounds), or (b) select `gdn_cp_mode="replicated"` for
correctness-critical runs. The observed ~8e-3 is acceptable for training and is
consistent with K-0125 (real Qwen3.5 GDN CP is numerically fine; earlier "CP
issues" were env/overlay, not the ring math).

## Environment (reuse, do not reinvent — see durable `mlite_env_setup`, K-0123)
- image: `verl.vllm023.sqsh` (qwen3.5 line canonical).
- overlay: `qwen35-cp-overlay-20260613/site` on `PYTHONPATH` + `tvm_ffi/lib` on
  `LD_LIBRARY_PATH` — REQUIRED for `sharded` (FLA/tilelang kernels). Without it
  FLA reports "Please install tilelang"; that is an env gap, NOT a CP-ring hang.
  (Job `13950849` FAILED here for exactly this reason before the overlay was added.)

## Reproduce
`run_gdn_cp_parity.sbatch` is the exact as-run recipe (self-contained; paths point
at the `runtime/gdn-cpmode-parity-1171` workspace on cw lustre). It rsyncs the
mlite worktree into `$R/mlite`, drops the harness at `$R/gdn_cp_mode_parity.py`,
and runs CP2 then CP4 under `torchrun` in one `srun`. Grep the log for
`GDN_CP_PARITY` (matrix), `GDN_CP_LOCALIZE` (stage report), `GDN_CP_PARITY_DONE`.

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
`tests/smoke/primitive/gdn_cp_mode_parity_report.py` builds a tiny **proxy** GDN
(REAL head dims `dk=dv=128`, `conv_kernel=4`; only head COUNT + seq truncated,
because dim=4 probes are non-representative per K-0125) and, per CP size in
`{2, 4}` (world-dependent):

1. CP-off reference: single rank, full sequence (`cp_size==1`) — the baseline.
2. `replicated` and `sharded` at that CP size.
3. Per-tensor diff of forward **output** and **input-activation grad**, each
   sliced to the rank's zigzag shard of the reference (`max_abs` / `max_rel`).
4. Direct `replicated`-vs-`sharded` cross diff.

Parameter grads are intentionally NOT diffed (no CP grad all-reduce in this bare
harness → each rank only holds its shard's param grad; not comparable to the
full-seq reference). Output + input-grad are per-token and slice cleanly, so
they localise any gather/split precision defect.

## Environment (reuse, do not reinvent — see durable `mlite_env_setup`)
- image: `verl.vllm023.sqsh` (qwen3.5 line canonical)
- overlay: `qwen35-cp-overlay-20260613/site` on `PYTHONPATH` + `tvm_ffi/lib` on
  `LD_LIBRARY_PATH` — REQUIRED for `sharded` (FLA/tilelang kernels, K-0123).
  Without it FLA reports "Please install tilelang"; that is an env gap, NOT a
  CP-ring hang.

## Run (8-GPU single node, <=25 min, well under 4 GPU-h AC cap)
```bash
sbatch --export=ALL,MLITE_REPO=/abs/path/to/this/worktree \
  experimental/lite/docs/runs/gdn_cp_mode_parity/run_gdn_cp_mode_parity.sbatch
```
Report JSON is bracketed by `GDN_CP_MODE_PARITY_REPORT_BEGIN/END` in the job log.

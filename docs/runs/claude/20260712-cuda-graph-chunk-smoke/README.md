# Chunk-wise CUDA Graph qualification smoke

Qualifies the delivered CPU spine on real H100 + Transformer Engine — the one
region the CPU box cannot reach (TE `make_graphed_callables` capture).

## Stages
- **A — unit tests in-container**: the CG unit tests run inside the canonical
  TE-bearing container (catches lazy-import breakage the CPU box would miss).
- **B — capture/replay (primary GPU evidence)**: a real `te.pytorch.Linear` is
  captured through `CudaGraphController._capture_slot` and replayed via
  `get_graphed`; replayed forward must match eager. Writes
  `stage_b_capture_replay.json` + `NON_SKIP_..._PASSED` on pass.
- **C — 8-card qwen3.5 FB + qualify probe (best-effort)**: proven `correctness.py`
  harness runs an 8-GPU forward-backward with `MLITE_CG_QUALIFY_PROBE=1`, so
  `qualify()` runs over the real runtime-built chunks. Non-fatal (A+B gate rc)
  because full-model build carries TE/mcore version-skew risk this smoke
  does not attempt to pin.

## Proven CW assets
- container: `.../code/verl_optimize/verl.vllm016.dev.qwen3_5.sqsh`
- deps/megatron.core: `.../code/megatron_lite/hy3-validation-bcf0dc22b` (`devlite-venv`)
- HF ckpt: `.../code/qwen35_dapo_mfsdp_62295f9b3/Qwen3.5-35B-A3B-nine-layer`

## Evidence — job 14277692 (2026-07-23, commit e2fcf140f)
- Workdir: `.../work/cg-smoke-e2fcf140f`
- `sacct`: COMPLETED `0:0`, Elapsed `00:01:18`, Start `2026-07-23T03:02:43`
- First diagnosis (~30s after RUNNING): H100×8 visible, container still
  starting (util 0%); log grew within 1 min — not hung, no scancel.
- Verdict: `rc_a=0 rc_b=0 rc_c=1` (A+B gate → job rc 0)
- Stage B: `allclose=true`, `max_abs_diff=0.0`, device=`NVIDIA H100 80GB HBM3`,
  marker `NON_SKIP_CUDA_GRAPH_CAPTURE_REPLAY_PASSED` present
- Stage C: ImportError `moe_permute_and_pad_with_probs` missing from container
  TE (`moe_permute_with_probs` only) — env skew, best-effort non-gate; fix in
  env/overlay layer, not a repo shim
- Artifacts: `evidence-14277692/`

Prior failure 13875020: Stage B `KeyError:0` from `num_warmup_iters=0`
(TE never populates `need_bwd_dw_graph`); fixed in e2fcf140f (default warmup=3).

## Invoke
```sh
sbatch --export=ALL,MYSRC=<staged>/experimental/lite,OUTPUT_ROOT=<out> smoke.sbatch
```

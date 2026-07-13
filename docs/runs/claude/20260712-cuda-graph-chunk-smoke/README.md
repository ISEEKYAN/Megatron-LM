# Chunk-wise CUDA Graph qualification smoke (TASK-1.21)

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
  because full-model build carries megatron.core version-skew risk this smoke
  does not attempt to pin.

## Proven CW assets (2026-07-12)
- container: `.../code/verl_optimize/verl.vllm016.dev.qwen3_5.sqsh`
- deps/megatron.core: `.../code/megatron_lite/hy3-validation-bcf0dc22b` (`devlite-venv`)
- HF ckpt: `.../code/qwen35_dapo_mfsdp_62295f9b3/Qwen3.5-35B-A3B-nine-layer`

## Invoke
```sh
sbatch --export=ALL,MYSRC=<staged>/experimental/lite,OUTPUT_ROOT=<out> smoke.sbatch
```

Evidence (job id, sacct rc, verdict.txt, stage_*.json) is logged to the task.

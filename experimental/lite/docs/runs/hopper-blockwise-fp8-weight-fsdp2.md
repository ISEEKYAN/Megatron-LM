# Hopper blockwise FP8-weight FSDP2 GPU regression

Runs the FP8-weight FSDP2 optimizer regressions on a real Hopper GPU against the
canonical training image's native Transformer Engine 2.15. These cases are
skipped on CPU (they require CUDA + FSDP2 `fully_shard`), so this Slurm run is
the only place they actually execute.

## What it covers

`tests/unit/primitive/test_fsdp2_offload_gpu.py -k fp8_weight`:

- `test_fsdp2_fp8_weight_uses_te_source_for_fp32_master_and_updates` — the FP8
  compute weights are built with TE `quantized_model_init`; the single FP32
  master is taken bit-for-bit from TE's preserved high-precision source (never
  by dequantizing the FP8 param), TE's source is consumed exactly once, and one
  AdamW step runs with a finite loss and grad norm.
- `test_fsdp2_fp8_weight_local_checkpoint_resume_matches_uninterrupted` — save
  after one step, rebuild, resume, and confirm the restored FP32 masters match
  the uninterrupted run bit-for-bit, then match again after the next step.

The FP32 master mirrors the sharded FP8 param as a DTensor (same mesh, placements,
shape, stride), so the master/`exp_avg`/`exp_avg_sq` and the FSDP2 gradient share
one tensor type through the AdamW update.

## Environment

- image: `pytorch_26.04-py3.sqsh` (native TE `2.15.0+42b84005`, CUDA 13.2)
- `PYTHONPATH`: extracted `experimental/lite` + a pure-python pytest target dir +
  the Megatron-LM repo root (namespace package, for `megatron.core` used by the
  checkpoint RNG-tracker path). No FP8-only overlay.
- single H100, `WORLD_SIZE=1` (`fully_shard` still yields DTensor params).

## Accepted result

Slurm job `14237043` on `pool0-01735` `COMPLETED 0:0` (elapsed 0:56). The Hopper
gate reported `block_scaling_supported=True` before allocation. Non-skipped
markers:

```text
_HopperEnvironment(compute_capability=(9, 0), transformer_engine_version='2.15.0+42b84005', cuda_version=(13, 2), cublas_version=130401, block_scaling_supported=True)
2 passed, 2 deselected
MLITE_FP8_WEIGHT_FSDP2_JOB_OK
```

JUnit: `tests=2 failures=0 errors=0 skipped=0`. Artifacts and immutable stdout:

```text
.../work/codex/fp8-primitives-c9c00dfac-cw/results/fp8-weight-fsdp2-14237043
.../work/codex/fp8-primitives-c9c00dfac-cw/fp8-weight-fsdp2-14237043.out
```

FP8-weight throughput A/B (fp8-vs-bf16 speedup, vs-Megatron-FP8 tokens/s) and the
FP8-weight precision-parity profile are the terminal Qwen3-MoE verification and
are run under the sibling terminal-verification task, not here.

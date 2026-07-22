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

## Distributed-optimizer FP8-weight path (fail-loud, no run by design)

The two optimizer paths for the FP8-weight profile are FSDP2 and the distributed
optimizer. Only FSDP2 is a supported FP8-weight training path in this version:
its per-parameter update keeps the FP8 compute weights live. The
distributed-optimizer FP8 write-back instead depends on the upstream Megatron
`fp8_param_gather` path, which is out of scope for the closed Hopper profiles.

A distributed-optimizer run against `hopper_blockwise_fp8_weight` is
*constructible* (the distributed optimizer is qwen3_moe's default), so leaving it
unguarded would silently update only the BF16-gathered parameters while the FP8
compute weights stayed stale — a run that looks FP8-trained but is not. The
precision gate (`require_optimizer_supports_precision`, wired into `build_model`)
therefore rejects the combination loudly. This is the fail-loud behaviour AC#4
requires; there is deliberately **no** distributed-optimizer FP8 Slurm run.

The guard is proven on CPU (no GPU needed — it fires before allocation):

- `tests/unit/model/test_qwen3_moe_precision_static.py::test_qwen3_moe_rejects_fp8_weight_with_distributed_optimizer`
  — `build_model` rejects the default (`dist_opt`) and explicit distributed-optimizer
  cases for the FP8-weight profile.
- `tests/unit/primitive/test_precision_profiles_unit.py::test_fp8_weight_profile_rejects_optimizers_without_fp8_param_gather`
  and `::test_optimizer_precision_gate_allows_supported_combinations` — the
  primitive gate rejects FP8-weight under any optimizer lacking `fp8_param_gather`
  and accepts FP8-weight+FSDP2 and BF16-weight under every backend.

Un-deferring the distributed-optimizer FP8-weight path (i.e. landing
`fp8_param_gather`) is tracked separately in the backlog and is a human decision.

## Terminal verification scope

FP8-weight throughput A/B (fp8-vs-bf16 speedup, vs-Megatron-FP8 tokens/s) and the
FP8-weight precision-parity profile are the terminal Qwen3-MoE verification and
are run under the sibling terminal-verification task, not here.

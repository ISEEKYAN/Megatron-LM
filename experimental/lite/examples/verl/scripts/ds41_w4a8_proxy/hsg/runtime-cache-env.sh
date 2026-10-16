#!/usr/bin/env bash
# /runtime-cache is a bind of the node-local, versioned recipe cache directory.
export DG_JIT_CACHE_DIR=/runtime-cache/deep-gemm
export TRITON_CACHE_DIR=/runtime-cache/triton
export TORCHINDUCTOR_CACHE_DIR=/runtime-cache/torchinductor
export TORCH_EXTENSIONS_DIR=/runtime-cache/torch-extensions
export CUDA_CACHE_PATH=/runtime-cache/cuda
export XDG_CACHE_HOME=/runtime-cache/xdg
export FLASHINFER_WORKSPACE_BASE=/runtime-cache/flashinfer
export CUTE_DSL_CACHE_DIR=/runtime-cache/cute
export TILELANG_CACHE_DIR=/runtime-cache/tilelang
export TILELANG_TMP_DIR=/runtime-cache/tmp/tilelang
export HUMMING_CACHE_DIR=/runtime-cache/humming
export HUMMING_TMP_DIR=/runtime-cache/humming-tmp
export HUMMING_DISABLE_PARALLEL_BUILD=1
export TMPDIR=/runtime-cache/tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export WANDB_DIR=/runtime-cache/wandb
export WANDB_CACHE_DIR=/runtime-cache/wandb-cache
export WANDB_DATA_DIR=/runtime-cache/wandb-data
export WANDB_CONFIG_DIR=/runtime-cache/wandb-config
mkdir -p "$DG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
 "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE" \
 "$CUTE_DSL_CACHE_DIR" "$TILELANG_CACHE_DIR" "$TILELANG_TMP_DIR" \
 "$HUMMING_CACHE_DIR" "$HUMMING_TMP_DIR" \
 "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
 "$WANDB_DATA_DIR" "$WANDB_CONFIG_DIR"
printf 'RECIPE_RUNTIME_CACHE_LOCAL DG=%s TRITON=%s INDUCTOR=%s\n' \
 "$DG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
printf 'RECIPE_RUNTIME_TEMP_LOCAL TMPDIR=%s WANDB_DIR=%s\n' "$TMPDIR" "$WANDB_DIR"

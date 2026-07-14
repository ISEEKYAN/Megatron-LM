#!/usr/bin/env bash
# Node launcher for Qwen3.5 GatedDeltaNet gdn_cp_mode parity proxy.
# Reuses the verl.vllm023 container env recipe (see mlite_env_setup §CW-H100
# and docs/runs/cursor/task-1.13.20) but only needs torchrun + mlite + FLA +
# tilelang overlay -- no Ray / vLLM / verl train loop.
set -euo pipefail

NPROC=${NPROC:-8}
BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}
MLITE_REPO=${MLITE_REPO:?set MLITE_REPO to the checked-out worktree root}
# FLA/TileLang cu12 overlay required for gdn_cp_mode=sharded (K-0123).
CP_SITE=${CP_SITE:-$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site}

export PATH=$(printf %s "$PATH" | tr : '\n' | grep -viE 'miniforge|/conda|/anaconda' | paste -sd: -)
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
unset CUDA_DEVICE_MAX_CONNECTIONS
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CP_SITE:$MLITE_REPO/experimental/lite:$MLITE_REPO:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton-gdncp-${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-0}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-gdncp-${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-0}"
export PYTHONPYCACHEPREFIX="/tmp/pycache-gdncp-${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-0}"
export NCCL_NVLS_ENABLE=0
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX"

mlite_head=$(git -C "$MLITE_REPO" rev-parse HEAD 2>/dev/null || echo missing)
echo "GDN_CP_PARITY_COMMITS mlite=$mlite_head nproc=$NPROC cp_site=$CP_SITE"

# Preflight: confirm FLA + tilelang import (else sharded mode cannot run).
python3 - <<'PY'
import fla, tilelang
print("GDN_CP_PARITY_PREFLIGHT_OK fla", fla.__version__, "tilelang", tilelang.__version__, flush=True)
PY

HARNESS="$MLITE_REPO/experimental/lite/tests/smoke/primitive/gdn_cp_mode_parity_report.py"
echo "GDN_CP_PARITY_LAUNCH nproc=$NPROC harness=$HARNESS"
torchrun --standalone --nproc_per_node="$NPROC" "$HARNESS"
echo "GDN_CP_PARITY_NODE_DONE rc=$?"

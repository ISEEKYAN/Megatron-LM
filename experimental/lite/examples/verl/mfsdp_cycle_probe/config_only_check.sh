#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Zero-GPU CONFIG_ONLY gate for the Branch B probe (TASK-1.13.8.2): inside the
# verl.vllm023 container on a CPU partition, build the runtime config for BOTH
# optimizer backends and import the probe's deps — no CUDA. Catches import /
# config-construction breaks before any GPU is burned (bayan 零-GPU CONFIG_ONLY 闸).
#   srun -A coreai_devtech_all -p cpu_short --container-image=$IMG \
#        --container-mounts=/lustre:/lustre bash <this>
set -euo pipefail

BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}
MLITE=${MLITE:-$BASE/qwen35_dapo_mfsdp_62295f9b3/mlite-133413497}
CP_SITE=${CP_SITE:-$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site}
PROBE_DIR=${PROBE_DIR:-$BASE/branchb-mfsdp/mfsdp_cycle_probe}
MODEL=${MODEL:-$BASE/models/Qwen3-30B-A3B}
CACHE=${CACHE:-$BASE/branchb-mfsdp/cache/config_only}

export PATH="$(printf %s "$PATH" | tr : '\n' | grep -viE 'miniforge|/conda|/anaconda' | paste -sd: -)"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="/vllm:$CP_SITE:$MLITE/experimental/lite/examples/verl:$MLITE/experimental/lite:$MLITE:$PROBE_DIR:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"
export HOME="$CACHE/home" XDG_CACHE_HOME="$CACHE/xdg" HF_HOME="$CACHE/hf"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$HF_HOME"
export CUDA_VISIBLE_DEVICES=""   # force CPU; --config-only must not need a GPU

cd /tmp
for OPT in mfsdp fsdp2; do
  python3 "$PROBE_DIR/mfsdp_cycle_probe.py" --config-only \
    --optimizer "$OPT" --hf-path "$MODEL" --model-name qwen3_moe \
    --truncate-layers 2 --keep-experts 4 --seq-len 512 --skip-load-hf-weights
done
echo "CONFIG_ONLY_GATE_DONE"

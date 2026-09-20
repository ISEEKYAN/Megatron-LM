#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
# Requires an MLite checkout with the deepseek_v41 registry/protocol installed.
export VERL_MLITE_HF_CONFIG_MODEL_TYPE=deepseek_v41
export MLITE_MODEL_NAME=deepseek_v41 MLITE_OPTIMIZER_BACKEND=dist_opt
export TP_SIZE=1 PP_SIZE=1 VPP_SIZE=1 CP_SIZE=1 EP_SIZE=1 ETP_SIZE=1
export NUM_GPUS="${NUM_GPUS:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"
export PARAM_OFFLOAD=False OPTIMIZER_OFFLOAD=False GRAD_OFFLOAD=False
export LR="${LR:-1e-4}"
export MIN_LR="${MIN_LR:-${LR}}"
export WANDB_ENTITY="${WANDB_ENTITY:-megatron-core-moe-dev}"
export PROJECT_NAME="${PROJECT_NAME:-verl-mlite-deepseek-v41-sft}"
export RUN_NAME="${RUN_NAME:-deepseek_v41_sft}" TOTAL_STEPS="${TOTAL_STEPS:-8}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/deepseek_v41_sft}"
exec bash "${SCRIPT_DIR}/run_qwen3moe_sft.sh" \
  '++engine.impl_cfg.optimizer=muon' \
  'trainer.logger=[console,file,wandb]' \
  "+engine.impl_cfg.optimizer_config={lr:${LR},ns_steps:${NS_STEPS:-5},coefficient_type:quintic,clip_grad:${CLIP_GRAD:-1.0}}" \
  "$@"

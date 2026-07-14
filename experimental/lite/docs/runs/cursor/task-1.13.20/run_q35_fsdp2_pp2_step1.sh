#!/usr/bin/env bash
# Qwen3.5 + FSDP2 + PP2 DAPO one-step smoke (TASK-1.13.20).
# Wraps the proven qwen35_dapo_mfsdp fsdp2 harness with PP=2 geometry.
set -euo pipefail

RUN_DIR=${RUN_DIR:-${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}/qwen35_dapo_mfsdp_62295f9b3}
BASE_SCRIPT="$RUN_DIR/run_dapo_h100_fsdp2.sh"
if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "missing base harness: $BASE_SCRIPT" >&2
  exit 2
fi

export BACKEND=${BACKEND:-mlite}
export MLITE_OPTIMIZER=${MLITE_OPTIMIZER:-fsdp2}
export NNODES=${NNODES:-1}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# 8-GPU colocated: TP1 × PP2 × CP1 × EP4
export TP=${TP:-1}
export PP=${PP:-2}
export CP=${CP:-1}
export EP=${EP:-4}
export ETP=${ETP:-1}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_N=${ROLLOUT_N:-4}
export GEN_TP=${GEN_TP:-2}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
export ALL_OFFLOAD=${ALL_OFFLOAD:-True}
export ENFORCE_EAGER=${ENFORCE_EAGER:-True}

product=$((TP * PP * CP * EP))
expected=$((NNODES * NGPUS_PER_NODE))
if (( product != expected )); then
  echo "parallel product mismatch: TP*PP*CP*EP=$product != NNODES*NGPUS=$expected" >&2
  exit 2
fi

export USE_THD=${USE_THD:-False}
export USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
export FILTER_OVERLONG=${FILTER_OVERLONG:-False}

echo "Q35_FSDP2_PP2_STEP1 geo=TP${TP}PP${PP}CP${CP}EP${EP} steps=${TOTAL_TRAINING_STEPS} batch=${TRAIN_BATCH_SIZE} seqlen=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) rollout_tp=${GEN_TP} use_thd=${USE_THD} fused=${USE_FUSED_KERNELS}"

# Extra hydra overrides: PP2+THD 在 job13941926 触发 2206/2213 token 不齐→FSDP2 _local_tensor 次生错误。
exec bash "$BASE_SCRIPT" \
  "actor_rollout_ref.actor.engine.impl_cfg.use_thd=${USE_THD}" \
  "actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}" \
  "data.filter_overlong_prompts=${FILTER_OVERLONG}"

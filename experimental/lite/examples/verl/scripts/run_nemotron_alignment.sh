#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set the full52 BF16 Nemotron checkpoint}"
: "${TRAIN_FILES:?Set the unchanged DAPO parquet path}"
: "${OUTPUT_ROOT:?Set a run directory outside the source tree}"
export VAL_FILES="${VAL_FILES:-${TRAIN_FILES}}"
export NNODES=2 NGPUS_PER_NODE=4
export ACTOR_TP=1 ACTOR_PP=2 ACTOR_CP=2 ACTOR_EP=4 ACTOR_ETP=1
export MLITE_MODEL_NAME=nemotron_h MLITE_IMPL=lite MLITE_OPTIMIZER_BACKEND=dist_opt
export PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-False}"
export GRAD_OFFLOAD="${GRAD_OFFLOAD:-False}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}" ROLLOUT_N="${ROLLOUT_N:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=1
export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=8192
export PPO_MAX_TOKEN_LEN_PER_GPU=20480 ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=20480
export ROLLOUT_TP=1 ROLLOUT_TEMPERATURE=1
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.3}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-128}"
export ROLLOUT_MAX_MODEL_LEN=12288
export SAVE_FREQ="${SAVE_FREQ:--1}" TEST_FREQ="${TEST_FREQ:--1}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VERL_ACTOR_BATCH_INVARIANT="${VERL_ACTOR_BATCH_INVARIANT:-1}"
# This BF16 Nemotron path does not use the DS4 FP8 DeepGEMM backend.
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VERL_ROLLOUT_BATCH_INVARIANT="${VERL_ROLLOUT_BATCH_INVARIANT:-1}"
export VERL_FULL_DETERMINISM="${VERL_FULL_DETERMINISM:-1}"
export VERL_REQUIRE_BITWISE_LOGPROBS="${VERL_REQUIRE_BITWISE_LOGPROBS:-1}"
export NEMOTRON_SHARED_NORMS=1 NEMOTRON_EP_SLOT_DIAGNOSTIC=1
export NVSHMEM_HCA_PREFIX=rocep
export RUN_NAME="${RUN_NAME:-nemotron-native-two-step}"

# Base weights have no chat template: preserve message content, without chat tokens.
template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_qwen3moe_gsm8k_grpo.sh" \
  'data.filter_overlong_prompts_workers=8' \
  'algorithm.rollout_correction.bypass_mode=False' \
  'actor_rollout_ref.actor.use_dynamic_bsz=False' \
  'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False' \
  'actor_rollout_ref.actor.engine.export_dtype=null' \
  '~actor_rollout_ref.actor.engine.grad_offload' \
  '+actor_rollout_ref.actor.engine.full_determinism=True' \
  'actor_rollout_ref.rollout.data_parallel_size=8' \
  'actor_rollout_ref.rollout.expert_parallel_size=8' \
  'actor_rollout_ref.rollout.enforce_eager=False' \
  'actor_rollout_ref.rollout.enable_prefix_caching=False' \
  'actor_rollout_ref.rollout.logprobs_mode=raw_logprobs' \
  'actor_rollout_ref.rollout.full_determinism=True' \
  '+actor_rollout_ref.model.override_config.nemotron_shared_norms=True' \
  '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.nemotron_shared_norms=True' \
  "actor_rollout_ref.model.custom_chat_template=\"${template}\"" \
  "$@"

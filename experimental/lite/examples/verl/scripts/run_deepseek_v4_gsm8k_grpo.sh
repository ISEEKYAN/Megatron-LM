#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
GRPO_RUNNER="${SCRIPT_DIR}/run_qwen3moe_gsm8k_grpo.sh"
DATASET_MODULE="${SCRIPT_DIR}/../verl_mlite/dataset.py"

: "${MODEL_PATH:?set MODEL_PATH to the official mixed DeepSeek V4 checkpoint}"
: "${TRAIN_FILES:?set TRAIN_FILES to a GSM8K-schema training parquet}"
: "${VAL_FILES:?set VAL_FILES to a GSM8K-schema validation parquet}"

export MLITE_MODEL_NAME="${MLITE_MODEL_NAME:-deepseek_v4}"
export MLITE_IMPL="${MLITE_IMPL:-lite}"
export MLITE_OPTIMIZER_BACKEND="${MLITE_OPTIMIZER_BACKEND:-fsdp2}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-fused}"
export USE_FUSED_KERNELS=True
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/ds4_gsm8k_grpo}"
export PROJECT_NAME="${PROJECT_NAME:-verl-mlite-ds4-gsm8k-grpo}"

export ACTOR_TP="${ACTOR_TP:-1}"
export ACTOR_PP="${ACTOR_PP:-4}"
export ACTOR_CP="${ACTOR_CP:-4}"
export ACTOR_EP="${ACTOR_EP:-8}"
export ACTOR_ETP="${ACTOR_ETP:-1}"
export ACTOR_VPP="${ACTOR_VPP:-null}"

export PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"
export OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
export GRAD_OFFLOAD="${GRAD_OFFLOAD:-False}"
export OPTIMIZER_STATE_OFFLOAD_FRACTION="${OPTIMIZER_STATE_OFFLOAD_FRACTION:-1.0}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-128}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-512}"
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-512}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export ROLLOUT_MODE="async"
export ROLLOUT_TP="${ROLLOUT_TP:-16}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-384}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
export TEST_FREQ="${TEST_FREQ:--1}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VERL_VLLM_FP8_QUANT_ENABLED=0
export RUN_NAME="${RUN_NAME:-ds4_gsm8k_grpo_pp${ACTOR_PP}_ep${ACTOR_EP}_cp${ACTOR_CP}_rtp${ROLLOUT_TP}}"

DS4_CHAT_TEMPLATE='{{ bos_token }}{% for message in messages %}'
DS4_CHAT_TEMPLATE+='{% if message["role"] == "user" %}'
DS4_CHAT_TEMPLATE+='{{ "<｜User｜>" + message["content"] }}'
DS4_CHAT_TEMPLATE+='{% elif message["role"] == "assistant" %}'
DS4_CHAT_TEMPLATE+='{{ "<｜Assistant｜>" + message["content"] + eos_token }}'
DS4_CHAT_TEMPLATE+='{% else %}'
DS4_CHAT_TEMPLATE+='{{ raise_exception("unsupported chat role: " + message["role"]) }}'
DS4_CHAT_TEMPLATE+='{% endif %}{% endfor %}'
DS4_CHAT_TEMPLATE+='{% if add_generation_prompt %}{{ "<｜Assistant｜>" }}{% endif %}'
readonly DS4_CHAT_TEMPLATE

exec bash "${GRPO_RUNNER}" \
  "data.custom_cls.path=${DATASET_MODULE}" \
  "data.custom_cls.name=ChatTemplateRLHFDataset" \
  "+data.chat_template='${DS4_CHAT_TEMPLATE}'" \
  "actor_rollout_ref.model.custom_chat_template='${DS4_CHAT_TEMPLATE}'" \
  "+actor_rollout_ref.actor.engine.cross_entropy_fusion=True" \
  "actor_rollout_ref.actor.engine.resync_format=vllm_checkpoint" \
  "+actor_rollout_ref.actor.engine.resync_config.expert_dtype=fp8" \
  "+actor_rollout_ref.actor.engine.impl_cfg.recompute=full" \
  "+actor_rollout_ref.actor.engine.impl_cfg.mtp_enable=True" \
  "+actor_rollout_ref.actor.engine.impl_cfg.mtp_enable_train=True" \
  "actor_rollout_ref.rollout.load_format=dummy" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=verl_mlite.rollout.verl_worker.VllmCheckpointWorkerExtension" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.expert_dtype=fp8" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.activation_scheme=dynamic" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.fmt=e4m3" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.quant_method=fp8" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.scale_fmt=float32" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.weight_block_size=[128,128]" \
  "$@"

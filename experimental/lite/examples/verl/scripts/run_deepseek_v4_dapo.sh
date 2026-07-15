#!/usr/bin/env bash
# DeepSeek-V4 DAPO run (self-contained: DS4 geometry + fp8 resync + DAPO recipe).
#
# DAPO recipe:
#   - clip-higher (asymmetric PPO clip) + dual-clip,
#   - no KL (neither in the reward nor as a loss),
#   - token-mean loss aggregation,
#   - overlong reward shaping (soft length penalty).
# Resync: mlite exports a pre-quantized block-fp8 checkpoint (resync_format=
# vllm_checkpoint, expert_dtype=fp8) that vLLM loads directly via hf_overrides.
# Point MODEL_PATH / TRAIN_FILES / VAL_FILES at the DS4 checkpoint and a
# dapo-math-style parquet; every value below is overridable from the environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
GRPO_RUNNER="${SCRIPT_DIR}/run_qwen3moe_gsm8k_grpo.sh"
DATASET_MODULE="${SCRIPT_DIR}/../verl_mlite/dataset.py"

: "${MODEL_PATH:?set MODEL_PATH to the official mixed DeepSeek V4 checkpoint}"
: "${TRAIN_FILES:?set TRAIN_FILES to a dapo-math-schema training parquet}"
: "${VAL_FILES:?set VAL_FILES to a dapo-math-schema validation parquet}"

export MLITE_MODEL_NAME="${MLITE_MODEL_NAME:-deepseek_v4}"
export MLITE_IMPL="${MLITE_IMPL:-lite}"
export MLITE_OPTIMIZER_BACKEND="${MLITE_OPTIMIZER_BACKEND:-fsdp2}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-fused}"
export USE_FUSED_KERNELS=True
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/ds4_dapo}"
export PROJECT_NAME="${PROJECT_NAME:-verl-mlite-ds4-dapo}"

# Geometry (PP4 x EP8 x CP4, rollout TP8).
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

# DAPO scale.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-10240}"
# Token budgets and rollout model length derive from one full prompt+response
# sequence, so they track the lengths above automatically.
_MAX_SEQ_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-${_MAX_SEQ_LEN}}"
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${_MAX_SEQ_LEN}}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ROLLOUT_MODE="async"
export ROLLOUT_TP="${ROLLOUT_TP:-8}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-${_MAX_SEQ_LEN}}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${_MAX_SEQ_LEN}}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
export TEST_FREQ="${TEST_FREQ:--1}"
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# DAPO drops KL entirely.
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"

# Resync: mlite exports a pre-quantized block-fp8 checkpoint (resync_format=
# vllm_checkpoint, expert_dtype=fp8) that vLLM loads directly via hf_overrides.
# verl's receiver-side quant_weights early-returns for DeepSeek-V4 (it skips
# quantization for DS4 and expects pre-quantized fp8), so VERL_VLLM_FP8_QUANT_
# ENABLED is inert on this path for DS4 -- inherit the sbatch's value instead of
# hard-clobbering to 0, which stays correct for the non-DS4 sibling models.
export VERL_VLLM_FP8_QUANT_ENABLED="${VERL_VLLM_FP8_QUANT_ENABLED:-1}"
export RUN_NAME="${RUN_NAME:-ds4_dapo_pp${ACTOR_PP}_ep${ACTOR_EP}_cp${ACTOR_CP}_rtp${ROLLOUT_TP}}"

# Fail loud if rollout_tp does not evenly divide the model's o_groups.
python - "${MODEL_PATH}/config.json" "${ROLLOUT_TP}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
og, tp = cfg.get("o_groups"), int(sys.argv[2])
assert isinstance(og, int) and og >= 1, f"invalid o_groups={og!r}"
assert og % tp == 0, f"DS4 o_groups={og} must be divisible by rollout_tp={tp}"
print(f"DS4_ROLLOUT_TP_PREFLIGHT_PASSED o_groups={og} rollout_tp={tp}", flush=True)
PY

# Chat template: verl-project's official DeepSeek-V4-Flash default (plain
# content concatenation), overridable via DEEPSEEK_V4_FLASH_CHAT_TEMPLATE.
DEFAULT_CHAT_TEMPLATE='{% for message in messages %}{% if message["content"] is string %}{{ message["content"] }}{% else %}{% for content in message["content"] %}{% if content["type"] == "text" %}{{ content["text"] }}{% endif %}{% endfor %}{% endif %}{% if not loop.last %}{{ "\n\n" }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ "\n" }}{% endif %}'
DS4_CHAT_TEMPLATE="${DEEPSEEK_V4_FLASH_CHAT_TEMPLATE:-${DEFAULT_CHAT_TEMPLATE}}"
readonly DS4_CHAT_TEMPLATE

# DAPO recipe knobs (verl-project recipe/dapo defaults).
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((1024 * 4))}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

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
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=flashinfer_cutlass" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.expert_dtype=fp8" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.activation_scheme=dynamic" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.fmt=e4m3" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.quant_method=fp8" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.scale_fmt=float32" \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.weight_block_size=[128,128]" \
  "actor_rollout_ref.actor.engine.load_hf_weights=${ENGINE_LOAD_HF_WEIGHTS:-True}" \
  "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS:-8}" \
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}" \
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}" \
  "actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}" \
  "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}" \
  "+reward.reward_kwargs.overlong_buffer_cfg.enable=True" \
  "+reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}" \
  "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}" \
  "+reward.reward_kwargs.overlong_buffer_cfg.log=False" \
  "$@"

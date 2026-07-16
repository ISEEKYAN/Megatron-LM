#!/usr/bin/env bash
# DeepSeek-V4 DAPO run -- SELF-CONTAINED (no delegation to any gsm8k/qwen runner).
# Directly invokes verl.trainer.main_ppo with the DS4 geometry + fp8 resync + DAPO
# recipe. Standard DAPO data: dapo-math-17k (train) + aime-2024 (val).
#
# DAPO recipe: clip-higher (asymmetric PPO clip) + dual-clip; no KL (reward or loss);
# token-mean loss aggregation; overlong reward shaping (soft length penalty).
# Resync: mlite exports a pre-quantized block-fp8 checkpoint (resync_format=block_fp8,
# expert_dtype=fp8) that vLLM loads directly via hf_overrides.
set -euo pipefail
[[ "${VERBOSE:-0}" == "1" ]] && set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"
DATASET_MODULE="${EXAMPLE_ROOT}/verl_mlite/dataset.py"

add_pythonpath() { if [[ -n "${1:-}" ]]; then export PYTHONPATH="${1}:${PYTHONPATH:-}"; fi; }
add_pythonpath "${EXAMPLE_ROOT}"; add_pythonpath "${LITE_ROOT}"; add_pythonpath "${REPO_ROOT}"
add_pythonpath "${VERL_ROOT:-}"; add_pythonpath "${MEGATRON_ROOT:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

: "${MODEL_PATH:?set MODEL_PATH to the official mixed DeepSeek V4 checkpoint}"
# DAPO standard data (dapo-math-17k train + aime-2024 val); override to relocate.
DAPO_DATA_DIR="${DAPO_DATA_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/verl_update_mcore/data}"
TRAIN_FILES="${TRAIN_FILES:-${DAPO_DATA_DIR}/dapo-math-17k.parquet}"
VAL_FILES="${VAL_FILES:-${DAPO_DATA_DIR}/aime-2024.parquet}"

INFER_BACKEND="${INFER_BACKEND:-vllm}"
NNODES="${NNODES:-1}"; NGPUS_PER_NODE="${NGPUS_PER_NODE:-${NPROC_PER_NODE:-8}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EXAMPLE_ROOT}/outputs/ds4_dapo}"
PROJECT_NAME="${PROJECT_NAME:-verl-mlite-ds4-dapo}"

# Geometry (PP4 x EP8 x CP4, rollout TP8).
ACTOR_TP="${ACTOR_TP:-1}"; ACTOR_PP="${ACTOR_PP:-4}"; ACTOR_CP="${ACTOR_CP:-4}"
ACTOR_EP="${ACTOR_EP:-8}"; ACTOR_ETP="${ACTOR_ETP:-1}"; ACTOR_VPP="${ACTOR_VPP:-null}"
DTYPE="${DTYPE:-bfloat16}"; MLITE_MODEL_NAME="${MLITE_MODEL_NAME:-deepseek_v4}"; MLITE_IMPL="${MLITE_IMPL:-lite}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fused}"; MLITE_OPTIMIZER_BACKEND="${MLITE_OPTIMIZER_BACKEND:-fsdp2}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-True}"; OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-True}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-False}"; OPTIMIZER_STATE_OFFLOAD_FRACTION="${OPTIMIZER_STATE_OFFLOAD_FRACTION:-1.0}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-True}"; DECOUPLED_WEIGHT_DECAY="${DECOUPLED_WEIGHT_DECAY:-True}"

# DAPO scale.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"; PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"; MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-10240}"
_MAX_SEQ_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-${_MAX_SEQ_LEN}}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
ROLLOUT_N="${ROLLOUT_N:-8}"; ROLLOUT_MODE="${ROLLOUT_MODE:-async}"; ROLLOUT_TP="${ROLLOUT_TP:-8}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${_MAX_SEQ_LEN}}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-${_MAX_SEQ_LEN}}"; ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${_MAX_SEQ_LEN}}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"; ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"; ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"; VAL_TOP_P="${VAL_TOP_P:-1.0}"; VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"; VAL_N="${VAL_N:-1}"

# Optim (DAPO: lr 1e-6, no KL).
ACTOR_LR="${ACTOR_LR:-1e-6}"; WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"; BETAS="${BETAS:-[0.9,0.95]}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"; LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"; LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"; POLICY_LOSS_MODE="${POLICY_LOSS_MODE:-vanilla}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"; USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
ROUTER_REPLAY_MODE="${ROUTER_REPLAY_MODE:-disabled}"
ENABLE_ROLLOUT_ROUTING_REPLAY="${ENABLE_ROLLOUT_ROUTING_REPLAY:-False}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"; TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-20}"; TEST_FREQ="${TEST_FREQ:-5}"; RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"; LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-10}"; LOGGER="${LOGGER:-[console,file]}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"; export VLLM_USE_V1
export VERL_VLLM_FP8_QUANT_ENABLED="${VERL_VLLM_FP8_QUANT_ENABLED:-1}"

# DAPO recipe knobs.
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"; CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"; CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"; OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((1024 * 4))}"; OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

MLITE_VPP_SIZE="${ACTOR_VPP}"; [[ "${MLITE_VPP_SIZE}" == "null" ]] && MLITE_VPP_SIZE=1
case "${MLITE_OPTIMIZER_BACKEND}" in dist_opt) MLITE_IMPL_OPTIMIZER="dist_opt";; fsdp2) MLITE_IMPL_OPTIMIZER="fsdp2";; *) echo "bad MLITE_OPTIMIZER_BACKEND"; exit 1;; esac

RUN_NAME="${RUN_NAME:-ds4_dapo_pp${ACTOR_PP}_ep${ACTOR_EP}_cp${ACTOR_CP}_rtp${ROLLOUT_TP}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"; JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"; CMD_FILE="${CMD_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.cmd.sh}"
mkdir -p "${OUTPUT_ROOT}" "${CKPT_DIR}" "$(dirname "${LOG_FILE}")"
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"

# Fail loud if rollout_tp does not evenly divide o_groups.
python3 - "${MODEL_PATH}/config.json" "${ROLLOUT_TP}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1])); og, tp = cfg.get("o_groups"), int(sys.argv[2])
assert isinstance(og, int) and og >= 1 and og % tp == 0, f"DS4 o_groups={og} must be divisible by rollout_tp={tp}"
print(f"DS4_ROLLOUT_TP_PREFLIGHT_PASSED o_groups={og} rollout_tp={tp}", flush=True)
PY

DEFAULT_CHAT_TEMPLATE='{% for message in messages %}{% if message["content"] is string %}{{ message["content"] }}{% else %}{% for content in message["content"] %}{% if content["type"] == "text" %}{{ content["text"] }}{% endif %}{% endfor %}{% endif %}{% if not loop.last %}{{ "\n\n" }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ "\n" }}{% endif %}'
DS4_CHAT_TEMPLATE="${DEEPSEEK_V4_FLASH_CHAT_TEMPLATE:-${DEFAULT_CHAT_TEMPLATE}}"

ALGORITHM=( "algorithm.adv_estimator=grpo" "algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}" "algorithm.kl_ctrl.kl_coef=${KL_COEF:-0.0}" "algorithm.rollout_correction.bypass_mode=${ROLLOUT_CORRECTION_BYPASS:-False}" "algorithm.norm_adv_by_std_in_grpo=False" )
DATA=( "data.train_files=${TRAIN_FILES}" "data.val_files=${VAL_FILES}" "data.train_batch_size=${TRAIN_BATCH_SIZE}" "data.prompt_key=prompt" "data.return_raw_chat=True" "data.max_prompt_length=${MAX_PROMPT_LENGTH}" "data.max_response_length=${MAX_RESPONSE_LENGTH}" "data.filter_overlong_prompts=True" "data.truncation=error" "data.custom_cls.path=${DATASET_MODULE}" "data.custom_cls.name=ChatTemplateRLHFDataset" "+data.chat_template='${DS4_CHAT_TEMPLATE}'" "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS:-8}" )
MODEL=( "actor_rollout_ref.model.path=${MODEL_PATH}" "actor_rollout_ref.model.trust_remote_code=True" "actor_rollout_ref.model.use_fused_kernels=False" "actor_rollout_ref.model.custom_chat_template='${DS4_CHAT_TEMPLATE}'" )
ACTOR=( "actor@actor_rollout_ref.actor=mlite_actor" "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}" "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}" "actor_rollout_ref.actor.optim.betas=${BETAS}" "actor_rollout_ref.actor.optim.clip_grad=${CLIP_GRAD}" "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}" "actor_rollout_ref.actor.optim.lr_warmup_init=0" "actor_rollout_ref.actor.optim.lr_decay_style=${LR_DECAY_STYLE}" "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}" "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}" "actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}" "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF:-0.0}" "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}" "actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE}" "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}" "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}" "actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}" "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}" "actor_rollout_ref.actor.engine.dtype=${DTYPE}" "actor_rollout_ref.actor.engine.model_name=${MLITE_MODEL_NAME}" "actor_rollout_ref.actor.engine.impl=${MLITE_IMPL}" "actor_rollout_ref.actor.engine.tp=${ACTOR_TP}" "actor_rollout_ref.actor.engine.pp=${ACTOR_PP}" "actor_rollout_ref.actor.engine.vpp=${MLITE_VPP_SIZE}" "actor_rollout_ref.actor.engine.cp=${ACTOR_CP}" "actor_rollout_ref.actor.engine.ep=${ACTOR_EP}" "actor_rollout_ref.actor.engine.etp=${ACTOR_ETP}" "actor_rollout_ref.actor.engine.param_offload=${PARAM_OFFLOAD}" "actor_rollout_ref.actor.engine.optimizer_offload=${OPTIMIZER_OFFLOAD}" "actor_rollout_ref.actor.engine.grad_offload=${GRAD_OFFLOAD}" "actor_rollout_ref.actor.engine.attention_backend_override=${ATTENTION_BACKEND}" "actor_rollout_ref.actor.engine.impl_cfg.use_thd=True" "+actor_rollout_ref.actor.engine.impl_cfg.optimizer=${MLITE_IMPL_OPTIMIZER}" "actor_rollout_ref.actor.engine.load_hf_weights=${ENGINE_LOAD_HF_WEIGHTS:-True}" "+actor_rollout_ref.actor.engine.cross_entropy_fusion=True" "actor_rollout_ref.actor.engine.resync_format=block_fp8" "+actor_rollout_ref.actor.engine.resync_config.expert_dtype=fp8" "+actor_rollout_ref.actor.engine.impl_cfg.recompute=full" "+actor_rollout_ref.actor.engine.impl_cfg.mtp_enable=True" "+actor_rollout_ref.actor.engine.impl_cfg.mtp_enable_train=True" )
ACTOR+=( "actor_rollout_ref.actor.engine.router_replay_mode=${ROUTER_REPLAY_MODE}" )
if [[ "${OPTIMIZER_OFFLOAD}" =~ ^(True|true|1)$ ]]; then ACTOR+=( "+actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=${OPTIMIZER_STATE_OFFLOAD_FRACTION}" "+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER}" "+actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=${DECOUPLED_WEIGHT_DECAY}" ); fi
if [[ "${USE_KL_LOSS}" =~ ^(True|true|1)$ || "${USE_KL_IN_REWARD}" =~ ^(True|true|1)$ ]]; then ACTOR+=("ref@actor_rollout_ref.ref=mlite_ref"); fi
ROLLOUT=( "actor_rollout_ref.rollout.name=${INFER_BACKEND}" "actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}" "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}" "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}" "actor_rollout_ref.rollout.n=${ROLLOUT_N}" "actor_rollout_ref.rollout.calculate_log_probs=True" "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}" "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}" "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}" "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}" "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}" "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}" "actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}" "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}" "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}" "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}" "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}" "actor_rollout_ref.rollout.free_cache_engine=True" "actor_rollout_ref.rollout.load_format=dummy" "+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True" "+actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls=verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension" "+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8" "+actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=flashinfer_cutlass" "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.expert_dtype=fp8" "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.activation_scheme=dynamic" "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.fmt=e4m3" "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.quant_method=fp8" "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.scale_fmt=float32" "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.quantization_config.weight_block_size=[128,128]" )
TRAINER=( "critic.enable=False" "trainer.balance_batch=True" "trainer.logger=${LOGGER}" "trainer.project_name=${PROJECT_NAME}" "trainer.experiment_name=${RUN_NAME}" "trainer.n_gpus_per_node=${NGPUS_PER_NODE}" "trainer.nnodes=${NNODES}" "trainer.save_freq=${SAVE_FREQ}" "trainer.test_freq=${TEST_FREQ}" "trainer.total_epochs=${TOTAL_EPOCHS}" "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" "trainer.resume_mode=${RESUME_MODE}" "trainer.resume_from_path=${RESUME_FROM_PATH}" "trainer.default_local_dir=${CKPT_DIR}" "trainer.val_before_train=False" "trainer.log_val_generations=${LOG_VAL_GENERATIONS}" )
REWARD=( "+reward.reward_kwargs.overlong_buffer_cfg.enable=True" "+reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}" "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}" "+reward.reward_kwargs.overlong_buffer_cfg.log=False" )
ROLLOUT+=( "actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY}" )

COMMAND=( python3 -m verl.trainer.main_ppo "hydra.searchpath=[pkg://verl_mlite.config]" "${ALGORITHM[@]}" "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" "${REWARD[@]}" "$@" )
printf '%q ' "${COMMAND[@]}" > "${CMD_FILE}"; printf '\n' >> "${CMD_FILE}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
echo "[ds4-dapo] self-contained; train=${TRAIN_FILES} val=${VAL_FILES} cmd=${CMD_FILE}"
set +e; "${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"; cmd_rc="${PIPESTATUS[0]}"; set -e
exit "${cmd_rc}"

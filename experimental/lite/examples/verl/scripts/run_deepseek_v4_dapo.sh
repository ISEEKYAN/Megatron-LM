#!/usr/bin/env bash
# DeepSeek-V4 DAPO run.
#
# Reuses the DS4 GRPO wrapper (run_deepseek_v4_gsm8k_grpo.sh) for the model
# geometry (PP/CP/EP), chat template, and the pre-quantized block-fp8 resync
# (mlite exports fp8; vLLM loads it directly), and layers the DAPO recipe on top:
#   - clip-higher (asymmetric PPO clip) + dual-clip,
#   - no KL (neither in the reward nor as a loss),
#   - token-mean loss aggregation,
#   - overlong reward shaping (soft length penalty).
# Knob defaults follow verl-project's recipe/dapo. Point MODEL_PATH /
# TRAIN_FILES / VAL_FILES at the DeepSeek-V4 checkpoint and a dapo-math-style
# parquet; every value below is overridable from the environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
GRPO_WRAPPER="${SCRIPT_DIR}/run_deepseek_v4_gsm8k_grpo.sh"

# DAPO long-generation defaults.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"

# DAPO drops KL entirely.
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"

export PROJECT_NAME="${PROJECT_NAME:-verl-mlite-ds4-dapo}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/ds4_dapo}"
export RUN_NAME="${RUN_NAME:-ds4_dapo_pp${ACTOR_PP:-4}_ep${ACTOR_EP:-8}_cp${ACTOR_CP:-4}}"

# DAPO recipe knobs (verl-project recipe/dapo defaults).
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((1024 * 4))}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

exec bash "${GRPO_WRAPPER}" \
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}" \
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}" \
  "actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}" \
  "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}" \
  "+reward.reward_kwargs.overlong_buffer_cfg.enable=True" \
  "+reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}" \
  "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}" \
  "+reward.reward_kwargs.overlong_buffer_cfg.log=False" \
  "$@"

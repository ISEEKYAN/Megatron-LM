#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Reproduce the four-arm MXFP4 QAT comparison with a public Qwen3-MoE model.
#
#   baseline: BF16 training and BF16 rollout.
#   qat_off:  MXFP4 rollout, but no training-side fake quantization.
#   qat_on:   MXFP4 rollout plus training-side MXFP4 fake quantization.
#   r3:       qat_on plus router replay.
#
# qat_off and qat_on intentionally have identical rollout settings. Their only
# QAT difference is actor_rollout_ref.actor.engine.impl_cfg.qat.enabled, so the
# comparison isolates whether training is aware of rollout quantization.
#
# In the measured four-arm run, the final rollout_probs_diff_mean values were
# 0.00598 (baseline), 0.0267 (qat_off), and 0.0166 (qat_on). Thus qat_on reduced
# the qat_off gap by about 38%, while remaining about 2.8x the BF16 baseline.
#
# The MXFP4 rollout is performed by vLLM compressed-tensors with
# verl.utils.qat.vllm_patch. engine.qat.enable remains false: the separate verl
# export_qat_weights path is not used by these arms.

set -euo pipefail

# This is the complete MXFP4 QAT delta for any verl launcher: copy this block
# into an existing GRPO/DAPO script; every other setting is ordinary training
# configuration. qat_off and qat_on must have byte-identical rollout settings;
# their sole QAT difference is impl_cfg.qat.enabled.
qat_overrides_for_mode() {
    local mode="$1"
    QAT_OVERRIDES=()
    case "$mode" in
        baseline)
            # BF16 training + BF16 rollout: the unquantized reference.
            QAT_OVERRIDES+=(
                "actor_rollout_ref.actor.engine.impl_cfg.qat.enabled=false"
                "actor_rollout_ref.actor.engine.impl_cfg.qat.format=mxfp4"
                "actor_rollout_ref.actor.engine.impl_cfg.router_replay.enabled=false"
                "actor_rollout_ref.rollout.enable_rollout_routing_replay=false"
            )
            ;;
        qat_off)
            # MXFP4 rollout without fake quant: measures training/rollout mismatch.
            QAT_OVERRIDES+=(
                "actor_rollout_ref.actor.engine.impl_cfg.qat.enabled=false"
                "actor_rollout_ref.actor.engine.impl_cfg.qat.format=mxfp4"
                "actor_rollout_ref.rollout.quantization=mxfp4"
                "actor_rollout_ref.actor.engine.impl_cfg.router_replay.enabled=false"
                "actor_rollout_ref.rollout.enable_rollout_routing_replay=false"
            )
            ;;
        qat_on)
            # MXFP4 rollout plus training-side MXFP4 fake quantization.
            QAT_OVERRIDES+=(
                "actor_rollout_ref.actor.engine.impl_cfg.qat.enabled=true"
                "actor_rollout_ref.actor.engine.impl_cfg.qat.format=mxfp4"
                "actor_rollout_ref.rollout.quantization=mxfp4"
                "actor_rollout_ref.actor.engine.impl_cfg.router_replay.enabled=false"
                "actor_rollout_ref.rollout.enable_rollout_routing_replay=false"
            )
            ;;
        r3)
            # qat_on plus router replay.
            QAT_OVERRIDES+=(
                "actor_rollout_ref.actor.engine.impl_cfg.qat.enabled=true"
                "actor_rollout_ref.actor.engine.impl_cfg.qat.format=mxfp4"
                "actor_rollout_ref.rollout.quantization=mxfp4"
                "actor_rollout_ref.actor.engine.impl_cfg.router_replay.enabled=true"
                "actor_rollout_ref.rollout.enable_rollout_routing_replay=true"
            )
            ;;
    esac
}

MODE="${MODE:-qat_on}"
if [[ "${1:-}" == --mode=* ]]; then
    MODE="${1#--mode=}"
    shift
elif [[ "${1:-}" == "--mode" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "--mode requires one of: baseline, qat_off, qat_on, r3" >&2
        exit 2
    fi
    MODE="$2"
    shift 2
fi

case "$MODE" in
    baseline | qat_off | qat_on | r3) ;;
    *)
        echo "Unknown mode '$MODE'; expected baseline, qat_off, qat_on, or r3" >&2
        exit 2
        ;;
esac
qat_overrides_for_mode "$MODE"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="${BASE_LAUNCHER:-${SCRIPT_DIR}/run_qwen3moe_gsm8k_grpo.sh}"

# User-provided inputs. The defaults name publicly available artifacts; local
# paths can be supplied when the runtime cannot download them directly.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B}"
TRAIN_FILES="${TRAIN_FILES:-}"
VAL_FILES="${VAL_FILES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/qwen3moe-mxfp4-qat}"
if [[ -z "$TRAIN_FILES" || -z "$VAL_FILES" ]]; then
    echo "Set TRAIN_FILES and VAL_FILES to verl-compatible parquet files." >&2
    echo "Public sources: BytedTsinghua-SIA/DAPO-Math-17k and AIME 2024." >&2
    exit 2
fi

# Resource and recipe parameters are intentionally overridable.
NNODES="${NNODES:-4}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-8}"
ACTOR_TP="${ACTOR_TP:-2}"
ACTOR_EP="${ACTOR_EP:-8}"
ACTOR_CP="${ACTOR_CP:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
N_RESPONSES="${N_RESPONSES:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-30}"

COMMON_OVERRIDES=(
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=2048"
    "data.max_response_length=14336"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.optim.lr=1e-5"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384"
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1"
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}"
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}"
    "actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}"
    "actor_rollout_ref.actor.engine.impl_cfg.recompute=full"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.n=${N_RESPONSES}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
    "actor_rollout_ref.rollout.max_model_len=16384"
    "algorithm.adv_estimator=grpo"
    "algorithm.rollout_correction.bypass_mode=false"
    "actor_rollout_ref.actor.clip_ratio_low=0.2"
    "actor_rollout_ref.actor.clip_ratio_high=0.28"
    "actor_rollout_ref.actor.loss_agg_mode=token-mean"
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.test_freq=5"
    "trainer.save_freq=5"
    "trainer.project_name=mlite-qat"
    "trainer.experiment_name=qwen3moe-mxfp4-${MODE}"
    "trainer.default_local_dir=${OUTPUT_ROOT}/${MODE}"
)

COMMAND=(bash "$BASE_LAUNCHER" "${COMMON_OVERRIDES[@]}" "${QAT_OVERRIDES[@]}" "$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    exec "${COMMAND[@]}"
fi

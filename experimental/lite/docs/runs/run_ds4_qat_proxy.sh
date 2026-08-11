#!/usr/bin/env bash
# Submit a one-node, eight-GPU DS4 QAT proxy run.  Supply local paths; this
# example intentionally contains no cluster-, user-, or checkpoint-specific data.
set -euo pipefail

: "${BASE_IMAGE:?set BASE_IMAGE to the qualified container image}"
: "${MLITE_SRC:?set MLITE_SRC to this checkout}"
: "${MEGATRON_ROOT:?set MEGATRON_ROOT to the compatible MCore checkout}"
: "${VERL_ROOT:?set VERL_ROOT to VERL at b9c513c4}"
: "${DS4_VLLM_SITE:?set DS4_VLLM_SITE to vLLM 0.25.1 site-packages}"
: "${MODELOPT_SITE:?set MODELOPT_SITE to ModelOpt plus cudnn-frontend >=1.27}"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR to a DS4 proxy checkpoint}"
: "${RUN_ROOT:?set RUN_ROOT to a fresh writable directory}"

sbatch --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:8 \
  --cpus-per-task=64 --mem=0 --time=01:00:00 \
  --export="ALL,BASE_IMAGE,MLITE_SRC,MEGATRON_ROOT,VERL_ROOT,DS4_VLLM_SITE,MODELOPT_SITE,CHECKPOINT_DIR,RUN_ROOT,DS4_SHARED_DATA=${DS4_SHARED_DATA:-},ENABLE_QAT=True,ROLLOUT_WEIGHT_BITS=4,ENABLE_R3=True,VERL_COMMIT=b9c513c4,RAY_OVERRIDE_JOB_RUNTIME_ENV=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,NNODES=1,NGPUS_PER_NODE=8,ACTOR_PP=2,ACTOR_CP=2,ACTOR_EP=2,ROLLOUT_TP=8,TOTAL_TRAINING_STEPS=2,MAX_PROMPT_LENGTH=1024,MAX_RESPONSE_LENGTH=192" \
  "${MLITE_SRC}/experimental/lite/examples/verl/slurm/run_ds4_gsm8k_grpo.sbatch"

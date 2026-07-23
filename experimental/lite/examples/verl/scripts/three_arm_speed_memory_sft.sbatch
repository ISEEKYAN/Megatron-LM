#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -J q3moe_speed_mem
set -euo pipefail

# AC#4: one SBATCH invocation = one arm (adamw / muon dist_opt / muon fsdp2).
ARM="${ARM:?set ARM=adamw|muon|muon_fsdp2}"
BASE="${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}"
BASE_IMAGE="${BASE_IMAGE:-$BASE/env/pytorch_26.04-py3.sqsh}"
VERL_DSA_SITE="${VERL_DSA_SITE:-$BASE/mlite-2604-verl-dsa-sm90-overlay/lib/python3.12/site-packages}"
MLITE_REPO="${MLITE_REPO:?set MLITE_REPO}"
MUON_SNAPSHOT="${MUON_SNAPSHOT:-$BASE/runtime/muon-mbridge-386bf7af6-r2}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$MUON_SNAPSHOT/megatron-d64}"
EMERGING_OPT_ROOT="${EMERGING_OPT_ROOT:-$MUON_SNAPSHOT/emerging-optimizers}"
VERL="${VERL:-$BASE/verl-main-latest}"
MODEL_PATH="${MODEL_PATH:-$BASE/models/Qwen3-30B-A3B}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"

SEED="${SEED:-1234}"
TOTAL_STEPS="${TOTAL_STEPS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
ETP_SIZE="${ETP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"

case "$ARM" in
  adamw) OPT_ALGO=adamw ; OPT_BACKEND=dist_opt ; MUON_TP_MODE=distributed ;;
  muon) OPT_ALGO=muon ; OPT_BACKEND=dist_opt ; MUON_TP_MODE=distributed ;;
  muon_fsdp2) OPT_ALGO=muon ; OPT_BACKEND=fsdp2 ; MUON_TP_MODE=distributed ;;
  *) echo "Unknown ARM=$ARM" >&2 ; exit 2 ;;
esac

TRAIN_FILES="${TRAIN_FILES:?set TRAIN_FILES}"
OUTPUT_ROOT="$RUN_ROOT/$ARM"
mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT/logs"

echo "Q3MOE_SPEED_MEM_ARM job=${SLURM_JOB_ID:-none} arm=$ARM opt=$OPT_ALGO backend=$OPT_BACKEND seed=$SEED steps=$TOTAL_STEPS tp=$TP_SIZE ep=$EP_SIZE"

NODE_SCRIPT="$MLITE_REPO/experimental/lite/examples/verl/scripts/three_arm_speed_memory_node.sh"
[[ -f "$NODE_SCRIPT" ]] || { echo "missing node script: $NODE_SCRIPT" >&2; exit 3; }

SA=(--container-image="$BASE_IMAGE" --container-mounts=/lustre:/lustre --export=ALL)
export ARM OPT_ALGO OPT_BACKEND MUON_TP_MODE
export BASE BASE_IMAGE VERL_DSA_SITE MLITE_REPO MEGATRON_ROOT EMERGING_OPT_ROOT VERL MODEL_PATH RUN_ROOT OUTPUT_ROOT TRAIN_FILES MUON_SNAPSHOT
export SEED TOTAL_STEPS TRAIN_BATCH_SIZE MICRO_BATCH_SIZE MAX_TOKENS_PER_GPU MAX_LENGTH
export LR MIN_LR LR_WARMUP_STEPS LR_DECAY_STYLE WEIGHT_DECAY CLIP_GRAD
export TP_SIZE PP_SIZE EP_SIZE ETP_SIZE CP_SIZE

srun --nodes=1 --ntasks=1 "${SA[@]}" bash "$NODE_SCRIPT"
rc=$?
echo "Q3MOE_SPEED_MEM_DONE job=${SLURM_JOB_ID:-none} arm=$ARM rc=$rc"
exit "$rc"

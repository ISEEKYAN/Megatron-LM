#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -J q35_muon_precision
# NOTE: --output is supplied by the submit wrapper so each arm lands in its own file.
set -euo pipefail

# =============================================================================
# Three-arm precision harness for Muon on a REAL workload (AC#3).
#
# One SBATCH invocation runs exactly one ARM of a fixed same-contract Qwen3.5-35B
# SFT run so that loss trajectories / weights can be compared across arms:
#
#   ARM=adamw   -> optimizer_algorithm=adamw, MLite dist_opt backend  (A/B baseline)
#   ARM=muon    -> optimizer_algorithm=muon,  muon_tp_mode=distributed (DistOpt Muon
#                  doing the *true* cross-TP Newton-Schulz, TP>1; this is the primary
#                  DistOpt Muon arm bayan asked to verify on a real workload)
#   ARM=muon_local -> optimizer_algorithm=muon, muon_tp_mode=blockwise (the per-shard
#                  local NS approximation; blockwise is aliased to `duplicated` inside
#                  megatron.core.optimizer.emerging_optimizers, i.e. NOT distributed).
#                  Kept as a contrast arm so the distributed path is provably different.
#
# NOTE on the optimizer implementation: MLite's dist_opt Muon forwards directly to
# NVIDIA Megatron-Core `get_megatron_muon_optimizer` (megatron/core/optimizer/muon.py
# -> emerging_optimizers.py).  So "MLite DistOpt Muon" IS Megatron's TensorParallel
# Muon by construction; MEGATRON_ROOT must therefore point at an NVIDIA Megatron-Core
# checkout that ships muon (the ISEEKYAN fork does not).
#
# Everything else (model, data order, init, seed, tokens, LR schedule, warmup,
# parallel geometry) is held identical by pinning it here.  This is a *training*
# harness: the precision verdict comes from the emitted per-step JSONL loss, not
# from any config assertion.
#
# Env recipe reused verbatim from the verified qwen3.5 mFSDP DAPO node script
# (qwen35_dapo_mfsdp_62295f9b3/run_dapo_h100_node_d62c5aa46.sh) and
# dead_ends/mlite-env-setup.md.  Container verl.vllm023.sqsh is self-contained
# (/vllm + verl_mlite deps); we only prepend the task's mlite checkout + verl src.
# =============================================================================

ARM="${ARM:?set ARM=adamw|muon|muon_mcore}"
BASE="${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}"
IMG="${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}"
# Task mlite checkout (this branch @ 7b3dc95e4, with the VERL Muon optimizer
# contract wiring) staged on lustre by the submit wrapper.
MLITE_REPO="${MLITE_REPO:?set MLITE_REPO to the staged mlite checkout}"
# NVIDIA Megatron-Core with megatron/core/optimizer/muon.py (dist_opt Muon impl).
MEGATRON_ROOT="${MEGATRON_ROOT:-$BASE/runtime/ds4-csacp-parity-eaa5b486d/mcore}"
VERL="${VERL:-$BASE/verl-main-latest}"
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT for outputs}"

# ---- fixed same-contract knobs (identical across the three arms) -------------
SEED="${SEED:-1234}"
TOTAL_STEPS="${TOTAL_STEPS:-40}"
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

MODE_LABEL="$ARM"
case "$ARM" in
  adamw)
    OPT_ALGO=adamw ; OPT_BACKEND=dist_opt ; MUON_TP_MODE=distributed ;;
  muon)
    OPT_ALGO=muon  ; OPT_BACKEND=dist_opt ; MUON_TP_MODE=distributed ;;
  muon_local)
    OPT_ALGO=muon  ; OPT_BACKEND=dist_opt ; MUON_TP_MODE=blockwise ;;
  *) echo "Unknown ARM=$ARM (want adamw|muon|muon_local)" >&2 ; exit 2 ;;
esac

TRAIN_FILES="${TRAIN_FILES:?set TRAIN_FILES to a verl messages-format parquet}"
OUTPUT_ROOT="$RUN_ROOT/$MODE_LABEL"
mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT/logs"

echo "Q35_PRECISION_ARM job=${SLURM_JOB_ID:-none} arm=$ARM opt_algo=$OPT_ALGO backend=$OPT_BACKEND muon_tp_mode=$MUON_TP_MODE model=$MODEL_PATH steps=$TOTAL_STEPS seed=$SEED tp=$TP_SIZE pp=$PP_SIZE ep=$EP_SIZE"

NODE_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/three_arm_precision_node.sh"

SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
export ARM OPT_ALGO OPT_BACKEND MUON_TP_MODE
export BASE IMG MLITE_REPO MEGATRON_ROOT VERL MODEL_PATH RUN_ROOT OUTPUT_ROOT TRAIN_FILES
export SEED TOTAL_STEPS TRAIN_BATCH_SIZE MICRO_BATCH_SIZE MAX_TOKENS_PER_GPU MAX_LENGTH
export LR MIN_LR LR_WARMUP_STEPS LR_DECAY_STYLE WEIGHT_DECAY CLIP_GRAD
export TP_SIZE PP_SIZE EP_SIZE ETP_SIZE CP_SIZE

srun --nodes=1 --ntasks=1 "${SA[@]}" bash "$NODE_SCRIPT"
rc=$?
echo "Q35_PRECISION_ARM_DONE job=${SLURM_JOB_ID:-none} arm=$ARM rc=$rc"
exit "$rc"

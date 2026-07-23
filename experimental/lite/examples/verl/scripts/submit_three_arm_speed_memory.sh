#!/usr/bin/env bash
# Orchestrator for AC#4 three-arm speed+memory experiment on CW.
#
#   GATE_ONLY=1 bash submit_three_arm_speed_memory.sh
#   ARMS="adamw muon muon_fsdp2" bash submit_three_arm_speed_memory.sh
set -euo pipefail

BASE="${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}"
RUN_ROOT="${RUN_ROOT:-$BASE/runtime/task-1-13-5-5-4-speed-memory}"
MLITE_REPO="${MLITE_REPO:-$BASE/megatron_lite/Megatron-LM}"
MUON_SNAPSHOT="${MUON_SNAPSHOT:-$BASE/runtime/muon-mbridge-386bf7af6-r2}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$MUON_SNAPSHOT/megatron-d64}"
EMERGING_OPT_ROOT="${EMERGING_OPT_ROOT:-$MUON_SNAPSHOT/emerging-optimizers}"
BASE_IMAGE="${BASE_IMAGE:-$BASE/env/pytorch_26.04-py3.sqsh}"
VERL_DSA_SITE="${VERL_DSA_SITE:-$BASE/mlite-2604-verl-dsa-sm90-overlay/lib/python3.12/site-packages}"
VERL="${VERL:-$BASE/verl-main-latest}"
MODEL_PATH="${MODEL_PATH:-$BASE/models/Qwen3-30B-A3B}"
DATA_DIR="${DATA_DIR:-$BASE/runtime/task-1-13-5-5-3-muon-precision/data/gsm8k_sft}"
TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/train.parquet}"
ARMS="${ARMS:-adamw muon muon_fsdp2}"
SBATCH="${SBATCH:-three_arm_speed_memory_sft.sbatch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
mkdir -p "$RUN_ROOT/logs"

echo "[preflight] mlite=$(git -C "$MLITE_REPO" rev-parse --short HEAD 2>/dev/null || echo MISSING)"
echo "[preflight] mcore=$(git -C "$MEGATRON_ROOT" rev-parse --short HEAD 2>/dev/null || echo MISSING)"
echo "[preflight] emerging=$(git -C "$EMERGING_OPT_ROOT" rev-parse --short HEAD 2>/dev/null || echo MISSING)"
test -f "$MUON_SNAPSHOT/READY"
test -f "$TRAIN_FILES"

echo "===================== CONFIG DRY-RUN GATE ====================="
for arm in adamw muon muon_fsdp2; do
  case "$arm" in
    adamw) oa=adamw; be=dist_opt ;;
    muon) oa=muon; be=dist_opt ;;
    muon_fsdp2) oa=muon; be=fsdp2 ;;
  esac
  echo "----- arm=$arm optimizer=$oa backend=$be -----"
  OUTPUT_ROOT="$(mktemp -d)" MODEL_PATH="$MODEL_PATH" TRAIN_FILES="$TRAIN_FILES" \
    OPTIMIZER_ALGORITHM="$oa" MUON_TP_MODE=distributed MLITE_OPTIMIZER_BACKEND="$be" \
    DRY_RUN=1 bash "$SCRIPT_DIR/run_qwen3moe_sft.sh" \
    | tr ' ' '\n' | grep -E 'optim\.optimizer=|muon_tp_mode=|impl_cfg\.optimizer=' || true
done

if [[ "${GATE_ONLY:-0}" == "1" ]]; then
  echo "[gate-only] stop before GPU."
  exit 0
fi

for arm in $ARMS; do
  out="$RUN_ROOT/logs/%x_%j_${arm}.out"
  jid=$(sbatch --parsable --output="$out" \
    --export=ALL,ARM="$arm",BASE="$BASE",RUN_ROOT="$RUN_ROOT",MLITE_REPO="$MLITE_REPO",\
MUON_SNAPSHOT="$MUON_SNAPSHOT",MEGATRON_ROOT="$MEGATRON_ROOT",EMERGING_OPT_ROOT="$EMERGING_OPT_ROOT",\
BASE_IMAGE="$BASE_IMAGE",VERL_DSA_SITE="$VERL_DSA_SITE",VERL="$VERL",MODEL_PATH="$MODEL_PATH",TRAIN_FILES="$TRAIN_FILES" \
    "$SCRIPT_DIR/$SBATCH")
  echo "SUBMITTED arm=$arm job=$jid log=$out"
done

#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -J q35_fla_thd_smoke
# --output supplied by the submit wrapper.
set -euo pipefail

# One node, one arm (adamw/dist_opt).  Proves Qwen3.5-35B GatedDeltaNet +
# use_thd=True forward produces a loss once the FLA overlay (CP_SITE) is wired.
BASE="${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}"
IMG="${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}"
CP_SITE="${CP_SITE:-$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site}"
MLITE_REPO="${MLITE_REPO:?set MLITE_REPO to the staged mlite checkout}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$BASE/runtime/ds4-csacp-parity-eaa5b486d/mcore}"
VERL="${VERL:-$BASE/verl-main-latest}"
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/users/bayan/code/models/Qwen3.5-35B-A3B}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT for outputs}"
TRAIN_FILES="${TRAIN_FILES:?set TRAIN_FILES to a verl messages-format parquet}"

OUTPUT_ROOT="$RUN_ROOT/fla_thd_smoke"
mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT/logs"

NODE_SCRIPT="$MLITE_REPO/experimental/lite/docs/runs/claude/task-1.13.5.5.5/fla_thd_smoke_node.sh"
if [[ ! -f "$NODE_SCRIPT" ]]; then
  # Fall back to the copy shipped alongside this sbatch (staged separately).
  NODE_SCRIPT="${NODE_SCRIPT_OVERRIDE:?node script missing under staged mlite; set NODE_SCRIPT_OVERRIDE}"
fi

echo "FLA_THD_SMOKE job=${SLURM_JOB_ID:-none} img=$IMG cp_site=$CP_SITE model=$MODEL_PATH"

export BASE IMG CP_SITE MLITE_REPO MEGATRON_ROOT VERL MODEL_PATH RUN_ROOT OUTPUT_ROOT TRAIN_FILES

srun --nodes=1 --ntasks=1 \
  --container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL \
  bash "$NODE_SCRIPT"
rc=$?
echo "FLA_THD_SMOKE_DONE job=${SLURM_JOB_ID:-none} rc=$rc"
exit "$rc"

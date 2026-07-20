#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -J glm5_dsa_cp
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/llmrl_runs/%x_%j.out
set -euo pipefail

( sleep "${WATCHDOG_SECS:-1500}"; echo "GLM5_DSA_CP_WATCHDOG job=$SLURM_JOB_ID"; scancel "$SLURM_JOB_ID" ) &
WD_PID=$!
BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
REPO_ROOT=${REPO_ROOT:?submit with REPO_ROOT pointing at the committed checkout}
IMG=${IMG:-$BASE/env/pytorch_26.04-py3.sqsh}
DSA_SITE=${DSA_SITE:-$BASE/mlite-2604-dsa-overlay/lib/python3.12/site-packages}
CORE_TREE=${CORE_TREE:-$BASE/megatron_lite/Megatron-LM}
WORK_DIR=$REPO_ROOT/experimental/lite/docs/runs/glm5_dsa_cp_native
OUTPUT_DIR=$BASE/llmrl_runs/glm5_dsa_cp/output
mkdir -p "$BASE/llmrl_runs/glm5_dsa_cp/logs" "$OUTPUT_DIR"
export REPO_ROOT IMG DSA_SITE CORE_TREE WORK_DIR OUTPUT_DIR

running_ts=$(date -Is)
echo "GLM5_DSA_CP_JOB job=$SLURM_JOB_ID submit=$running_ts running=$running_ts first_diag=$running_ts repo=$REPO_ROOT commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
set +e
srun --nodes=1 --ntasks=1 --gpus-per-node=2 \
  --container-image="$IMG" \
  --container-mounts=/lustre:/lustre,/home/scratch.bayan_gpu:/home/scratch.bayan_gpu \
  --container-workdir="$REPO_ROOT" \
  --export=ALL,REPO_ROOT,DSA_SITE,CORE_TREE,WORK_DIR,OUTPUT_DIR \
  bash "$WORK_DIR/run.sh"
rc=$?
set -e
kill "$WD_PID" 2>/dev/null || true
echo "GLM5_DSA_CP_JOB_DONE job=$SLURM_JOB_ID rc=$rc"
exit "$rc"

#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH -J pp2_arch_rerun
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/llmrl_runs/task-1.2.5/logs/%x_%j.out
set -euo pipefail

( sleep "${WATCHDOG_SECS:-2400}"; echo "PP2_ARCHAEOLOGY_WATCHDOG job=$SLURM_JOB_ID"; scancel "$SLURM_JOB_ID" ) &
WD_PID=$!

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
REPO_ROOT=${REPO_ROOT:-$BASE/megatron_lite/Megatron-LM}
# STAGING is optional: when empty the run script uses the harness/bench that
# already live in REPO_ROOT (correct once REPO_ROOT is checked out to the branch
# under test) and skips the staging copy-over.
STAGING=${STAGING:-}
IMG=${IMG:-$BASE/env/pytorch_26.04-py3.sqsh}
DSA_SITE=${DSA_SITE:-$BASE/mlite-2604-dsa-overlay/lib/python3.12/site-packages}
WORK_DIR=${WORK_DIR:-${STAGING:-$REPO_ROOT}/experimental/lite/docs/runs/cursor/task-1.2.5}
OUTPUT_DIR=$BASE/llmrl_runs/task-1.2.5/output
mkdir -p $BASE/llmrl_runs/task-1.2.5/logs "$OUTPUT_DIR"

# srun --export=ALL,VAR only forwards vars present in this shell's *environment*;
# locally-assigned vars must be exported or they arrive empty in the container
# (job 13946170 died at 44s on "DSA_SITE required" for exactly this reason).
export REPO_ROOT STAGING IMG DSA_SITE WORK_DIR OUTPUT_DIR

submit_ts=$(date -Is)
running_ts=$(date -Is)
echo "PP2_ARCHAEOLOGY_JOB job=$SLURM_JOB_ID submit=$submit_ts running=$running_ts first_diag=$running_ts repo=$REPO_ROOT"

set +e
srun \
  --nodes=1 --ntasks=1 --gpus-per-node=8 \
  --container-image="$IMG" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO_ROOT" \
  --export=ALL,REPO_ROOT,STAGING,DSA_SITE,WORK_DIR,OUTPUT_DIR,WORLD_SIZE=8 \
  bash "$WORK_DIR/run_pp2_archaeology_rerun.sh"
rc=$?
set -e

kill "$WD_PID" 2>/dev/null || true
echo "PP2_ARCHAEOLOGY_JOB_DONE job=$SLURM_JOB_ID rc=$rc"
exit "$rc"

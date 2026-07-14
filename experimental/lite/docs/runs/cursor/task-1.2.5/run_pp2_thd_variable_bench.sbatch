#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -J pp2_thd_var_bench
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/llmrl_runs/task-1.2.5/logs/%x_%j.out
set -euo pipefail

( sleep "${WATCHDOG_SECS:-1500}"; echo "PP2_THD_VAR_BENCH_WATCHDOG job=$SLURM_JOB_ID"; scancel "$SLURM_JOB_ID" ) &
WD_PID=$!

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
REPO_ROOT=${REPO_ROOT:-/home/scratch.bayan_gpu/code/llmrl/vicky-llmrl/.vicky/worktrees/TASK-1.2.5}
IMG=${IMG:-$BASE/env/pytorch_26.04-py3.sqsh}
DSA_SITE=${DSA_SITE:-$BASE/mlite-2604-dsa-overlay/lib/python3.12/site-packages}
WORK_DIR=$REPO_ROOT/experimental/lite/docs/runs/cursor/task-1.2.5
OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/llmrl_runs/task-1.2.5/output
mkdir -p "$(dirname "$SLURM_OUTPUT")" "$OUTPUT_DIR"

submit_ts=$(date -Is)
echo "PP2_THD_VAR_BENCH_JOB job=$SLURM_JOB_ID submit=$submit_ts repo=$REPO_ROOT image=$IMG"

set +e
srun \
  --nodes=1 --ntasks=1 --gpus-per-node=8 \
  --container-image="$IMG" \
  --container-mounts=/lustre:/lustre,/home/scratch.bayan_gpu:/home/scratch.bayan_gpu \
  --container-workdir="$REPO_ROOT" \
  --export=ALL,REPO_ROOT,DSA_SITE,WORK_DIR,OUTPUT_DIR,WORLD_SIZE=8 \
  bash "$WORK_DIR/run_pp2_thd_variable_bench.sh"
rc=$?
set -e

kill "$WD_PID" 2>/dev/null || true
echo "PP2_THD_VAR_BENCH_JOB_DONE job=$SLURM_JOB_ID rc=$rc"
exit "$rc"

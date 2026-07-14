#!/usr/bin/env bash
# Submit PP2+THD variable-length bench (core boundary test).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/home/scratch.bayan_gpu/code/llmrl/vicky-llmrl/.vicky/worktrees/TASK-1.2.5}
LUSTRE_LOG_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/llmrl_runs/task-1.2.5/logs
if [[ -d /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan ]]; then
  mkdir -p "$LUSTRE_LOG_ROOT"
  LOG_DIR="$LUSTRE_LOG_ROOT"
else
  LOG_DIR="$REPO_ROOT/experimental/lite/docs/runs/cursor/task-1.2.5/logs"
  mkdir -p "$LOG_DIR"
fi

job=$(sbatch --output="$LOG_DIR/%x_%j.out" --export=ALL,REPO_ROOT="$REPO_ROOT" "$SCRIPT_DIR/run_pp2_thd_variable_bench.sbatch" | awk '{print $NF}')
echo "PP2_THD_VAR_BENCH_SUBMITTED job=$job repo=$REPO_ROOT log_dir=$LOG_DIR"

#!/usr/bin/env bash
# Submit PP2+THD variable-length bench (core boundary test).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-/home/scratch.bayan_gpu/code/llmrl/vicky-llmrl/.vicky/worktrees/TASK-1.2.5}
mkdir -p /lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/llmrl_runs/task-1.2.5/logs

job=$(sbatch --export=ALL,REPO_ROOT="$REPO_ROOT" "$SCRIPT_DIR/run_pp2_thd_variable_bench.sbatch" | awk '{print $NF}')
echo "PP2_THD_VAR_BENCH_SUBMITTED job=$job repo=$REPO_ROOT"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
REPO_ROOT=${REPO_ROOT:-$BASE/megatron_lite/Megatron-LM}
STAGING=${STAGING:-$BASE/llmrl_runs/task-1.2.5/staging}
LOG_ROOT=$BASE/llmrl_runs/task-1.2.5/logs
mkdir -p "$LOG_ROOT"

job=$(sbatch --export=ALL,REPO_ROOT="$REPO_ROOT",STAGING="$STAGING" "$SCRIPT_DIR/run_pp2_archaeology_rerun.sbatch" | awk '{print $NF}')
echo "PP2_ARCHAEOLOGY_SUBMITTED job=$job repo=$REPO_ROOT staging=$STAGING"

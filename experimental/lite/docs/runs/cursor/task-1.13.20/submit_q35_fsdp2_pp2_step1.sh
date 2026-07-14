#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}
RUN_DIR=${RUN_DIR:-$BASE/qwen35_dapo_mfsdp_62295f9b3}
SCRIPT_DIR=${SCRIPT_DIR:-$RUN_DIR/task-1.13.20}
cd "$RUN_DIR"
job=$(sbatch --export=ALL,BACKEND=mlite,MLITE_OPTIMIZER=fsdp2 "$SCRIPT_DIR/run_q35_fsdp2_pp2_step1.sbatch" | awk '{print $NF}')
echo "Q35_PP2_SUBMITTED job=$job script=$SCRIPT_DIR"

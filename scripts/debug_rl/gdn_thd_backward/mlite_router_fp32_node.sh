#!/usr/bin/env bash
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
SOURCE=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/mlite_node_replay_base.sh
TARGET_WT=$BASE/qwen35_dapo_g2/wt-gdn-router-fp32

# Preserve the proven cache coordinates and recipe; change only the code worktree
# and output directory for the FP32-router production candidate.
sed \
  -e "s|WT=\$BASE/qwen35_dapo_g2/wt-runtime-fix|WT=$TARGET_WT|" \
  -e 's/out_mlite_replay/out_mlite_router_fp32/g' \
  "$SOURCE" | /bin/bash -s -- "$@"

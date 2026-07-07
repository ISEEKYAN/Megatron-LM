#!/usr/bin/env bash
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
SOURCE=$BASE/qwen35_dapo_g2/mlite_node_g2.sh
TARGET_WT=$BASE/qwen35_dapo_g2/wt-gdn-router-fp32
export TOTAL_TRAINING_STEPS=5 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False

# Keep the proven round-4 recipe and alter only the candidate worktree, bounded
# step count, file-only logger, and isolated output/run names.
sed \
  -e "s|WT=\$BASE/megatron_lite/wt-qwen35-dapo|WT=$TARGET_WT|" \
  -e "s|export LOGGER='\[console,file,wandb\]' PROJECT_NAME=qwen35-dapo-ab RUN_NAME=g2_mlite_10step|export LOGGER='[console,file]' PROJECT_NAME=qwen35-dapo-ab RUN_NAME=g2_mlite_router_fp32_5step|" \
  -e 's|out_mlite|out_mlite_router_fp32_5step|g' \
  "$SOURCE" | /bin/bash -s -- "$@"

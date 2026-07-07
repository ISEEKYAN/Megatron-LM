#!/usr/bin/env bash
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
SOURCE=$BASE/qwen35_dapo_g2/mlite_node_g2.sh
TARGET_WT=$BASE/qwen35_dapo_g2/wt-gdn-backward
export TOTAL_TRAINING_STEPS=5 TEST_FREQ=5 VAL_BEFORE_TRAIN=True

# Keep the proven round-4 recipe and alter only the candidate worktree,
# bounded step count, file-only logger, and isolated output/run names.
sed \
  -e "s|WT=\$BASE/megatron_lite/wt-qwen35-dapo|WT=$TARGET_WT|" \
  -e "s|export LOGGER='\[console,file,wandb\]' PROJECT_NAME=qwen35-dapo-ab RUN_NAME=g2_mlite_10step|export LOGGER='[console,file]' PROJECT_NAME=qwen35-dapo-ab RUN_NAME=g2_mlite_compiled_gdn_5step|" \
  -e 's|out_mlite|out_mlite_compiled_gdn_5step|g' \
  "$SOURCE" | /bin/bash -s -- "$@"

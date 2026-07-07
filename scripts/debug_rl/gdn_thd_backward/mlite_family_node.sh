#!/usr/bin/env bash
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
SOURCE=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/mlite_node_replay_base.sh
TARGET_WT=$BASE/qwen35_dapo_g2/wt-gdn-backward

# Keep the proven replay launcher byte-for-byte apart from the experiment worktree,
# family diagnostic switch, and output directory. Keep RUN_NAME unchanged
# because VERL includes it in the serialized replay-cache key.
sed \
  -e "s|WT=\$BASE/qwen35_dapo_g2/wt-runtime-fix|WT=$TARGET_WT|" \
  -e '/export MLITE_GRAD_NORM_DEBUG=1/a export MLITE_GRAD_FAMILY_DEBUG=1' \
  -e 's/out_mlite/out_mlite_gdn_family/g' \
  "$SOURCE" | /bin/bash -s -- "$@"

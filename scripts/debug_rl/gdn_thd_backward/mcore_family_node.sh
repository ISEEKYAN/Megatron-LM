#!/usr/bin/env bash
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
SOURCE=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/megatron_node_replay_base.sh
TARGET_MCORE=$BASE/qwen35_dapo_g2/mcore-gdn-family

sed \
  -e "s|MCORE=\$BASE/qwen35_dapo_ab/megatron_nvidia_dev|MCORE=$TARGET_MCORE|" \
  -e '/export HYDRA_FULL_ERROR=1/a export MCORE_GRAD_FAMILY_DEBUG=1' \
  -e 's/out_megatron_replay/out_megatron_gdn_family/g' \
  "$SOURCE" | /bin/bash -s -- "$@"

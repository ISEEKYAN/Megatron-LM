#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J gdn_runtime_audit
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/gdn_runtime_audit_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
REPO=$BASE/qwen35_dapo_g2/wt-gdn-backward
MCORE=$BASE/qwen35_dapo_g2/mcore-gdn-family
MBRIDGE=$BASE/qwen35_dapo_ab/mbridge_main

echo "AUDIT_REPO_HEAD=$(git -C "$REPO" rev-parse HEAD)"
echo "AUDIT_MCORE_HEAD=$(git -C "$MCORE" rev-parse HEAD)"
echo "AUDIT_MBRIDGE_HEAD=$(git -C "$MBRIDGE" rev-parse HEAD)"
echo AUDIT_STATUS_BEGIN
git -C "$REPO" status --short
echo AUDIT_STATUS_END

echo AUDIT_PROTOCOL_BEGIN
sed -n '160,185p' "$REPO/experimental/lite/megatron/lite/model/qwen3_5/lite/protocol.py"
echo AUDIT_PROTOCOL_END

echo AUDIT_MLITE_GDN_BEGIN
sed -n '90,215p' "$REPO/experimental/lite/megatron/lite/primitive/modules/gated_delta_net.py"
echo AUDIT_MLITE_GDN_END

echo AUDIT_MCORE_GDN_BEGIN
grep -R -n -E 'causal_conv1d|deterministic_mode|_prepare_qkv_for_gated_delta_rule' \
  "$MCORE/megatron" "$MBRIDGE" --include='*.py' 2>/dev/null | head -n 360
echo AUDIT_MCORE_GDN_END

echo AUDIT_CORRECTNESS_BEGIN
grep -R -n -E 'MLITE_CORRECTNESS_NONDETERMINISTIC|MLITE_BENCH_PACKED_SEQ_LENS|activation.probe|activation_probes' \
  "$REPO/experimental/lite/examples/bench" "$REPO/experimental/lite/megatron/lite" \
  --include='*.py' 2>/dev/null | head -n 360
echo AUDIT_CORRECTNESS_END

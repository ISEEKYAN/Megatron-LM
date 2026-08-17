#!/usr/bin/env bash
# Run the shared-FC dist-opt DCP proof without scheduler assumptions; requires 2 CUDA GPUs.
set -euo pipefail

test_path="${1:-experimental/lite/tests/smoke/primitive/test_shared_fc_bank_distopt_dcp_gpu.py}"
artifact_dir="$(mktemp -d)"
cleanup() { rm -rf "$artifact_dir"; }
trap cleanup EXIT

run_phase() {
  local phase="$1"
  MLITE_TEST_HARNESS=1 \
  MLITE_SHARED_FC_DCP_PHASE="$phase" \
  MLITE_SHARED_FC_DCP_ARTIFACT_DIR="$artifact_dir" \
  torchrun --standalone --nproc_per_node=2 -m pytest -q -rA "$test_path"
  printf 'SHARED_FC_DCP_2_GPU_%s_COMPLETE\n' "${phase^^}"
}

printf 'SHARED_FC_DCP_REQUIRES_2_GPUS\n'
run_phase save
run_phase load

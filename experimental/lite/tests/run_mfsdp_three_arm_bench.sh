#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "M-FSDP three-arm benchmark must run inside a Slurm allocation." >&2
  exit 2
fi

MODE="${1:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ "${SLURM_NNODES:-0}" != "1" || "${NPROC_PER_NODE}" != "8" ]]; then
  echo "M-FSDP three-arm benchmark requires one node and NPROC_PER_NODE=8." >&2
  exit 2
fi
: "${MCORE_SOURCE_ROOT:?point to NVIDIA Megatron-LM commit 00309a source}"
: "${MCORE_COMMIT_FILE:?point to the staged MCore commit marker}"
: "${MLITE_COMMIT:?set the staged MLite base commit}"

case "${MODE}" in
  three-arm)
    TEST_EXPR="three_arm_torch_adamw_benchmark"
    ;;
  ablation)
    TEST_EXPR="feature_ablation"
    ;;
  *)
    echo "usage: $0 {three-arm|ablation}" >&2
    exit 2
    ;;
esac

export PYTHONPATH="${ROOT}/experimental/lite:${MCORE_SOURCE_ROOT}:${PYTHONPATH:-}"
export MLITE_RUN_SMOKE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m pytest \
  -c /dev/null \
  --rootdir="${ROOT}" \
  --confcutdir="${ROOT}" \
  -q -s -rs \
  experimental/lite/tests/smoke/primitive/test_mfsdp_three_arm_bench.py \
  -k "${TEST_EXPR}"

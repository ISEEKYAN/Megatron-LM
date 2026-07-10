#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "M-FSDP Hopper validation must run inside a Slurm allocation." >&2
  exit 2
fi

MODE="${1:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
export PYTHONPATH="${ROOT}/experimental/lite:${PYTHONPATH:-}"
export MLITE_RUN_SMOKE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

case "${MODE}" in
  throughput)
    if [[ "${NNODES}" != "1" || "${NPROC_PER_NODE}" != "8" ]]; then
      echo "throughput mode requires NNODES=1 and NPROC_PER_NODE=8." >&2
      exit 2
    fi
    TEST_EXPR="throughput_exceeds_fsdp2"
    ;;
  full-parallel)
    if [[ "${NNODES}" != "1" || "${NPROC_PER_NODE}" != "8" ]]; then
      echo "full-parallel mode requires NNODES=1 and NPROC_PER_NODE=8." >&2
      exit 2
    fi
    TEST_EXPR="full_parallel_precision_curve"
    ;;
  *)
    echo "usage: $0 {throughput|full-parallel}" >&2
    exit 2
    ;;
esac

DIST_ARGS=(--nproc_per_node="${NPROC_PER_NODE}")
if [[ "${NNODES}" == "1" ]]; then
  DIST_ARGS+=(--standalone)
else
  NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-}}"
  : "${NODE_RANK:?set NODE_RANK or launch one Slurm task per node}"
  : "${MASTER_ADDR:?set MASTER_ADDR to the first allocated node}"
  MASTER_PORT="${MASTER_PORT:-29571}"
  DIST_ARGS+=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi

python -m torch.distributed.run "${DIST_ARGS[@]}" \
  -m pytest \
  -c /dev/null \
  --rootdir="${ROOT}" \
  --confcutdir="${ROOT}" \
  -q -s -rs \
  experimental/lite/tests/smoke/primitive/test_mfsdp_parity_smoke.py \
  -k "${TEST_EXPR}"

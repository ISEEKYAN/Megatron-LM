#!/usr/bin/env bash
#SBATCH --job-name=ds4-resync-tp4
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00

set -euo pipefail

BASE_SQSH=${BASE_SQSH:?set BASE_SQSH}
VLLM_OVERLAY=${VLLM_OVERLAY:?set VLLM_OVERLAY}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR}
MLITE_SRC=${MLITE_SRC:?set MLITE_SRC}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
RESYNC_DIR=${OUTPUT_DIR}/resync-checkpoint

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${MLITE_SRC}/experimental/lite:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_DEEP_GEMM_WARMUP || true

run_container() {
  srun --mpi=pmix --export=ALL --no-container-mount-home \
    --container-image="${BASE_SQSH}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${MLITE_SRC}" \
    bash -lc "source '${VLLM_OVERLAY}/bin/activate' && $1"
}

if [[ "${COMPARE_ONLY:-0}" == 1 ]]; then
  [[ -s "${OUTPUT_DIR}/direct.pt" && -s "${OUTPUT_DIR}/resync.pt" ]] || {
    echo "COMPARE_ONLY requires non-empty direct.pt and resync.pt" >&2
    exit 6
  }
elif [[ "${SKIP_DIRECT:-0}" == 1 ]]; then
  run_container "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_resync_tp4 \
    convert --source '${CHECKPOINT_DIR}' --output '${RESYNC_DIR}'"
  [[ -s "${OUTPUT_DIR}/direct.pt" ]] || {
    echo "SKIP_DIRECT requires an existing non-empty direct.pt" >&2
    exit 5
  }
  echo "DS4_TP4_DIRECT_REUSED=${OUTPUT_DIR}/direct.pt"
else
  run_container "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_resync_tp4 \
    convert --source '${CHECKPOINT_DIR}' --output '${RESYNC_DIR}'"
  run_container "python -m examples.verl.ds4_resync_tp4 collect \
    --model '${CHECKPOINT_DIR}' --output '${OUTPUT_DIR}/direct.pt'"
fi
if [[ "${COMPARE_ONLY:-0}" != 1 ]]; then
  run_container "python -m examples.verl.ds4_resync_tp4 collect \
    --model '${RESYNC_DIR}' --output '${OUTPUT_DIR}/resync.pt'"
fi
run_container "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_resync_tp4 \
  compare --reference '${OUTPUT_DIR}/direct.pt' \
  --candidate '${OUTPUT_DIR}/resync.pt' --output '${OUTPUT_DIR}/report.json'"

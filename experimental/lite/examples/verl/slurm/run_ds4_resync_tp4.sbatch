#!/usr/bin/env bash
#SBATCH --job-name=ds4-resync-tp4
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00

set -euo pipefail

BASE_SQSH=${BASE_SQSH:?set BASE_SQSH}
VLLM_OVERLAY=${VLLM_OVERLAY:?set VLLM_OVERLAY}
MLITE_OVERLAY=${MLITE_OVERLAY:?set MLITE_OVERLAY}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR}
MLITE_SRC=${MLITE_SRC:?set MLITE_SRC}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
RESYNC_DIR=${OUTPUT_DIR}/resync-checkpoint

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${MLITE_SRC}/experimental/lite:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_DEEP_GEMM_WARMUP || true

run_container() {
  local overlay=$1
  local command=$2
  srun --mpi=pmix --export=ALL --no-container-mount-home \
    --container-image="${BASE_SQSH}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${MLITE_SRC}" \
    bash -lc "source '${overlay}/bin/activate' && ${command}"
}

if [[ "${COMPARE_ONLY:-0}" == 1 ]]; then
  [[ -s "${OUTPUT_DIR}/direct.pt" && -s "${OUTPUT_DIR}/resync.pt" \
    && -s "${OUTPUT_DIR}/mlite.pt" ]] || {
    echo "COMPARE_ONLY requires non-empty direct.pt, resync.pt, and mlite.pt" >&2
    exit 6
  }
else
  [[ ! -e "${RESYNC_DIR}" ]] || {
    echo "refusing to reuse existing resync checkpoint: ${RESYNC_DIR}" >&2
    exit 5
  }
  run_container "${MLITE_OVERLAY}" \
    "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_resync_tp4 \
      verify --model '${CHECKPOINT_DIR}'"
  run_container "${MLITE_OVERLAY}" \
    "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_checkpoint_roundtrip \
      --checkpoint '${CHECKPOINT_DIR}' --output '${OUTPUT_DIR}/weight-report.json'"
  run_container "${MLITE_OVERLAY}" \
    "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_resync_tp4 \
      convert --source '${CHECKPOINT_DIR}' --output '${RESYNC_DIR}'"
  run_container "${VLLM_OVERLAY}" \
    "python -m examples.verl.ds4_resync_tp4 collect \
      --model '${CHECKPOINT_DIR}' --output '${OUTPUT_DIR}/direct.pt' \
      --resync-model '${RESYNC_DIR}' --resync-output '${OUTPUT_DIR}/resync.pt'"
  run_container "${MLITE_OVERLAY}" \
    "python -m torch.distributed.run --standalone --nproc_per_node=4 \
      -m examples.verl.ds4_resync_tp4 collect-mlite \
      --model '${CHECKPOINT_DIR}' --output '${OUTPUT_DIR}/mlite.pt'"
fi
run_container "${MLITE_OVERLAY}" \
  "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_resync_tp4 \
    compare --reference '${OUTPUT_DIR}/direct.pt' \
    --candidate '${OUTPUT_DIR}/resync.pt' --mlite '${OUTPUT_DIR}/mlite.pt' \
    --output '${OUTPUT_DIR}/report.json'"

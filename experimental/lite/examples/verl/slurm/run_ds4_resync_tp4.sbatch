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
MLITE_COMMIT=${MLITE_COMMIT:?set MLITE_COMMIT}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}
RESYNC_DIR=${OUTPUT_DIR}/resync-checkpoint
VLLM_COMPLETE=${OUTPUT_DIR}/DS4_TP4_VLLM_COMPLETE

[[ "$(git -C "${MLITE_SRC}" rev-parse HEAD)" == "${MLITE_COMMIT}" ]] || {
  echo "MLite source is not at ${MLITE_COMMIT}" >&2
  exit 7
}
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
  [[ -s "${OUTPUT_DIR}/cold-load.pt" && -s "${OUTPUT_DIR}/online-reload.pt" \
    && -s "${OUTPUT_DIR}/mlite.pt" \
    && -s "${OUTPUT_DIR}/engine-weight-report.json" \
    && -s "${OUTPUT_DIR}/export-coverage.json" \
    && -s "${VLLM_COMPLETE}" ]] || {
    echo "COMPARE_ONLY requires three logprob artifacts and both weight reports" >&2
    exit 6
  }
else
  [[ ! -e "${RESYNC_DIR}" ]] || {
    echo "refusing to reuse existing resync checkpoint: ${RESYNC_DIR}" >&2
    exit 5
  }
  run_container "${MLITE_OVERLAY}" \
    "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_checkpoint_roundtrip \
      --checkpoint '${CHECKPOINT_DIR}' --target-expert-dtype fp8 \
      --output '${OUTPUT_DIR}/weight-report.json'"
  run_container "${MLITE_OVERLAY}" \
    "python -m torch.distributed.run --standalone --nproc_per_node=4 \
      -m examples.verl.ds4_resync_tp4 collect-mlite \
      --model '${CHECKPOINT_DIR}' --output '${OUTPUT_DIR}/mlite.pt' \
      --fp8-output '${RESYNC_DIR}' \
      --coverage-output '${OUTPUT_DIR}/export-coverage.json'"
  run_container "${MLITE_OVERLAY}" \
    "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_resync_tp4 \
      verify --model '${RESYNC_DIR}'"
  run_container "${VLLM_OVERLAY}" \
    "python -m examples.verl.ds4_resync_tp4 collect --model '${RESYNC_DIR}' \
      --output '${OUTPUT_DIR}/cold-load.pt' --resync-model '${RESYNC_DIR}' \
      --resync-output '${OUTPUT_DIR}/online-reload.pt' \
      --weight-output '${OUTPUT_DIR}/engine-weight-report.json'"
  [[ -s "${OUTPUT_DIR}/cold-load.pt" && -s "${OUTPUT_DIR}/online-reload.pt" \
    && -s "${OUTPUT_DIR}/engine-weight-report.json" \
    && -s "${VLLM_COMPLETE}" ]] || {
    echo "vLLM collect exited without durable completion artifacts" >&2
    exit 8
  }
fi
run_container "${MLITE_OVERLAY}" \
  "CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_resync_tp4 \
    compare --cold '${OUTPUT_DIR}/cold-load.pt' \
    --online '${OUTPUT_DIR}/online-reload.pt' --mlite '${OUTPUT_DIR}/mlite.pt' \
    --weights '${OUTPUT_DIR}/engine-weight-report.json' \
    --coverage '${OUTPUT_DIR}/export-coverage.json' \
    --output '${OUTPUT_DIR}/report.json'"

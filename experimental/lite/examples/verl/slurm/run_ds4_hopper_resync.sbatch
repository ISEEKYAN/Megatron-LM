#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=ds4_hopper_resync

set -euo pipefail

BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to the Hopper-qualified container}
MLITE_SRC=${MLITE_SRC:?set MLITE_SRC to a committed Megatron Lite snapshot}
MLITE_COMMIT=${MLITE_COMMIT:?set MLITE_COMMIT to the snapshot commit}
MEGATRON_ROOT=${MEGATRON_ROOT:?set MEGATRON_ROOT to a compatible MCore snapshot}
VERL_ROOT=${VERL_ROOT:?set VERL_ROOT to a committed VERL snapshot}
MLITE_SM90_SITE=${MLITE_SM90_SITE:?set MLITE_SM90_SITE to the SM90 training overlay}
DS4_VLLM_SITE=${DS4_VLLM_SITE:?set DS4_VLLM_SITE to the DS4 vLLM thin overlay}
DS4_VLLM_SHIM=${DS4_VLLM_SHIM:?set DS4_VLLM_SHIM to the Torch/vLLM ABI shim}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the official mixed checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a fresh result directory}

for path in \
  "${BASE_IMAGE}" \
  "${MLITE_SRC}" \
  "${MEGATRON_ROOT}" \
  "${VERL_ROOT}" \
  "${MLITE_SM90_SITE}" \
  "${DS4_VLLM_SITE}" \
  "${DS4_VLLM_SHIM}" \
  "${CHECKPOINT_DIR}"; do
  [[ -e "${path}" ]] || {
    echo "required Hopper validation path does not exist: ${path}" >&2
    exit 2
  }
done
[[ "$(git -C "${MLITE_SRC}" rev-parse HEAD)" == "${MLITE_COMMIT}" ]] || {
  echo "MLite source is not at ${MLITE_COMMIT}" >&2
  exit 7
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "refusing to reuse existing output directory: ${OUTPUT_DIR}" >&2
  exit 5
}

export BASE_IMAGE MLITE_SRC MLITE_COMMIT MEGATRON_ROOT VERL_ROOT
export MLITE_SM90_SITE DS4_VLLM_SITE DS4_VLLM_SHIM CHECKPOINT_DIR OUTPUT_DIR

srun \
  --nodes=1 \
  --ntasks=1 \
  --container-image="${BASE_IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${MLITE_SRC}" \
  bash -lc '
    set -euo pipefail
    export PATH=/usr/local/bin:/usr/bin:/bin
    hash -r
    export TMPDIR=/tmp/d4h-${SLURM_JOB_ID}
    export PYTHONPYCACHEPREFIX=${TMPDIR}/pycache
    export TORCHINDUCTOR_CACHE_DIR=${TMPDIR}/torchinductor
    export TRITON_CACHE_DIR=${TMPDIR}/triton
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_NVLS_ENABLE=0
    export PYTHONNOUSERSITE=1
    unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    mkdir -p "${PYTHONPYCACHEPREFIX}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

    echo "DS4_HOPPER_RESYNC_START source=${MLITE_COMMIT} h100s=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "python=$(command -v python) torchrun=$(command -v torchrun)"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    CHECK_ONLY=1 bash experimental/lite/examples/verl/scripts/run_deepseek_v4_hopper_resync.sh
    NPROC_PER_NODE=2 bash experimental/lite/examples/verl/scripts/run_deepseek_v4_hopper_resync.sh
  '


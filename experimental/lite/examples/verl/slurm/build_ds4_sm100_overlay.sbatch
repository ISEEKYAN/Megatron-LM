#!/usr/bin/env bash
#SBATCH --job-name=mlite-dsa-sm100-build
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/llmrl/gb200-vllm-ds4/logs/mlite-dsa-sm100-build-%j.log

set -euo pipefail
SCRIPT_DIR=${DS4_SM100_SCRIPT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/llmrl/gb200-vllm-ds4/src/Megatron-LM-ds4-sm100/experimental/lite/examples/verl/slurm}
# shellcheck source=ds4_sm100_versions.env
source "${SCRIPT_DIR}/ds4_sm100_versions.env"

[[ -s "${BASE_SQSH}" ]] || { echo "missing base image: ${BASE_SQSH}" >&2; exit 2; }
[[ -f "${VLLM_OVERLAY}/BUILD_OK" ]] || {
  echo "missing verified vLLM overlay: ${VLLM_OVERLAY}" >&2
  exit 3
}
[[ -d "${VLLM_SITE}/cudnn/deepseek_sparse_attention" ]] || {
  echo "vLLM overlay does not contain cudnn-frontend DSA" >&2
  exit 4
}
[[ "$(git -C "${MLITE_SRC}" rev-parse HEAD)" == "${MLITE_COMMIT:?set MLITE_COMMIT}" ]] || {
  echo "MLite source is not at ${MLITE_COMMIT}" >&2
  exit 5
}

mkdir -p "${ROOT}/logs" "${ROOT}/overlays" "${ROOT}/src"
if [[ ! -e "${FLASHMLA_SRC}/.git" ]]; then
  git clone --filter=blob:none "${FLASHMLA_REPO}" "${FLASHMLA_SRC}"
fi
git -C "${FLASHMLA_SRC}" fetch origin "${FLASHMLA_COMMIT}"
git -C "${FLASHMLA_SRC}" checkout --detach "${FLASHMLA_COMMIT}"
git -C "${FLASHMLA_SRC}" submodule update --init csrc/cutlass
[[ "$(git -C "${FLASHMLA_SRC}" rev-parse HEAD)" == "${FLASHMLA_COMMIT}" ]]

srun --mpi=pmix --export=ALL --no-container-mount-home \
  --container-image="${BASE_SQSH}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${MLITE_SRC}" \
  bash -s -- "${MLITE_OVERLAY}" "${MLITE_SITE}" "${VLLM_SITE}" \
    "${FLASHMLA_SRC}" "${MLITE_SRC}" "${FLASHMLA_COMMIT}" <<'BUILD'
set -euo pipefail
overlay=$1
site=$2
vllm_site=$3
flashmla_src=$4
mlite_src=$5
flashmla_commit=$6

python - <<'PY'
import torch

print(f"NGC_TORCH={torch.__version__}")
print(f"NGC_CUDA={torch.version.cuda}")
print(f"GPU={torch.cuda.get_device_name(0)}")
print(f"GPU_CAPABILITY={torch.cuda.get_device_capability(0)}")
assert torch.cuda.get_device_capability(0)[0] == 10
PY

if [[ ! -x "${overlay}/bin/python" ]]; then
  python -m venv --system-site-packages "${overlay}"
fi
{
  printf '%s\n' "${vllm_site}"
  printf '%s\n' "${vllm_site}/nvidia_cutlass_dsl/python_packages"
} > "${site}/gb200_vllm_overlay.pth"
source "${overlay}/bin/activate"
export FLASH_MLA_DISABLE_SM90=1
# CUDA 13 installs libcudacxx below include/cccl. NVCC discovers it, but the
# host C++ compilation of FlashMLA's API translation unit does not.
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl:${CPLUS_INCLUDE_PATH:-}
export MAX_JOBS=${MAX_JOBS:-32}
export NVCC_THREADS=${NVCC_THREADS:-2}
python -m pip install --no-build-isolation --no-deps "${flashmla_src}"

export PYTHONPATH="${mlite_src}/experimental/lite:${PYTHONPATH:-}"
python -m examples.verl.ds4_sm100_env probe
{
  printf 'FLASHMLA_COMMIT=%s\n' "${flashmla_commit}"
  python - <<'PY'
from importlib.metadata import version

print(f"FLASH_MLA_VERSION={version('flash-mla')}")
print(f"CUDNN_FRONTEND_VERSION={version('nvidia-cudnn-frontend')}")
PY
} > "${overlay}/BUILD_OK"
echo "MLITE_DSA_SM100_OVERLAY_READY=${overlay}"
BUILD

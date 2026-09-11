#!/usr/bin/env bash
#SBATCH --job-name=ds41-fp4-codecs
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00

# Environment inherited from the verified GB200 official smoke recipe.
set -euo pipefail
SCRIPT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/llmrl/gb200-vllm-ds4/scripts/gb200-vllm-ds4
source "${SCRIPT_DIR}/versions.env"
: "${DS41_SOURCE:?immutable source snapshot required}"
[[ -s "${BASE_SQSH}" ]]
[[ -f "${VLLM_OVERLAY}/BUILD_OK" ]]
export PYTHONPATH="${DS41_SOURCE}/experimental/lite"
unset VLLM_DEEP_GEMM_WARMUP || true
srun --mpi=pmix --export=ALL --no-container-mount-home \
  --container-image="${BASE_SQSH}" \
  --container-mounts="${ROOT}:${ROOT},${SCRIPT_DIR}:${SCRIPT_DIR}" \
  --container-workdir="${DS41_SOURCE}" \
  bash -lc "source '${VLLM_OVERLAY}/bin/activate' && python '${DS41_SOURCE}/experimental/lite/tools/deepseek_v41/validate_fp4_gpu.py' --official-kernel '${DS41_SOURCE}/reference/kernel.py' --expected-sha256 1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"

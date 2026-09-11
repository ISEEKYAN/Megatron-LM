#!/usr/bin/env bash
#SBATCH --job-name=ds41-fixture
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00

set -euo pipefail
SCRIPT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/llmrl/gb200-vllm-ds4/scripts/gb200-vllm-ds4
source "${SCRIPT_DIR}/versions.env"
: "${DS41_SOURCE:?immutable source snapshot required}"
[[ -s "${BASE_SQSH}" ]]
[[ -f "${VLLM_OVERLAY}/BUILD_OK" ]]
export PYTHONPATH="${DS41_SOURCE}/experimental/lite"
srun --mpi=pmix --export=ALL --no-container-mount-home \
  --container-image="${BASE_SQSH}" \
  --container-mounts="${ROOT}:${ROOT},${SCRIPT_DIR}:${SCRIPT_DIR}" \
  --container-workdir="${DS41_SOURCE}" \
  bash -lc "source '${VLLM_OVERLAY}/bin/activate' && OMP_NUM_THREADS=4 python '${DS41_SOURCE}/experimental/lite/tools/deepseek_v41/fixtures.py' --reference-dir '${DS41_SOURCE}/reference' --output '${DS41_SOURCE}/fixture'"

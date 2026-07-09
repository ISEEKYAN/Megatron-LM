#!/usr/bin/env bash
#SBATCH --job-name=ds4-mlite-bf16-sm100
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/llmrl/gb200-vllm-ds4/logs/ds4-mlite-bf16-sm100-%j.log

set -euo pipefail
SCRIPT_DIR=${DS4_SM100_SCRIPT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/llmrl/gb200-vllm-ds4/src/Megatron-LM-ds4-sm100/experimental/lite/examples/verl/slurm}
# shellcheck source=ds4_sm100_versions.env
source "${SCRIPT_DIR}/ds4_sm100_versions.env"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-${MODEL_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a new result directory}
[[ -s "${BASE_SQSH}" ]] || { echo "missing base image: ${BASE_SQSH}" >&2; exit 2; }
[[ -f "${MLITE_OVERLAY}/BUILD_OK" ]] || {
  echo "missing verified MLite overlay: ${MLITE_OVERLAY}" >&2
  exit 3
}
[[ -s "${CHECKPOINT_DIR}/model.safetensors.index.json" ]] || {
  echo "missing official checkpoint: ${CHECKPOINT_DIR}" >&2
  exit 4
}
[[ ! -e "${OUTPUT_DIR}/mlite.pt" && ! -e "${OUTPUT_DIR}/resync-checkpoint" ]] || {
  echo "refusing to reuse an existing formal output: ${OUTPUT_DIR}" >&2
  exit 5
}
[[ "$(git -C "${MLITE_SRC}" rev-parse HEAD)" == "${MLITE_COMMIT:?set MLITE_COMMIT}" ]] || {
  echo "MLite source is not at ${MLITE_COMMIT}" >&2
  exit 6
}
mkdir -p "${OUTPUT_DIR}"

srun --mpi=pmix --export=ALL --no-container-mount-home \
  --container-image="${BASE_SQSH}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${MLITE_SRC}" \
  bash -lc "
    set -euo pipefail
    source '${MLITE_OVERLAY}/bin/activate'
    export PYTHONPATH='${MLITE_SRC}/experimental/lite':\"\${PYTHONPATH:-}\"
    python -m examples.verl.ds4_sm100_env probe
    python -m torch.distributed.run --standalone --nproc_per_node=4 \\
      -m examples.verl.ds4_resync_tp4 collect-mlite \\
      --model '${CHECKPOINT_DIR}' \\
      --output '${OUTPUT_DIR}/mlite.pt' \\
      --fp8-output '${OUTPUT_DIR}/resync-checkpoint'
    python -m examples.verl.ds4_sm100_env verify-payload \\
      --payload '${OUTPUT_DIR}/mlite.pt' --expected-prompts 36
    echo DS4_MLITE_SM100_FORWARD_OK
  "

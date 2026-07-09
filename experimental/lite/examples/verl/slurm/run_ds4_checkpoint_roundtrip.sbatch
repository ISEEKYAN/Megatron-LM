#!/bin/bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --job-name=ds4-official-roundtrip

set -euo pipefail

BASE_SQSH=${BASE_SQSH:?set BASE_SQSH}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set CHECKPOINT_DIR}
MLITE_SRC=${MLITE_SRC:?set MLITE_SRC}
OUTPUT_JSON=${OUTPUT_JSON:?set OUTPUT_JSON}

mkdir -p "$(dirname "${OUTPUT_JSON}")"
export PYTHONPATH="${MLITE_SRC}/experimental/lite:${PYTHONPATH:-}"

# HSG's batch QOS allocates complete four-GPU nodes.  The audit intentionally
# uses one visible GPU and streams one safetensors shard at a time.
srun --mpi=pmix --export=ALL --no-container-mount-home \
  --container-image="${BASE_SQSH}" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="${MLITE_SRC}" \
  bash -lc "CUDA_VISIBLE_DEVICES=0 python -m \
    examples.verl.ds4_checkpoint_roundtrip \
    --checkpoint '${CHECKPOINT_DIR}' \
    --output '${OUTPUT_JSON}' \
    --device cuda:0"

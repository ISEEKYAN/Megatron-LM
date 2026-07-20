#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -J glm52_weight_roundtrip
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/llmrl_runs/%x_%j.out
set -euxo pipefail
B=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
RUN=$B/code/runtime/glm52-cpnative-validate
BASE_IMAGE=$B/code/env/pytorch_26.04-py3.sqsh
MLITE_SRC=$B/code/runtime/glm52-mlite-src
OUT=$B/code/llmrl_runs/glm52_weight_roundtrip_${SLURM_JOB_ID}.json
srun --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --container-image="$BASE_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --container-workdir="$MLITE_SRC/experimental/lite" \
  bash -lc "set -euo pipefail; source $RUN/env-common.sh; \
    python docs/runs/glm5_dsa_cp_native/validate_weight_roundtrip.py --output $OUT"

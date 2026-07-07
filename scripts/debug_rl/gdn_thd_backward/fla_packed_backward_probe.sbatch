#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J gdn_thd_bwd
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/gdn_thd_bwd_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
PROBE=${PROBE:?set PROBE to the shared-filesystem probe path}

srun --container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL \
  env PYTHONNOUSERSITE=1 PYTHONPATH="/vllm:$CP_SITE" \
  LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}" \
  python "$PROBE"

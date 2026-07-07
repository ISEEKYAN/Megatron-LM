#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J l2norm_autotune
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/l2norm_autotune_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
PROBE=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/l2norm_autotune_probe.py
OUT=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/l2norm_autotune_${SLURM_JOB_ID}
mkdir -p "$OUT"

SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
for tokens in 4096 8192; do
  srun "${SA[@]}" env PYTHONNOUSERSITE=1 PYTHONPATH="$CP_SITE:/vllm" \
    FLA_CACHE_RESULTS=0 TRITON_CACHE_DIR="$OUT/cache_$tokens" \
    python "$PROBE" --tokens "$tokens"
done
echo "L2NORM_AUTOTUNE_OUTPUT=$OUT"

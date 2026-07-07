#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J replay_l2_configs
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/replay_l2_configs_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site
PROBE=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/replay_l2norm_configs.py
DUMP_JOB=${DUMP_JOB:?set DUMP_JOB to the completed gdn_conv_probe job id}
DUMPS=$BASE/qwen35_dapo_g2/diagnosis/task_1_1_15_9/gdn_conv_probe_${DUMP_JOB}

srun --container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL \
  env PYTHONNOUSERSITE=1 PYTHONPATH="$CP_SITE:/vllm" FLA_CACHE_RESULTS=0 \
  TRITON_CACHE_DIR="$DUMPS/replay-cache-${SLURM_JOB_ID}" \
  python "$PROBE" \
  --mlite-dump "$DUMPS/mlite_call1.pt" \
  --mbridge-dump "$DUMPS/mbridge_call1.pt"

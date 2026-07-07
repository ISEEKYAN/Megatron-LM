#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -J weighted_logprob
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/qwen35_dapo_g2/logs/weighted_logprob_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=$BASE/verl_optimize/verl.vllm023.sqsh
REPO=$BASE/qwen35_dapo_g2/wt-gdn-backward
CP_SITE=$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site

srun --container-image="$IMG" --container-mounts=/lustre:/lustre \
  --container-workdir="$REPO/experimental/lite" --export=ALL \
  env PYTHONNOUSERSITE=1 PYTHONPATH="$REPO/experimental/lite:$REPO:$CP_SITE:/vllm" \
  python -m pytest -q \
  "tests/unit/examples/test_bench_example.py::test_correctness_weighted_logprob_loss_matches_backend_output_layouts" \
  tests/unit/primitive/test_train_step_unit.py \
  tests/unit/primitive/test_fsdp2_unit.py

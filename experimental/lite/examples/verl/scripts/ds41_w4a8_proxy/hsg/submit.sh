#!/usr/bin/env bash
set -euo pipefail
DS41_BUNDLE=${DS41_BUNDLE:?set the shared release directory}
DS41_OUTPUT=${1:?usage: bash hsg/submit.sh /absolute/shared/output}
: "${DS41_SLURM_ACCOUNT:?set DS41_SLURM_ACCOUNT to your HSG project account}"
: "${WANDB_API_KEY:?export W&B credentials before submitting}"
test "${DS41_OUTPUT:0:1}" = /
test ! -e "$DS41_OUTPUT"
source "$DS41_BUNDLE/paths.env"
export DS41_BUNDLE DS41_RELEASE DS41_DATA_DIR DS41_DATA_FILE
mkdir -p "$DS41_OUTPUT/logs"
DS41_CPU_JOB=$(sbatch --parsable --account="$DS41_SLURM_ACCOUNT" -p cpu_datamover --exclude=cpu-dm-0003 \
 --nodes=2 --ntasks-per-node=1 --gpus=0 --cpus-per-task=16 --mem=128G --time=01:00:00 \
 --output="$DS41_OUTPUT/logs/cpu-%j.out" \
 "$DS41_BUNDLE/recipe/hsg/cluster.sbatch" "$DS41_OUTPUT" cpu)
DS41_GPU_JOB=$(sbatch --parsable --account="$DS41_SLURM_ACCOUNT" -p batch --qos interactive \
 --dependency="afterok:$DS41_CPU_JOB" --kill-on-invalid-dep=yes \
 --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=32 --mem=0 --time=01:20:00 \
 --output="$DS41_OUTPUT/logs/gpu-%j.out" \
 "$DS41_BUNDLE/recipe/hsg/cluster.sbatch" "$DS41_OUTPUT" gpu)
printf 'CPU gate: %s\nGPU C8: %s (after successful CPU gate)\nResults: %s/logs\n' \
 "$DS41_CPU_JOB" "$DS41_GPU_JOB" "$DS41_OUTPUT"

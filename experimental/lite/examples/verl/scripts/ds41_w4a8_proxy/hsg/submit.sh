#!/usr/bin/env bash
set -euo pipefail
DS41_BUNDLE=${DS41_BUNDLE:?set the shared release directory}
DS41_LAYERS=2
if [[ "${1:-}" == --layers ]]; then
 DS41_LAYERS=${2:?--layers takes 2, 4, 6 or 40}
 shift 2
fi
DS41_OUTPUT=${1:?usage: bash hsg/submit.sh [--layers 2|4|6|40] /absolute/shared/output}
DS41_CPU_TIME=01:00:00
case "$DS41_LAYERS" in
 2) DS41_GPU_NODES=2 DS41_GPU_TIME=01:15:00 ;;
 4) DS41_GPU_NODES=2 DS41_GPU_TIME=02:00:00 ;;
 # Release subset [1,14,15,20,21,24] with 17408-token prompts; the CPU stage
 # writes ~0.27 TB of renumbered shards, and its trainable count comes from
 # the 6L CPU construction receipt.
 6)
  : "${DS41_PARAMETER_COUNTS:?6L needs the trainable count from its CPU construction receipt}"
  DS41_GPU_NODES=2 DS41_CPU_TIME=02:00:00 DS41_GPU_TIME=02:30:00
  ;;
 # Full model: EP64 on 16 GB200 nodes; see plan-40l for capacity and GPU-hour budget.
 40)
  : "${DS41_40L_GPU_AUTHORIZATION:?16-node jobs need an explicit authorization reference}"
  DS41_GPU_NODES=16 DS41_GPU_TIME=04:00:00
  echo 'WARNING: 40L fresh+resume run in one 4h job; HF load plus two stages can exceed batch MaxTime (see plan-40l).' >&2
  ;;
 *) echo 'DS41 C8 supports --layers 2, 4, 6 or 40' >&2; exit 2 ;;
esac
DS41_GPU_QOS=(--qos interactive)
if (( DS41_GPU_NODES > 2 )); then
 DS41_GPU_QOS=()
fi
: "${DS41_SLURM_ACCOUNT:?set DS41_SLURM_ACCOUNT to your HSG project account}"
: "${WANDB_API_KEY:?export W&B credentials before submitting}"
test "${DS41_OUTPUT:0:1}" = /
test ! -e "$DS41_OUTPUT"
source "$DS41_BUNDLE/paths.env"
export DS41_BUNDLE DS41_RELEASE DS41_DATA_DIR DS41_DATA_FILE DS41_LAYERS
if [[ -n "${DS41_PARAMETER_COUNTS:-}" ]]; then export DS41_PARAMETER_COUNTS; fi
mkdir -p "$DS41_OUTPUT/logs"
DS41_CPU_ARGS=()
if [[ -n "${DS41_CPU_EXCLUDE:-}" ]]; then
 DS41_CPU_ARGS+=(--exclude="$DS41_CPU_EXCLUDE")
fi
DS41_CPU_JOB=$(sbatch --parsable --account="$DS41_SLURM_ACCOUNT" -p cpu_datamover "${DS41_CPU_ARGS[@]}" \
 --nodes=2 --ntasks-per-node=1 --gpus=0 --cpus-per-task=16 --mem=128G --time="$DS41_CPU_TIME" \
 --output="$DS41_OUTPUT/logs/cpu-%j.out" \
 "$DS41_BUNDLE/recipe/hsg/cluster.sbatch" "$DS41_OUTPUT" cpu)
DS41_GPU_JOB=$(sbatch --parsable --account="$DS41_SLURM_ACCOUNT" -p batch "${DS41_GPU_QOS[@]}" \
 --dependency="afterok:$DS41_CPU_JOB" --kill-on-invalid-dep=yes \
 --nodes="$DS41_GPU_NODES" --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=32 --mem=0 --time="$DS41_GPU_TIME" \
 --output="$DS41_OUTPUT/logs/gpu-%j.out" \
 "$DS41_BUNDLE/recipe/hsg/cluster.sbatch" "$DS41_OUTPUT" gpu)
printf 'CPU gate: %s\nGPU C8: %s (after successful CPU gate)\nResults: %s/logs\n' \
 "$DS41_CPU_JOB" "$DS41_GPU_JOB" "$DS41_OUTPUT"

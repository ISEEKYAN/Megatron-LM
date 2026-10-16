#!/usr/bin/env bash
set -euo pipefail
DS41_BUNDLE=${DS41_BUNDLE:?set the shared release directory}
DS41_LAYERS=2
if [[ "${1:-}" == --layers ]]; then
 DS41_LAYERS=${2:?--layers takes 2, 4, 6, 8, 9 or 40}
 shift 2
fi
DS41_OUTPUT=${1:?usage: bash hsg/submit.sh [--layers 2|4|6|8|9|40] /absolute/shared/output}
DS41_CPU_TIME_DEFAULT=01:00:00
case "$DS41_LAYERS" in
 2) DS41_GPU_NODES=2 DS41_GPU_TIME_DEFAULT=01:15:00 ;;
 4) DS41_GPU_NODES=2 DS41_GPU_TIME_DEFAULT=02:00:00 ;;
 # Release subset [1,14,15,20,21,24] with 17408-token prompts; the CPU stage
 # writes ~0.27 TB of renumbered shards, and its trainable count comes from
 # the 6L CPU construction receipt.
 6)
  : "${DS41_PARAMETER_COUNTS:?6L needs the trainable count from its CPU construction receipt}"
  DS41_GPU_NODES=2 DS41_CPU_TIME_DEFAULT=02:00:00 DS41_GPU_TIME_DEFAULT=02:30:00
  ;;
 # Long-sequence proxy (9L; 8L fallback without layer 15): EP8xCP2, recompute.
 8|9)
  : "${DS41_PARAMETER_COUNTS:?${DS41_LAYERS}L needs the trainable count from its CPU construction receipt}"
  DS41_GPU_NODES=2 DS41_CPU_TIME_DEFAULT=02:00:00 DS41_GPU_TIME_DEFAULT=03:00:00
  ;;
 # Full model: EP64 on 16 GB200 nodes; see plan-40l for capacity and GPU-hour budget.
 40)
  : "${DS41_40L_GPU_AUTHORIZATION:?16-node jobs need an explicit authorization reference}"
  DS41_GPU_NODES=16 DS41_GPU_TIME_DEFAULT=04:00:00
  echo 'WARNING: 40L fresh+resume run in one 4h job; HF load plus two stages can exceed batch MaxTime (see plan-40l).' >&2
  ;;
 *) echo 'DS41 C8 supports --layers 2, 4, 6, 8, 9 or 40' >&2; exit 2 ;;
esac
# Resource overrides are public launcher inputs; model topology stays variant-defined.
DS41_CPU_TIME=${DS41_CPU_TIME:-$DS41_CPU_TIME_DEFAULT}
DS41_GPU_TIME=${DS41_GPU_TIME:-$DS41_GPU_TIME_DEFAULT}
DS41_CPU_CPUS=${DS41_CPU_CPUS:-16}
DS41_GPU_CPUS=${DS41_GPU_CPUS:-32}
DS41_CPU_MEM=${DS41_CPU_MEM:-128G}
DS41_GPU_MEM=${DS41_GPU_MEM:-0}
DS41_CPU_PARTITION=${DS41_CPU_PARTITION:-cpu_datamover}
DS41_GPU_PARTITION=${DS41_GPU_PARTITION:-batch}
DS41_CPU_QOS_ARGS=()
if [[ -n "${DS41_CPU_QOS:-}" ]]; then
 DS41_CPU_QOS_ARGS=(--qos "$DS41_CPU_QOS")
fi
# DS41_GPU_QOS='' selects the partition's default QOS for a 2-node job.
DS41_GPU_QOS_NAME=${DS41_GPU_QOS-interactive}
DS41_GPU_QOS=(--qos "$DS41_GPU_QOS_NAME")
if (( DS41_GPU_NODES > 2 )) || [[ -z "$DS41_GPU_QOS_NAME" ]]; then
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
DS41_CPU_JOB=$(sbatch --parsable --account="$DS41_SLURM_ACCOUNT" -p "$DS41_CPU_PARTITION" "${DS41_CPU_QOS_ARGS[@]}" "${DS41_CPU_ARGS[@]}" \
 --nodes=2 --ntasks-per-node=1 --gpus=0 --cpus-per-task="$DS41_CPU_CPUS" --mem="$DS41_CPU_MEM" --time="$DS41_CPU_TIME" \
 --output="$DS41_OUTPUT/logs/cpu-%j.out" \
 "$DS41_BUNDLE/recipe/hsg/cluster.sbatch" "$DS41_OUTPUT" cpu)
DS41_GPU_JOB=$(sbatch --parsable --account="$DS41_SLURM_ACCOUNT" -p "$DS41_GPU_PARTITION" "${DS41_GPU_QOS[@]}" \
 --dependency="afterok:$DS41_CPU_JOB" --kill-on-invalid-dep=yes \
 --nodes="$DS41_GPU_NODES" --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task="$DS41_GPU_CPUS" --mem="$DS41_GPU_MEM" --time="$DS41_GPU_TIME" \
 --output="$DS41_OUTPUT/logs/gpu-%j.out" \
 "$DS41_BUNDLE/recipe/hsg/cluster.sbatch" "$DS41_OUTPUT" gpu)
printf 'CPU gate: %s\nGPU C8: %s (after successful CPU gate)\nResults: %s/logs\n' \
 "$DS41_CPU_JOB" "$DS41_GPU_JOB" "$DS41_OUTPUT"

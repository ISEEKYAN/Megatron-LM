#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Branch C cheap experiment (TASK-1.13.8.1) — standalone vLLM sleep/wake per-cycle
# VRAM probe. Runs the DECISIVE A0/A1 pair back-to-back on ONE GPU so the only
# variable is expandable_segments True vs False:
#   A0: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   (Branch A's global lever)
#   A1: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# gmu is held at 0.7 (bayan 铁律). Budget ≈ 20 cycles × 2 arms on a 0.6B model
# ≈ well under 1 GPU-h. Reuses the proven vllm023 container recipe from
# kernel_rollout docs/runs (env-probe.sbatch / gb200.sbatch).
#
# Submit (single GPU is enough — leak is per-device, not multi-rank):
#   sbatch --export=ALL,MODEL=/lustre/.../Qwen3-0.6B run_cumem_cycle_probe.sbatch
#
#SBATCH --job-name=branchc-cumem-cycle-probe
#SBATCH --partition=batch
#SBATCH --account=coreai_devtech_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/kernel-rollout/branchc-cumem/evidence/%x-%j.out

set -euo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan
IMAGE=${VLLM_IMAGE:-$ROOT/rl-mlite/containers/vllm023-aarch64-dev1.sqsh}
# A small proxy model — leak mechanism is model-independent (bayan). Override with
# --export=ALL,MODEL=... ; must exist on lustre inside the container mount.
MODEL=${MODEL:-$ROOT/kernel-rollout/branchc-cumem/models/Qwen3-0.6B}
CYCLES=${CYCLES:-20}
SLEEP_LEVEL=${SLEEP_LEVEL:-1}
GMU=${GMU:-0.7}
RELOAD_WEIGHTS=${RELOAD_WEIGHTS:-0}   # set 1 for the escalation arm
EVID=$ROOT/kernel-rollout/branchc-cumem/evidence
# This worktree's probe script, mounted read-only into the container.
PROBE=${PROBE:-$(cd "$(dirname "$0")" && pwd)/cumem_cycle_probe.py}

test -r "$IMAGE"
test -f "$MODEL/config.json"
test -f "$PROBE"
mkdir -p "$EVID"

RELOAD_FLAG=""
if [ "$RELOAD_WEIGHTS" = "1" ]; then RELOAD_FLAG="--reload-weights"; fi

for ARM_CONF in "A0:expandable_segments:True" "A1:expandable_segments:False"; do
  ARM="${ARM_CONF%%:*}"
  CONF="${ARM_CONF#*:}"
  echo "=================== ARM $ARM  PYTORCH_CUDA_ALLOC_CONF=$CONF ==================="
  srun --export=ALL \
    --container-image="$IMAGE" \
    --container-mounts=/lustre:/lustre \
    /bin/bash --noprofile --norc -c "
      set -euo pipefail
      export PYTHONNOUSERSITE=1
      export PYTORCH_CUDA_ALLOC_CONF='$CONF'
      python3 '$PROBE' \
        --model '$MODEL' \
        --cycles '$CYCLES' \
        --sleep-level '$SLEEP_LEVEL' \
        --gpu-mem-util '$GMU' \
        $RELOAD_FLAG \
        --csv '$EVID/${SLURM_JOB_ID}-$ARM.csv'
    "
done

echo "=== per-cycle curves written to $EVID/${SLURM_JOB_ID}-A0.csv and -A1.csv ==="
echo "=== analyze: compare device_used_MiB slope across cycles A0 vs A1 ==="

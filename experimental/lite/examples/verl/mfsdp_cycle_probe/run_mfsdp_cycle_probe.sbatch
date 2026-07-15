#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Branch B M-FSDP training-side per-cycle retention probe (TASK-1.13.8.2).
# One node, NGPUS GPUs, runs the 2x2 matrix {mfsdp,fsdp2} x expandable{True,False}
# on a ~0.7B qwen3_moe proxy. Budget: NGPUS=2 x <=30 min = <=1 GPU-h (AC#4 <=2).
# The FSDP DoubleBuffer export mechanism is per-rank, so dp=2 reproduces it; bump
# NGPUS=8 to match the validated proxy topology if a wider check is wanted.
#
# Submit from cw (cw-dfw-cs-001-login-02) after staging PROBE_DIR onto lustre:
#   sbatch --export=ALL,NGPUS=2 run_mfsdp_cycle_probe.sbatch
#
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH -J branchb-mfsdp-cycle-probe
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code/branchb-mfsdp/evidence/%x_%j.out
set -euo pipefail

BASE=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code
IMG=${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}
PROBE_DIR=${PROBE_DIR:-$BASE/branchb-mfsdp/mfsdp_cycle_probe}
export BASE PROBE_DIR
export MLITE=${MLITE:-$BASE/qwen35_dapo_mfsdp_62295f9b3/mlite-ad1364f76}
export CP_SITE=${CP_SITE:-$BASE/mlite-newenv-cache/qwen35-cp-overlay-20260613/site}
export MODEL=${MODEL:-$BASE/models/Qwen3-30B-A3B}
export NGPUS=${NGPUS:-2}
export CYCLES=${CYCLES:-20}
export SEQ_LEN=${SEQ_LEN:-512}
export TRUNCATE_LAYERS=${TRUNCATE_LAYERS:-2}
export KEEP_EXPERTS=${KEEP_EXPERTS:-4}
export EVID=$BASE/branchb-mfsdp/evidence/${SLURM_JOB_ID}
export CACHE=$BASE/branchb-mfsdp/cache/${SLURM_JOB_ID}
mkdir -p "$EVID" "$CACHE"

test -r "$IMG"                       || { echo "FATAL: image not readable: $IMG" >&2; exit 2; }
test -f "$MODEL/config.json"         || { echo "FATAL: model missing: $MODEL" >&2; exit 2; }
test -f "$PROBE_DIR/run_mfsdp_cycle_probe_inner.sh" || { echo "FATAL: staged probe dir missing: $PROBE_DIR" >&2; exit 2; }

echo "job=$SLURM_JOB_ID nodes=$SLURM_NNODES ngpus=$NGPUS model=$MODEL evid=$EVID"
srun --container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL \
  bash "$PROBE_DIR/run_mfsdp_cycle_probe_inner.sh"
echo "=== done; evidence in $EVID ==="

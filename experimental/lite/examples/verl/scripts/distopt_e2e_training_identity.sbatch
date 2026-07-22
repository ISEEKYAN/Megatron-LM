#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH -J distopt_e2e_muon
set -euo pipefail

# =============================================================================
# End-to-end TRAINING bitwise identity: mcore-native vs MLite-DistOpt Muon.
# (bayan 2026-07-22 07:51 decisive gate). Runs a REAL training step through both
# stacks -- fwd -> bwd -> DDP grad reduce -> DistributedOptimizer master-grad ->
# finalize -> muon.step() -- and compares trained weights with torch.equal.
# 4 GPUs => TP=2, DP=2 (exercises DDP grad all-reduce + DistOpt master-grad
# sharding across DP + distributed cross-TP muon Newton-Schulz).
#
# Env recipe reused verbatim from tp_distributed_muon_identity.sbatch (same
# container / mcore / emerging_optimizers / PYTHONPATH).
# =============================================================================

BASE="${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}"
IMG="${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}"
MLITE_REPO="${MLITE_REPO:?set MLITE_REPO to the staged mlite checkout}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$BASE/runtime/ds4-csacp-parity-eaa5b486d/mcore}"
EMERGING_OPT_SITE="${EMERGING_OPT_SITE:?set EMERGING_OPT_SITE to a site with emerging_optimizers}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT for outputs}"
NGPU="${NGPU:-4}"
TP_SIZE="${TP_SIZE:-2}"

mkdir -p "$RUN_ROOT/logs"
echo "DISTOPT_E2E job=${SLURM_JOB_ID:-none} ngpu=$NGPU tp=$TP_SIZE mcore=$MEGATRON_ROOT emerging=$EMERGING_OPT_SITE"

SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
export BASE IMG MLITE_REPO MEGATRON_ROOT EMERGING_OPT_SITE RUN_ROOT NGPU TP_SIZE

srun --nodes=1 --ntasks=1 "${SA[@]}" bash -c '
set -euo pipefail
export PATH="$(printf %s "$PATH" | tr : "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1
MLITE_LITE="$MLITE_REPO/experimental/lite"
export PYTHONPATH="$EMERGING_OPT_SITE:/vllm:$MLITE_LITE:$MEGATRON_ROOT:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export TP_SIZE="$TP_SIZE"
export PYTHONPYCACHEPREFIX="/tmp/pycache-${SLURM_JOB_ID:-x}"
echo "DISTOPT_E2E_PREFLIGHT mlite=$(git -C "$MLITE_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) mcore=$(git -C "$MEGATRON_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
python3 -c "import megatron.core, megatron.lite; from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_stack, finalize_dist_opt_grads; from megatron.core.optimizer import get_megatron_optimizer; print(\"IMPORT_OK\", megatron.core.__file__)"
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" \
  "$MLITE_LITE/examples/verl/scripts/distopt_e2e_training_identity.py"
'
rc=$?
echo "DISTOPT_E2E_DONE job=${SLURM_JOB_ID:-none} rc=$rc"
exit "$rc"

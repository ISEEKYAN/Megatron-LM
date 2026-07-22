#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH -J muon_tp_identity
set -euo pipefail

# =============================================================================
# REAL multi-GPU (TP=2) Megatron-native vs DistOpt-mlite Muon identity receipt.
# Drives Megatron-Core's native TensorParallelMuon with a REAL TP process group
# (tp_mode=distributed -> cross-rank all_reduce in newton_schulz_tp) and compares
# the MLite-lowered config vs the native config with torch.equal on the sharded
# weights. See tp_distributed_muon_identity.py header for the full argument.
#
# Env recipe reused from three_arm_precision_node.sh (same container / PYTHONPATH).
# =============================================================================

BASE="${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/bayan/code}"
IMG="${IMG:-$BASE/verl_optimize/verl.vllm023.sqsh}"
MLITE_REPO="${MLITE_REPO:?set MLITE_REPO to the staged mlite checkout}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$BASE/runtime/ds4-csacp-parity-eaa5b486d/mcore}"
EMERGING_OPT_SITE="${EMERGING_OPT_SITE:?set EMERGING_OPT_SITE to a site with emerging_optimizers}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT for outputs}"
NGPU="${NGPU:-2}"

mkdir -p "$RUN_ROOT/logs"
echo "MUON_TP_IDENTITY job=${SLURM_JOB_ID:-none} ngpu=$NGPU mcore=$MEGATRON_ROOT emerging=$EMERGING_OPT_SITE"

SA=(--container-image="$IMG" --container-mounts=/lustre:/lustre --export=ALL)
export BASE IMG MLITE_REPO MEGATRON_ROOT EMERGING_OPT_SITE RUN_ROOT NGPU

srun --nodes=1 --ntasks=1 "${SA[@]}" bash -c '
set -euo pipefail
export PATH="$(printf %s "$PATH" | tr : "\n" | grep -viE "miniforge|/conda|/anaconda" | paste -sd: -)"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1
MLITE_LITE="$MLITE_REPO/experimental/lite"
export PYTHONPATH="$EMERGING_OPT_SITE:/vllm:$MLITE_LITE:$MEGATRON_ROOT:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPYCACHEPREFIX="/tmp/pycache-${SLURM_JOB_ID:-x}"
echo "MUON_TP_IDENTITY_PREFLIGHT mlite=$(git -C "$MLITE_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) mcore=$(git -C "$MEGATRON_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
python3 -c "import megatron.core, megatron.lite; from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon; from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp; print(\"IMPORT_OK\", megatron.core.__file__)"
torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" \
  "$MLITE_LITE/examples/verl/scripts/tp_distributed_muon_identity.py"
'
rc=$?
echo "MUON_TP_IDENTITY_DONE job=${SLURM_JOB_ID:-none} rc=$rc"
exit "$rc"

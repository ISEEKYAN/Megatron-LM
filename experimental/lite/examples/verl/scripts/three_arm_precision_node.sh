#!/usr/bin/env bash
# In-container driver for one precision arm.  Sets the verified mlite/verl env
# (reused from qwen35_dapo_mfsdp_62295f9b3/run_dapo_h100_node_d62c5aa46.sh) and
# invokes the ready-made verl SFT launcher run_qwen3moe_sft.sh.  SFT needs no ray
# / vllm rollout, so this is a plain single-node torchrun.
set -euo pipefail

# --- strip host conda / user site so the container's python + TE win ----------
export PATH="$(printf %s "$PATH" | tr : '\n' | grep -viE 'miniforge|/conda|/anaconda' | paste -sd: -)"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1

MLITE_LITE="$MLITE_REPO/experimental/lite"
export VERL_ROOT="$VERL"
# megatron.core (with the Muon optimizer) comes from the NVIDIA Megatron-Core
# checkout; megatron.lite comes from the task's mlite fork.  run_qwen3moe_sft.sh
# prepends MEGATRON_ROOT last, so its megatron.core wins over the fork's.
export MEGATRON_ROOT="${MEGATRON_ROOT:?set MEGATRON_ROOT to an NVIDIA mcore that ships megatron/core/optimizer/muon.py}"
export PYTHONPATH="/vllm:$MLITE_LITE/examples/verl:$MLITE_LITE:$VERL:$MEGATRON_ROOT:${PYTHONPATH:-}"
# Megatron-Core's dist_opt Muon requires the external `emerging_optimizers` package
# (the actual Newton-Schulz kernels), which the verl.vllm023 container does NOT ship.
# Point EMERGING_OPT_SITE at a site-packages dir that provides it (e.g. a pip
# --target install) so the muon arms can construct the optimizer.
if [[ -n "${EMERGING_OPT_SITE:-}" ]]; then
  export PYTHONPATH="$EMERGING_OPT_SITE:$PYTHONPATH"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export HYDRA_FULL_ERROR=1
JOBTAG="${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-0}-$ARM"
export TRITON_CACHE_DIR="/tmp/triton-$JOBTAG"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-$JOBTAG"
export PYTHONPYCACHEPREFIX="/tmp/pycache-$JOBTAG"
export HF_HOME="$RUN_ROOT/hf-home"
export HF_DATASETS_CACHE="$RUN_ROOT/hf-datasets-cache"
export VERL_MLITE_CACHE_ROOT="/tmp/verl_mlite-$JOBTAG"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$HF_HOME" "$HF_DATASETS_CACHE"

echo "Q35_PRECISION_PREFLIGHT arm=$ARM mlite=$(git -C "$MLITE_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) mcore=$(git -C "$MEGATRON_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown) verl=$(git -C "$VERL" rev-parse --short HEAD 2>/dev/null || echo unknown)"
python3 - <<'PY'
# Namespace merge + optimizer-availability preflight: megatron.core (NVIDIA, with
# muon) and megatron.lite (fork) must both import, and the Muon getter must exist.
import megatron.core  # noqa: F401
import megatron.lite  # noqa: F401
from megatron.core.optimizer.muon import get_megatron_muon_optimizer  # noqa: F401
from megatron.lite.primitive.optimizers import muon_routing  # noqa: F401
import verl_mlite.engine.mlite_engine  # noqa: F401
print("Q35_PRECISION_IMPORT_OK core=", megatron.core.__file__)
PY

# --- fixed contract knobs forwarded to run_qwen3moe_sft.sh --------------------
export MODEL_PATH TRAIN_FILES OUTPUT_ROOT
export SEED TP_SIZE PP_SIZE EP_SIZE ETP_SIZE CP_SIZE
export TOTAL_STEPS TRAIN_BATCH_SIZE MICRO_BATCH_SIZE MAX_TOKENS_PER_GPU MAX_LENGTH
export LR MIN_LR LR_WARMUP_STEPS LR_DECAY_STYLE WEIGHT_DECAY CLIP_GRAD
export OPTIMIZER_ALGORITHM="$OPT_ALGO"
export MLITE_OPTIMIZER_BACKEND="$OPT_BACKEND"
export MUON_TP_MODE="$MUON_TP_MODE"
# Megatron-Core's precision-aware optimizer (fp8/fp16 optimizer state) is adam-only
# (`--use-precision-aware-optimizer only supported with adam`).  For the Muon arms we
# keep optimizer offload for 35B memory but disable precision-aware state; this is an
# optimizer-implementation detail, not a training hyperparameter, so the same-contract
# (model/data/init/seed/tokens/LR) still holds across arms.
if [[ "$OPT_ALGO" == "muon" ]]; then
  export USE_PRECISION_AWARE_OPTIMIZER=False
fi
export TOTAL_EPOCHS=1
export SAVE_FREQ="$TOTAL_STEPS"
export TEST_FREQ=-1
export LOAD_HF_WEIGHTS=True
export RUN_NAME="q35_precision_${ARM}"
export PROJECT_NAME="verl-mlite-muon-precision"
export JSONL_FILE="$OUTPUT_ROOT/${RUN_NAME}.jsonl"
export LOG_FILE="$OUTPUT_ROOT/${RUN_NAME}.log"

bash "$MLITE_LITE/examples/verl/scripts/run_qwen3moe_sft.sh"

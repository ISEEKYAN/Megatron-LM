#!/usr/bin/env bash
# In-container driver: prove Qwen3.5-35B GatedDeltaNet + use_thd=True forward
# produces a loss once the FLA-enabled overlay (CP_SITE) is on the path.
#
# This is the three-arm precision node recipe (TASK-1.13.5.5.3) with the ONE
# fix it was missing: the CP_SITE (fla + tilelang + tvm_ffi) wiring reused
# verbatim from the known-good DAPO node script
# (qwen35_dapo_mfsdp_62295f9b3/run_dapo_h100_node_d62c5aa46.sh).  Arm is pinned
# to adamw/dist_opt so the FLA fix is isolated from the separate muon /
# emerging_optimizers path.
set -euo pipefail

# --- strip host conda / user site so the container python + TE win -----------
export PATH="$(printf %s "$PATH" | tr : '\n' | grep -viE 'miniforge|/conda|/anaconda' | paste -sd: -)"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES || true
export PYTHONNOUSERSITE=1

MLITE_LITE="$MLITE_REPO/experimental/lite"
export VERL_ROOT="$VERL"
export MEGATRON_ROOT="${MEGATRON_ROOT:?set MEGATRON_ROOT to an NVIDIA mcore that ships megatron/core}"

# ---- the FLA-enabled overlay (fla + tilelang + tvm_ffi) ---------------------
# CP_SITE prepended just after /vllm so the container's own vllm still wins, but
# fla/tilelang/tvm_ffi become importable.  Without this the GatedDeltaNet packed
# THD path raises "GatedDeltaNet packed THD requires FLA causal conv."
: "${CP_SITE:?set CP_SITE to the qwen35 FLA overlay site (fla+tilelang+tvm_ffi)}"
export PYTHONPATH="/vllm:$CP_SITE:$MLITE_LITE/examples/verl:$MLITE_LITE:$VERL:$MEGATRON_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$CP_SITE/tvm_ffi/lib:${LD_LIBRARY_PATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export HYDRA_FULL_ERROR=1
JOBTAG="${SLURM_JOB_ID:-x}-${SLURMD_NODENAME:-0}-fla_thd"
export TRITON_CACHE_DIR="/tmp/triton-$JOBTAG"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-$JOBTAG"
export PYTHONPYCACHEPREFIX="/tmp/pycache-$JOBTAG"
export HF_HOME="$RUN_ROOT/hf-home"
export HF_DATASETS_CACHE="$RUN_ROOT/hf-datasets-cache"
export VERL_MLITE_CACHE_ROOT="/tmp/verl_mlite-$JOBTAG"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$HF_HOME" "$HF_DATASETS_CACHE"

echo "FLA_THD_PREFLIGHT mlite=$(git -C "$MLITE_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) mcore=$(git -C "$MEGATRON_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown) cp_site=$CP_SITE"
python3 - <<'PY'
# FLA availability preflight: the exact imports the GatedDeltaNet THD path needs.
import fla
import tilelang
from fla.modules.convolution import causal_conv1d          # noqa: F401
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
from fla.modules.l2norm import l2norm                       # noqa: F401
import megatron.core   # noqa: F401
import megatron.lite   # noqa: F401
import verl_mlite.engine.mlite_engine  # noqa: F401
print("FLA_THD_IMPORT_OK fla=%s tilelang=%s" % (fla.__version__, tilelang.__version__))
PY

# --- fixed contract knobs: adamw / dist_opt, real weights, 2 steps -----------
export MODEL_PATH TRAIN_FILES OUTPUT_ROOT
export SEED="${SEED:-1234}"
export TP_SIZE="${TP_SIZE:-2}" PP_SIZE="${PP_SIZE:-1}" EP_SIZE="${EP_SIZE:-8}" ETP_SIZE="${ETP_SIZE:-1}" CP_SIZE="${CP_SIZE:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export LR="${LR:-1e-5}" MIN_LR="${MIN_LR:-1e-5}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2}" LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}" CLIP_GRAD="${CLIP_GRAD:-1.0}"
export OPTIMIZER_ALGORITHM=adamw
export MLITE_OPTIMIZER_BACKEND=dist_opt
export TOTAL_EPOCHS=1
export SAVE_FREQ="$TOTAL_STEPS"
export TEST_FREQ=-1
export LOAD_HF_WEIGHTS=True
export RUN_NAME="q35_fla_thd_smoke"
export PROJECT_NAME="verl-mlite-fla-thd-smoke"
export JSONL_FILE="$OUTPUT_ROOT/${RUN_NAME}.jsonl"
export LOG_FILE="$OUTPUT_ROOT/${RUN_NAME}.log"

bash "$MLITE_LITE/examples/verl/scripts/run_qwen3moe_sft.sh"

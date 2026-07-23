#!/usr/bin/env bash
# In-container driver for one AC#4 speed+memory arm (Qwen3-30B-A3B verl SFT).
set -euo pipefail

export PATH="$(printf %s "$PATH" | tr : '\n' | grep -viE 'miniforge|/conda|/anaconda' | paste -sd: -)"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CC CXX || true
export PYTHONNOUSERSITE=1
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1

MLITE_LITE="$MLITE_REPO/experimental/lite"
export VERL_ROOT="$VERL"
export MEGATRON_ROOT="${MEGATRON_ROOT:?set MEGATRON_ROOT}"
if [[ -n "${VERL_DSA_SITE:-}" ]]; then
  export PYTHONPATH="/vllm:${VERL_DSA_SITE}/nvidia_cutlass_dsl/python_packages:${VERL_DSA_SITE}:${MLITE_LITE}/examples/verl:${MLITE_LITE}:${VERL}:${EMERGING_OPT_ROOT:-}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="/vllm:${MLITE_LITE}/examples/verl:${MLITE_LITE}:${VERL}:${EMERGING_OPT_ROOT:-}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
fi
if [[ -n "${NVRX_VENV_SITE:-}" ]]; then
  export PYTHONPATH="${NVRX_VENV_SITE}:${PYTHONPATH}"
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

echo "Q3MOE_SPEED_MEM_PREFLIGHT arm=$ARM mlite=$(git -C "$MLITE_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) mcore=$(git -C "$MEGATRON_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown) verl=$(git -C "$VERL" rev-parse --short HEAD 2>/dev/null || echo unknown)"
python3 - <<'PY'
import megatron.core  # noqa: F401
import megatron.lite  # noqa: F401
from megatron.core.optimizer.muon import get_megatron_muon_optimizer  # noqa: F401
from megatron.lite.primitive.optimizers.megatron_wrap import build_dist_opt_optimizer_config
from megatron.lite.runtime.contracts.config import OptimizerConfig
import verl_mlite.engine.mlite_engine  # noqa: F401
core = build_dist_opt_optimizer_config(
    OptimizerConfig(optimizer_algorithm="muon", muon_tp_mode="distributed", lr=1e-5, weight_decay=0.1, clip_grad=1.0),
    complete_muon_lowering=True,
)
assert core.use_layer_wise_distributed_optimizer is True, core.use_layer_wise_distributed_optimizer
print("Q3MOE_SPEED_MEM_IMPORT_OK layer_wise_dist_opt=", core.use_layer_wise_distributed_optimizer)
PY

export MODEL_PATH TRAIN_FILES OUTPUT_ROOT
export SEED TP_SIZE PP_SIZE EP_SIZE ETP_SIZE CP_SIZE
export TOTAL_STEPS TRAIN_BATCH_SIZE MICRO_BATCH_SIZE MAX_TOKENS_PER_GPU MAX_LENGTH
export LR MIN_LR LR_WARMUP_STEPS LR_DECAY_STYLE WEIGHT_DECAY CLIP_GRAD
export OPTIMIZER_ALGORITHM="$OPT_ALGO"
export MLITE_OPTIMIZER_BACKEND="$OPT_BACKEND"
export MUON_TP_MODE="$MUON_TP_MODE"
if [[ "$OPT_ALGO" == "muon" ]]; then
  export USE_PRECISION_AWARE_OPTIMIZER=False
fi
export TOTAL_EPOCHS=1
export SAVE_FREQ="$TOTAL_STEPS"
export TEST_FREQ=-1
export LOAD_HF_WEIGHTS=True
export RUN_NAME="q3moe_speed_mem_${ARM}"
export PROJECT_NAME="verl-mlite-muon-speed-mem"
export JSONL_FILE="$OUTPUT_ROOT/${RUN_NAME}.jsonl"
export LOG_FILE="$OUTPUT_ROOT/${RUN_NAME}.log"

bash "$MLITE_LITE/examples/verl/scripts/run_qwen3moe_sft.sh"

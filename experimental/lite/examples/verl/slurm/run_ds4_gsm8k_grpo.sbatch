#!/usr/bin/env bash
# Generic single-node DeepSeek-V4 DAPO/GRPO submission runner.
# Submit through docs/runs/run_ds4_qat_proxy.sh; no site-specific paths belong
# in this file.
set -euo pipefail

: "${MLITE_SRC:?set MLITE_SRC to a Megatron-LM checkout}"
: "${MEGATRON_ROOT:?set MEGATRON_ROOT to a compatible MCore checkout}"
: "${VERL_ROOT:?set VERL_ROOT to a pinned VERL checkout}"
: "${DS4_VLLM_SITE:?set DS4_VLLM_SITE to the vLLM site-packages directory}"
: "${MODELOPT_SITE:?set MODELOPT_SITE to ModelOpt and cudnn frontend site-packages}"
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the DeepSeek-V4 checkpoint}"
: "${RUN_ROOT:?set RUN_ROOT to an empty writable output directory}"

ENABLE_QAT="${ENABLE_QAT:-False}"
ROLLOUT_WEIGHT_BITS="${ROLLOUT_WEIGHT_BITS:-8}"
ENABLE_R3="${ENABLE_R3:-False}"
VERL_COMMIT="${VERL_COMMIT:-b9c513c4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
RAY_OVERRIDE_JOB_RUNTIME_ENV="${RAY_OVERRIDE_JOB_RUNTIME_ENV:-1}"

export OMP_NUM_THREADS MKL_NUM_THREADS
export PYTHONPATH="${MODELOPT_SITE}:${DS4_VLLM_SITE}:${VERL_ROOT}:${MEGATRON_ROOT}:${MLITE_SRC}/experimental/lite/examples/verl:${MLITE_SRC}/experimental/lite:${MLITE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

# Ray workers receive only explicitly serialized values.  These switches must
# stay in the allowlist: omitting the first three produces a QAT-labelled job
# whose actor or rollout worker silently uses its default configuration.
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

names = (
    "ENABLE_QAT", "ROLLOUT_WEIGHT_BITS", "ENABLE_R3", "MODELOPT_SITE",
    "VERL_COMMIT", "RAY_OVERRIDE_JOB_RUNTIME_ENV", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
print(json.dumps({"env_vars": {name: os.environ[name] for name in names}}))
PY
)"
export RUNTIME_ENV_JSON

DATA_DIR="${DS4_SHARED_DATA:-${RUN_ROOT}/data}"
if [[ ! -f "${DATA_DIR}/dapo-math-17k.parquet" || ! -f "${DATA_DIR}/aime-2024.parquet" ]]; then
  mkdir -p "${DATA_DIR}"
  python3 "${VERL_ROOT}/examples/data_preprocess/dapo_math.py" \
    --local_save_dir "${DATA_DIR}"
fi

export MODEL_PATH="${CHECKPOINT_DIR}"
export TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/dapo-math-17k.parquet}"
export VAL_FILES="${VAL_FILES:-${DATA_DIR}/aime-2024.parquet}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/outputs}"
exec bash "${MLITE_SRC}/experimental/lite/examples/verl/scripts/run_deepseek_v4_dapo.sh"

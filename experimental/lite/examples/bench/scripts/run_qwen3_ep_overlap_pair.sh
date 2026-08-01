#!/usr/bin/env bash
set -euo pipefail

HF_PATH=${HF_PATH:?set HF_PATH to a HuggingFace Qwen3 MoE model directory}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/experimental/lite/examples/bench/outputs"}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-8}
DRY_RUN=${DRY_RUN:-1}
NSYS_PROFILE=${NSYS_PROFILE:-0}
CYCLES=${CYCLES:-10}
WARMUP=${WARMUP:-3}

export PYTHONPATH="${REPO_ROOT}/experimental/lite:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MEGATRON_LITE_MOE_PERMUTE_FUSION=0

run_arm() {
  local arm=$1
  local port=$2
  if [[ "${DRY_RUN}" == "1" ]]; then
    "${PYTHON_BIN}" \
      "${REPO_ROOT}/experimental/lite/examples/bench/qwen3_ep_overlap_probe.py" \
      --hf-path "${HF_PATH}" --config-only
    return
  fi

  mkdir -p "${OUTPUT_DIR}"
  local command=(
    torchrun --nproc_per_node "${NPROC}" --master_port "${port}"
    "${REPO_ROOT}/experimental/lite/examples/bench/qwen3_ep_overlap_probe.py"
    --hf-path "${HF_PATH}" --out-dir "${OUTPUT_DIR}"
    --cycles "${CYCLES}" --warmup "${WARMUP}" --arm "${arm}"
  )
  if [[ "${NSYS_PROFILE}" == "1" ]]; then
    nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
      --output "${OUTPUT_DIR}/qwen3_ep_${arm}" "${command[@]}" \
      2>&1 | tee "${OUTPUT_DIR}/qwen3_ep_${arm}.log"
  else
    "${command[@]}" 2>&1 | tee "${OUTPUT_DIR}/qwen3_ep_${arm}.log"
  fi
}

run_arm baseline "${MASTER_PORT_BASELINE:-31851}"
run_arm overlap "${MASTER_PORT_OVERLAP:-31852}"

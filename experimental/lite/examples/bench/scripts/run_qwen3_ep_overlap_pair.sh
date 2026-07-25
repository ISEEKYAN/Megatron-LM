#!/usr/bin/env bash
set -euo pipefail

HF_PATH=${HF_PATH:?set HF_PATH to a HuggingFace Qwen3 MoE model directory}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/experimental/lite/examples/bench/outputs"}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-8}
DRY_RUN=${DRY_RUN:-1}

export PYTHONPATH="${REPO_ROOT}/experimental/lite:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MEGATRON_LITE_MOE_PERMUTE_FUSION=0

COMMON_ARGS=(
  --backend mlite
  --hf-path "${HF_PATH}"
  --model-name qwen3_moe
  --tp "${TP:-1}"
  --etp "${ETP:-1}"
  --ep "${EP:-8}"
  --pp 1
  --cp 1
  --steps "${STEPS:-12}"
  --warmup "${WARMUP:-4}"
  --num-microbatches "${NUM_MICROBATCHES:-4}"
  --seq-len "${SEQ_LEN:-1024}"
  --truncate-layers "${TRUNCATE_LAYERS:-8}"
  --keep-experts "${KEEP_EXPERTS:-8}"
  --disable-mtp
  --same-data-across-dp
  --skip-load-hf-weights
)

run_arm() {
  local arm=$1
  local impl_cfg=$2
  local port=$3
  if [[ "${DRY_RUN}" == "1" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
      "${COMMON_ARGS[@]}" \
      --impl-cfg-json "${impl_cfg}" \
      --dry-run
    return
  fi

  mkdir -p "${OUTPUT_DIR}"
  torchrun --nproc_per_node "${NPROC}" --master_port "${port}" \
    "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    "${COMMON_ARGS[@]}" \
    --impl-cfg-json "${impl_cfg}" \
    --output-json "${OUTPUT_DIR}/qwen3_ep_${arm}.json" \
    2>&1 | tee "${OUTPUT_DIR}/qwen3_ep_${arm}.log"
}

run_arm baseline '{}' "${MASTER_PORT_BASELINE:-31851}"
run_arm overlap \
  '{"overlap_moe_expert_parallel_comm":true,"use_deepep":false,"recompute":[]}' \
  "${MASTER_PORT_OVERLAP:-31852}"

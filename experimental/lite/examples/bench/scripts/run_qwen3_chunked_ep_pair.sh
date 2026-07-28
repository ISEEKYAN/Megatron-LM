#!/usr/bin/env bash
set -euo pipefail

HF_PATH=${HF_PATH:?set HF_PATH to a HuggingFace Qwen3 MoE model directory}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/experimental/lite/examples/bench/outputs"}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-8}
DRY_RUN=${DRY_RUN:-1}

export PYTHONPATH="${REPO_ROOT}/experimental/lite:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

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
  --use-thd
  --same-data-across-dp
  --skip-load-hf-weights
)

run_arm() {
  local arm=$1
  local impl_cfg=$2
  local port=$3
  local correctness_port=$4
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
    --output-json "${OUTPUT_DIR}/qwen3_chunked_ep_${arm}.json" \
    2>&1 | tee "${OUTPUT_DIR}/qwen3_chunked_ep_${arm}.log"

  torchrun --nproc_per_node "${NPROC}" --master_port "${correctness_port}" \
    "${REPO_ROOT}/experimental/lite/examples/bench/correctness.py" run \
    "${COMMON_ARGS[@]}" \
    --impl-cfg-json "${impl_cfg}" \
    --steps "${CORRECTNESS_STEPS:-3}" \
    --warmup 0 \
    --num-microbatches "${CORRECTNESS_NUM_MICROBATCHES:-1}" \
    --seq-len "${CORRECTNESS_SEQ_LEN:-${SEQ_LEN:-1024}}" \
    $([[ "${CORRECTNESS_SKIP_WEIGHT_HASH:-0}" == "1" ]] && printf '%s' '--skip-weight-hash') \
    --output-json "${OUTPUT_DIR}/qwen3_chunked_ep_${arm}_correctness.json" \
    2>&1 | tee "${OUTPUT_DIR}/qwen3_chunked_ep_${arm}_correctness.log"
}

run_arm baseline \
  '{"use_deepep":true,"recompute":["moe"],"num_chunks_ep_a2a_overlap":1}' \
  "${MASTER_PORT_BASELINE:-31851}" \
  "${CORRECTNESS_PORT_BASELINE:-31951}"
run_arm chunked \
  '{"use_deepep":true,"recompute":["moe"],"num_chunks_ep_a2a_overlap":2}' \
  "${MASTER_PORT_CHUNKED:-31852}" \
  "${CORRECTNESS_PORT_CHUNKED:-31952}"

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/correctness.py" compare \
    "${OUTPUT_DIR}/qwen3_chunked_ep_baseline_correctness.json" \
    "${OUTPUT_DIR}/qwen3_chunked_ep_chunked_correctness.json" \
    --output-json "${OUTPUT_DIR}/qwen3_chunked_ep_correctness_compare.json" \
    --loss-atol "${CORRECTNESS_LOSS_ATOL:-0}" \
    --loss-rtol "${CORRECTNESS_LOSS_RTOL:-0}" \
    --grad-atol "${CORRECTNESS_GRAD_ATOL:-0}" \
    --grad-rtol "${CORRECTNESS_GRAD_RTOL:-0}" \
    --fail-on-mismatch
fi

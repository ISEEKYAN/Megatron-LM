#!/usr/bin/env bash
set -euo pipefail

HF_PATH=${HF_PATH:?set HF_PATH to a HuggingFace Qwen3.5 model directory}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/experimental/lite/examples/bench/outputs/tp_replication"}

export PYTHONPATH="${REPO_ROOT}/experimental/lite:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MEGATRON_LITE_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --hf-path "${HF_PATH}"
  --model-name qwen3_5
  --etp 1
  --ep 1
  --pp 1
  --cp 1
  --steps "${STEPS:-1}"
  --num-microbatches 1
  --seq-len "${SEQ_LEN:-32}"
  --seed "${SEED:-42}"
  --truncate-layers "${TRUNCATE_LAYERS:-4}"
  --disable-mtp
  --same-data-across-dp
  --no-optimizer
  --skip-optimizer-build
  --skip-weight-hash
  --impl-cfg-json '{"mount_vision_model": false}'
)

run_tp() {
  local tp=$1
  local output_json="${OUTPUT_DIR}/qwen35_mlite_tp${tp}.json"
  torchrun --standalone --nproc_per_node "${tp}" \
    "${REPO_ROOT}/experimental/lite/examples/bench/correctness.py" run \
    --backend mlite --tp "${tp}" "${COMMON_ARGS[@]}" \
    --output-json "${output_json}" \
    2>&1 | tee "${OUTPUT_DIR}/qwen35_mlite_tp${tp}.log"
}

run_tp 2
run_tp 4

"${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/correctness.py" compare \
  "${OUTPUT_DIR}/qwen35_mlite_tp2.json" \
  "${OUTPUT_DIR}/qwen35_mlite_tp4.json" \
  --output-json "${OUTPUT_DIR}/qwen35_mlite_tp2_vs_tp4.json" \
  --fail-on-mismatch \
  2>&1 | tee "${OUTPUT_DIR}/qwen35_mlite_tp2_vs_tp4.log"

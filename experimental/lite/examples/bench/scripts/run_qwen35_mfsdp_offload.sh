#!/usr/bin/env bash
set -euo pipefail

HF_PATH=${HF_PATH:?set HF_PATH to a HuggingFace Qwen3.5 model directory}
REPO_ROOT=${REPO_ROOT:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-8}
DRY_RUN=${DRY_RUN:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/experimental/lite/examples/bench/outputs/mfsdp_offload"}
OUTPUT_JSON="${OUTPUT_DIR}/qwen35_mfsdp_offload.json"

if [[ "${DRY_RUN}" != "1" && -z "${SLURM_JOB_ID:-}" ]]; then
  echo "M-FSDP offload benchmark must run inside a Slurm allocation." >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/experimental/lite:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  --backend mlite
  --hf-path "${HF_PATH}"
  --model-name qwen3_5
  --tp "${TP:-1}"
  --etp "${ETP:-1}"
  --ep "${EP:-1}"
  --pp "${PP:-1}"
  --cp "${CP:-1}"
  --steps "${STEPS:-15}"
  --warmup "${WARMUP:-5}"
  --num-microbatches "${NUM_MICROBATCHES:-4}"
  --seq-len "${SEQ_LEN:-1024}"
  --truncate-layers "${TRUNCATE_LAYERS:-8}"
  --keep-experts "${KEEP_EXPERTS:-8}"
  --disable-mtp
  --same-data-across-dp
  --impl-cfg-json '{"optimizer":"mfsdp"}'
  --override-optimizer-json '{"offload_fraction":1.0,"use_precision_aware_optimizer":false}'
)

if [[ "${SKIP_LOAD_HF_WEIGHTS:-1}" == "1" ]]; then
  ARGS+=(--skip-load-hf-weights)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
    "${ARGS[@]}" --dry-run
  exit 0
fi

: "${WANDB_PROJECT:?set WANDB_PROJECT for GPU benchmark evidence}"
mkdir -p "${OUTPUT_DIR}"
torchrun --nproc_per_node "${NPROC}" \
  "${REPO_ROOT}/experimental/lite/examples/bench/bench.py" \
  "${ARGS[@]}" --output-json "${OUTPUT_JSON}" \
  2>&1 | tee "${OUTPUT_DIR}/qwen35_mfsdp_offload.log"

OUTPUT_JSON="${OUTPUT_JSON}" WANDB_ENTITY="${WANDB_ENTITY:-megatron-core-moe-dev}" \
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-mfsdp-offload-${SLURM_JOB_ID}}" \
  "${PYTHON_BIN}" -c '
import json
import os

import wandb

artifact = json.load(open(os.environ["OUTPUT_JSON"], encoding="utf-8"))
summary = artifact["summary"]
run = wandb.init(
    entity=os.environ["WANDB_ENTITY"],
    project=os.environ["WANDB_PROJECT"],
    name=os.environ["WANDB_RUN_NAME"],
    config=artifact["result"],
)
run.log(
    {
        "benchmark/step_s": summary["avg_step_ms"] / 1000.0,
        "benchmark/optimizer_step_s": summary["avg_optimizer_step_ms"] / 1000.0,
        "benchmark/tokens_per_s": summary["tok_per_s"],
        "benchmark/peak_mem_gb": summary["peak_mem_gb"],
    }
)
print(f"[MLITE_BENCH_WANDB] {run.url}", flush=True)
run.finish()
'

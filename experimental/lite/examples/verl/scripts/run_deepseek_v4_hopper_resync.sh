#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"
HOPPER_SMOKE="${SCRIPT_DIR}/run_deepseek_v4_hopper_smoke.sh"

: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the official mixed DS4 checkpoint}"
: "${MLITE_COMMIT:?set MLITE_COMMIT to the committed validation source}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a fresh result directory}"
: "${MLITE_SM90_SITE:?set MLITE_SM90_SITE to the SM90 training overlay}"
: "${MEGATRON_ROOT:?set MEGATRON_ROOT to a compatible MCore checkout}"
: "${VERL_ROOT:?set VERL_ROOT to the VERL source checkout}"
: "${DS4_VLLM_SITE:?set DS4_VLLM_SITE to the DS4 vLLM thin overlay}"
: "${DS4_VLLM_SHIM:?set DS4_VLLM_SHIM to the Torch/vLLM ABI shim}"

[[ -d "${CHECKPOINT_DIR}" ]] || {
  echo "CHECKPOINT_DIR is not a directory: ${CHECKPOINT_DIR}" >&2
  exit 2
}
[[ -f "${CHECKPOINT_DIR}/config.json" ]] || {
  echo "missing official checkpoint config.json" >&2
  exit 2
}
[[ -f "${CHECKPOINT_DIR}/model.safetensors.index.json" ]] || {
  echo "missing official checkpoint model.safetensors.index.json" >&2
  exit 2
}
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${MLITE_COMMIT}" ]] || {
  echo "MLite source is not at ${MLITE_COMMIT}" >&2
  exit 7
}
[[ ! -e "${OUTPUT_DIR}" ]] || {
  echo "refusing to reuse existing output directory: ${OUTPUT_DIR}" >&2
  exit 5
}

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  CHECK_ONLY=1 "${HOPPER_SMOKE}" training
  CHECK_ONLY=1 "${HOPPER_SMOKE}" rollout-probe
  echo "DS4_HOPPER_RESYNC_ENV_CHECK_PASSED"
  exit 0
fi

COMMON_PYTHONPATH="${MLITE_SM90_SITE}/nvidia_cutlass_dsl/python_packages:${MLITE_SM90_SITE}:${VERL_ROOT}:${MEGATRON_ROOT}:${EXAMPLE_ROOT}:${LITE_ROOT}:${REPO_ROOT}"
export PYTHONPATH="${COMMON_PYTHONPATH}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}" "${HOPPER_SMOKE}" training
CUDA_VISIBLE_DEVICES=0 python -m examples.verl.ds4_hopper_resync_proxy \
  --checkpoint "${CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --steps "${TRAIN_STEPS:-4}" \
  --learning-rate "${LEARNING_RATE:-1e-2}"
"${HOPPER_SMOKE}" rollout-probe

for artifact in \
  "${OUTPUT_DIR}/report.json" \
  "${OUTPUT_DIR}/bf16-trained.pt" \
  "${OUTPUT_DIR}/fp8-export.pt" \
  "${OUTPUT_DIR}/DS4_HOPPER_RESYNC_PROXY_COMPLETE"; do
  [[ -s "${artifact}" ]] || {
    echo "missing non-skip Hopper resync artifact: ${artifact}" >&2
    exit 6
  }
done

python - "${OUTPUT_DIR}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["source"]["expert_dtype"] == "fp4"
assert report["export"]["target_expert_dtype"] == "fp8"
assert report["export"]["expert_weight_dtype"] == "torch.float8_e4m3fn"
assert report["gate"]["acceptable"] is True
assert report["logprobs"]["fp32"]["clipping_boundary_crossings"] == 0
assert report["training"]["loss_trace"][-1] <= report["training"]["loss_trace"][0]
assert report["training"]["param_delta_sum"] > 0
PY

echo "DS4_HOPPER_TRAIN_RESYNC_PASSED output=${OUTPUT_DIR}"

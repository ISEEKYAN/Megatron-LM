#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "training" && "${MODE}" != "rollout-probe" ]]; then
  echo "usage: $0 {training|rollout-probe}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
EXAMPLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"
LITE_ROOT="$(cd "${EXAMPLE_ROOT}/../.." && pwd -L)"
REPO_ROOT="$(cd "${LITE_ROOT}/../.." && pwd -L)"

require_directory() {
  local variable_name="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "${variable_name} is not a directory: ${path}" >&2
    exit 2
  fi
}

require_file() {
  local variable_name="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "${variable_name} is not a file: ${path}" >&2
    exit 2
  fi
}

require_package_metadata() {
  local site="$1"
  local distribution_glob="$2"
  if ! compgen -G "${site}/${distribution_glob}" >/dev/null; then
    echo "missing ${distribution_glob} under ${site}" >&2
    exit 2
  fi
}

: "${MLITE_SM90_SITE:?set MLITE_SM90_SITE to the SM90 training overlay site-packages}"
: "${MEGATRON_ROOT:?set MEGATRON_ROOT to a compatible Megatron Core source checkout}"
: "${VERL_ROOT:?set VERL_ROOT to the VERL source checkout}"
require_directory MLITE_SM90_SITE "${MLITE_SM90_SITE}"
require_directory MEGATRON_ROOT "${MEGATRON_ROOT}"
require_directory MEGATRON_ROOT/megatron/core "${MEGATRON_ROOT}/megatron/core"
require_directory VERL_ROOT "${VERL_ROOT}"
require_package_metadata "${MLITE_SM90_SITE}" 'flash_mla-*.dist-info'
require_package_metadata "${MLITE_SM90_SITE}" 'nvidia_cudnn_frontend-*.dist-info'
require_package_metadata "${MLITE_SM90_SITE}" 'nvidia_cutlass_dsl-*.dist-info'

COMMON_PYTHONPATH="${MLITE_SM90_SITE}/nvidia_cutlass_dsl/python_packages:${MLITE_SM90_SITE}:${VERL_ROOT}:${MEGATRON_ROOT}:${EXAMPLE_ROOT}:${LITE_ROOT}:${REPO_ROOT}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONNOUSERSITE=1

case "${MODE}" in
  training)
    # Keep the vLLM thin overlay out of the training process: it overrides
    # transformers and other packages that the validated SM90 stack owns.
    export PYTHONPATH="${COMMON_PYTHONPATH}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"
    export MLITE_RUN_SMOKE=1
    COMMAND=(
      torchrun
      --standalone
      "--nproc_per_node=${NPROC_PER_NODE:-2}"
      -m pytest -q
      "${LITE_ROOT}/tests/unit/verl/test_mlite_engine_cp_smoke.py"
      -k deepseek_v4
      -s -rs -vv
    )
    ;;
  rollout-probe)
    : "${DS4_VLLM_SITE:?set DS4_VLLM_SITE to the DeepSeek-V4 vLLM thin overlay site-packages}"
    : "${DS4_VLLM_SHIM:?set DS4_VLLM_SHIM to the torch/vLLM ABI compatibility library}"
    require_directory DS4_VLLM_SITE "${DS4_VLLM_SITE}"
    require_file DS4_VLLM_SHIM "${DS4_VLLM_SHIM}"
    require_package_metadata "${DS4_VLLM_SITE}" 'apache_tvm_ffi-*.dist-info'
    require_package_metadata "${DS4_VLLM_SITE}" 'tilelang-*.dist-info'
    require_package_metadata "${DS4_VLLM_SITE}" 'transformers-*.dist-info'
    require_package_metadata "${DS4_VLLM_SITE}" 'vllm-*.dist-info'

    export PYTHONPATH="${DS4_VLLM_SITE}:${COMMON_PYTHONPATH}${EXTRA_PYTHONPATH:+:${EXTRA_PYTHONPATH}}"
    export LD_PRELOAD="${DS4_VLLM_SHIM}${EXTRA_LD_PRELOAD:+:${EXTRA_LD_PRELOAD}}"
    export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    COMMAND=(python "${EXAMPLE_ROOT}/deepseek_v4_hopper_vllm_probe.py")
    ;;
esac

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH}"
  if [[ "${MODE}" == "rollout-probe" ]]; then
    printf 'LD_PRELOAD=%s\n' "${LD_PRELOAD}"
  fi
  echo "DS4_HOPPER_ENV_CHECK_PASSED mode=${MODE}"
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH}"
  if [[ "${MODE}" == "rollout-probe" ]]; then
    printf 'LD_PRELOAD=%s\n' "${LD_PRELOAD}"
  fi
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
